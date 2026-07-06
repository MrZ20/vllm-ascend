# 009. 并行、内存和图捕获

本章回答三个现实问题：

```text
为什么可见设备是 4 张，但只看到 TP0/TP1？
为什么真实 prompt 是 5 个 token，却 padding 到 8？
为什么真实 prompt 前还有 8192、8、4 这些 dummy forward？
```

先说大白话：大模型推理不只是数学公式，还要让巨大的权重、KV cache、通信和硬件 kernel 在有限 NPU 内存里稳定工作。TP 决定计算怎么拆，KV cache 决定历史怎么存，图捕获决定常见形状怎么复用。

## 1. 可见设备和实际使用设备不是一回事

你设置：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
```

这表示容器进程可以看见 4 张物理 NPU。

测试设置：

```python
tensor_parallel_size=2
```

这表示本次模型使用 2 个 tensor parallel rank。

因此逻辑关系是：

```text
可见设备池 = 4 张
本次 TP world size = 2
worker 进程数约等于 TP rank 数 = 2
```

日志里看到：

```text
Worker_TP0
Worker_TP1
```

并不表示 `ASCEND_RT_VISIBLE_DEVICES` 只生效了两张卡，而是因为本测试只要求 TP=2。

## 2. TP=2 在矩阵乘里怎么拆

Transformer 里最贵的计算通常是矩阵乘：

```text
Y = X W
```

Tensor parallel 会把大矩阵拆到多个 rank 上。

### Column parallel

按输出列拆权重：

```text
W = [W_0, W_1]
```

两个 rank 分别计算：

```text
Y_0 = X W_0
Y_1 = X W_1
```

如果下一层需要完整输出，就把结果拼起来：

```text
Y = concat(Y_0, Y_1)
```

### Row parallel

按输入维度拆权重：

```text
X = [X_0, X_1]
W = [W_0; W_1]
```

两个 rank 分别计算局部结果：

```text
Z_0 = X_0 W_0
Z_1 = X_1 W_1
```

最终需要求和：

```text
Y = Z_0 + Z_1
```

这个求和通常对应 `all-reduce`。

所以 TP=2 会引入通信。Ascend 场景中通信由 HCCL 负责，这也是测试设置 `HCCL_BUFFSIZE=1024` 的原因。

## 3. EP=false 表示什么

测试里：

```python
enable_expert_parallel=False
```

这表示不按 expert 维度做 expert parallel。

对于 MoE，理论上可以让不同 rank 拥有不同 expert，然后 token 根据 router 结果被发送到对应 expert rank。这叫 EP，常见通信模式是 all-to-all。

但本测试关闭 EP，所以重点是：

```text
TP 拆 tensor
不按 expert 拆 expert
MoE 通信走默认兼容路径
```

源码地图里对应的默认路径包括：

- `vllm_ascend/ops/fused_moe/token_dispatcher.py:342`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py:122`

更深的 MoE 数学和 dispatch/combine 在 `chapters/013-fusedmoe-router-experts.md`。

## 4. `gpu_memory_utilization=0.9` 如何影响 KV cache

测试配置：

```python
gpu_memory_utilization=0.9
```

worker 初始化时会记录启动时的 NPU 内存快照，并计算：

```text
requested_memory = total_memory * gpu_memory_utilization
```

源码入口：

- `vllm_ascend/worker/worker.py:467`
- `vllm_ascend/worker/worker.py:472`

然后通过 dummy forward 做 profiling，估算：

```text
non_kv_cache_memory = weights_memory + activation_peak + non_torch_memory
```

再得到可用于 KV cache 的空间：

```text
available_kv_cache_memory_bytes
  = requested_memory
    - non_kv_cache_memory
    - npugraph_memory_estimate_if_enabled
```

源码入口：

- `vllm_ascend/worker/worker.py:558`
- `vllm_ascend/worker/worker.py:590`
- `vllm_ascend/worker/worker.py:613`

这一步解释了为什么真实 prompt 前会先有 dummy run：系统要知道除了权重和激活，剩下多少内存能安全分配给 KV cache。

## 5. KV cache 为什么那么重要

prefill 后，历史 token 的 K/V 会被缓存。

对某一层 full attention 来说，decode 第 t 步只输入一个新 token：

```text
q_t = x_t W_Q
k_t = x_t W_K
v_t = x_t W_V
```

但是它计算 attention 时要看历史：

```text
K_cache = [k_0, k_1, ..., k_t]
V_cache = [v_0, v_1, ..., v_t]
```

于是：

```text
out_t = softmax(q_t K_cache^T / sqrt(d_head)) V_cache
```

如果没有 KV cache，每次 decode 都要重新计算：

```text
token 0 到 token t 的所有 K/V
```

这会让生成第 1000 个 token 时重复计算前 999 个 token，速度不可接受。

KV cache 的代价是显存。它大致随下面这些量增长：

```text
层数
KV head 数
head_dim
最大序列长度
并发 request 数
dtype 字节数
```

本模型的 full attention 层数不是 40 全部，而是其中 10 层；但还有 30 层 linear attention/GDN 相关状态。不同层的缓存形态不完全一样，所以 vLLM 通过 KV cache spec 和 metadata 来统一管理。

## 6. 图捕获为什么需要固定形状

测试设置：

```python
cudagraph_capture_sizes=[1, 2, 4, 8]
```

虽然日志里沿用了 `CUDA graph` 这个名字，在 Ascend 场景可以理解为类似的图捕获机制：把某些常见输入形状的执行过程录下来，后续遇到同样形状就复用。

图捕获喜欢固定形状，因为动态图形状会让运行时不断重新调度、编译或选择 kernel。

本次 prompt 长度是 5。捕获大小里没有 5，但有 8，所以 runner 会把本步 padding 到 8：

```text
真实 token 数 = 5
图执行 token 数 = 8
padding 数 = 3
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2956`
- `vllm_ascend/worker/model_runner_v1.py:2989`
- `vllm_ascend/worker/model_runner_v1.py:2998`

动态追踪证据：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=5))
NPUModelRunner._preprocess(... num_tokens_padded=8)
inputs_embeds shape = (8, 2048)
```

decode 阶段真实 token 数是 1，刚好命中 capture size 1：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=1))
inputs_embeds shape = (1, 2048)
```

## 7. padding 为什么不改变输出

padding 会改变 kernel 看到的张量形状，但不会改变 request 的真实语义。

关键有三层保护：

第一，scheduler 仍然记录真实 token 数：

```text
scheduler_output.total_num_scheduled_tokens = 5
```

第二，attention metadata 记录真实 request 长度、真实 sequence length、真实 slot mapping。

第三，计算 logits 时只取真实要采样的位置：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = compute_logits(sample_hidden_states)
```

因此 padding 的 3 个位置只是为了让硬件执行图形状对齐，不会参与最终采样。

## 8. 为什么真实 prompt 前有 8192、8、4

动态追踪中看到：

```text
_model_forward(num_tokens_padded=8192)
_model_forward(num_tokens_padded=8)
_model_forward(num_tokens_padded=4)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 4/4
Capturing CUDA graphs (decode, FULL): 4/4
```

这些不是用户 prompt。

它们来自初始化阶段：

```text
determine_available_memory()
  -> model_runner.profile_run()
  -> profile_cudagraph_memory()

compile_or_warm_up_model()
  -> _dummy_run(size)
  -> capture_model()
```

源码入口：

- `vllm_ascend/worker/worker.py:530`
- `vllm_ascend/worker/worker.py:564`
- `vllm_ascend/worker/worker.py:774`
- `vllm_ascend/worker/worker.py:800`

这些 dummy forward 的目的有三个：

- 估算激活峰值；
- 估算或分配图捕获内存；
- 预热常见 batch/token 形状，减少真实请求第一次运行的开销。

所以看到 8192 不要误解为 prompt 有 8192 token。它是运行时为了 profiling 和 warmup 构造的假输入规模。

## 9. FULL 和 PIECEWISE graph 的直觉

日志中有：

```text
mixed prefill-decode, PIECEWISE
decode, FULL
```

直觉上可以这样理解：

```text
FULL:
  尽可能把整个固定形状 forward 捕获成完整图。

PIECEWISE:
  对复杂或混合场景，只捕获部分稳定片段。
```

decode 通常形状更规整：每个 request 一次生成 1 个 token，所以更容易做 FULL capture。

prefill 或 mixed prefill-decode 更复杂：不同 request 的 prompt 长度可能不同，metadata 更复杂，所以更常见的是 PIECEWISE。

本次只有一个 prompt，但框架仍按通用服务场景初始化。

## 10. 本章小结

这章把几个容易混淆的数字分开：

```text
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
  -> 可见物理设备池，不等于一定全部使用

tensor_parallel_size=2
  -> 本次使用两个 TP rank，所以看到 Worker_TP0 / Worker_TP1

prompt token 数 = 5
  -> 语义上的真实 token 数

num_tokens_padded = 8
  -> 为图捕获和固定形状做的运行时 padding

dummy forward 8192 / 8 / 4
  -> 初始化 profiling、warmup、graph capture，不是用户请求
```

从下一章开始，就可以进入真正的数学主线：prompt token 如何一路经过 embedding、40 层 Qwen3.5、MoE、W8A8 matmul、LM head，最后变成输出 token。
