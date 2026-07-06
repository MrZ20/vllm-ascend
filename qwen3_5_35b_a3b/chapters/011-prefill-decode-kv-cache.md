# 011. Prefill、Decode 和 KV Cache

本章回答：为什么第一次模型 step 处理 5 个 token，后面每次只处理 1 个 token？

先说大白话：prefill 是“把用户给的整段 prompt 读一遍，并建立历史缓存”；decode 是“每次只看刚生成的一个新 token，同时查历史缓存，继续生成下一个 token”。KV cache 的作用就是把历史 token 的 K/V 保存起来，避免每生成一个字都把整段 prompt 重新算一遍。

## 1. 本次动态追踪证据

第一个真实用户请求 step：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=5))
NPUModelRunner._prepare_inputs(... ndarray(shape=(1,), dtype=int32, head=[5]))
NPUModelRunner._preprocess(... num_tokens_padded=8)
```

解释：

- 当前只有 1 个 request。
- prompt token 数是 5。
- 所以 scheduler 本轮安排了 5 个 token。
- runner 为图捕获对齐到 8。

后续 decode step：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=1))
NPUModelRunner._prepare_inputs(... head=[1])
NPUModelRunner._preprocess(... Tensor(shape=(1, 2048), dtype=torch.bfloat16, device=npu:0))
NPUModelRunner._sample(logits=Tensor(shape=(1, 248320), ...))
```

解释：

- prefill 之后，历史信息已经进入 cache。
- decode 每次只安排新 token。
- 每次仍然产出 `[1, 248320]` logits，用来选择下一个 token。

## 2. scheduler 的 `total_num_scheduled_tokens`

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2042`
- `vllm_ascend/worker/model_runner_v1.py:2101`
- `vllm_ascend/worker/model_runner_v1.py:2111`

关键代码意图：

```text
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
logits_indices, spec_decode_metadata, total_num_scheduled_tokens = self._prepare_inputs(...)
```

`scheduler_output.total_num_scheduled_tokens` 是这一轮 worker 要真正送进模型的 token 总数。

本测试只有一个 request，所以它很直观：

```text
prefill: total_num_scheduled_tokens = 5
decode:  total_num_scheduled_tokens = 1
```

如果有很多 request，这个数会是多个 request 本轮 token 数的总和。vLLM 的 scheduler 会把不同请求混成 batch，提高吞吐。

## 3. 为什么 prefill 必须处理整个 prompt

输入 prompt：

```text
[9419, 11, 821, 803, 369]
```

自回归语言模型要预测第 6 个 token，必须让第 5 个位置的 hidden state 能看到前面所有 token：

```text
h_5 = Transformer(token_1, token_2, token_3, token_4, token_5)
logits_6 = LMHead(h_5)
token_6 = argmax(logits_6)
```

在 full attention 层里，第 5 个 token 的 Q 会关注所有历史 K/V：

```text
score_5,j = q_5 dot k_j / sqrt(d_head), j <= 5
prob_5 = softmax(score_5)
out_5 = sum_j prob_5,j * v_j
```

所以第一次必须把 prompt 的每个 token 都变成 K/V，写入 KV cache。

## 4. KV cache 保存的是什么

full attention 每层都会产生 K/V：

```text
K = X W_K
V = X W_V
```

KV cache 保存的是每层、每个历史 token 的 K 和 V。抽象形状可以理解为：

```text
K_cache[layer, token_position, kv_head, head_dim]
V_cache[layer, token_position, kv_head, head_dim]
```

本模型 full attention 配置：

```text
num_key_value_heads = 2
head_dim = 256
```

因此每个 full attention 层、每个 token 至少需要缓存：

```text
K: 2 * 256
V: 2 * 256
```

再乘 dtype 字节数、block 管理开销、batch 数和最大上下文长度，KV cache 会成为推理服务里非常重要的一块显存/显存等价资源。

注意：本模型还有 30 层 `linear_attention`。这类层不完全使用传统 full attention 的 K/V cache 形式，而会维护自己的状态或 mamba/gated delta net 相关缓存。runner 里也能看到对 mamba cache 的处理分支：

```text
if self.cache_config.mamba_cache_mode == "align":
    mamba_utils.preprocess_mamba(...)
```

但本章核心先抓住 full attention 的 KV cache，因为 decode 加速的基本思想最清楚。

## 5. Decode 为什么只喂 1 个 token

prefill 后，已有：

```text
K_cache[1..5]
V_cache[1..5]
```

生成第 6 个 token 后，下一轮 decode 的输入只是：

```text
token_6
```

模型只需要计算这个新 token 的：

```text
q_6, k_6, v_6
```

然后：

```text
K_cache[6] = k_6
V_cache[6] = v_6
out_6 = attention(q_6, K_cache[1..6], V_cache[1..6])
```

这样就不用重新计算 token 1 到 token 5 的 K/V，也不用重新跑它们在每层的 projection。

如果没有 KV cache，每生成一个 token 都要重新处理整个前缀：

```text
生成第 6 个：算 5 个 prompt token
生成第 7 个：重新算 6 个 token
生成第 8 个：重新算 7 个 token
...
```

有 KV cache 后：

```text
生成第 6 个：prefill 算 5 个 token
生成第 7 个：decode 算 1 个 token
生成第 8 个：decode 算 1 个 token
...
```

这就是推理系统能服务长上下文和多轮输出的核心优化。

## 6. padding 和 CUDA graph / ACL graph

测试参数：

```python
cudagraph_capture_sizes=[1, 2, 4, 8]
```

虽然这里叫 `cudagraph`，在 Ascend 侧会映射到对应的图捕获/静态执行思路。日志里也看到：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 4/4
Capturing CUDA graphs (decode, FULL): 4/4
```

真实 prompt 有 5 个 token，但 5 不在 `[1, 2, 4, 8]` 中，所以运行时选择能容纳 5 的捕获大小 8：

```text
num_tokens_unpadded = 5
num_tokens_padded = 8
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2133`
- `vllm_ascend/worker/model_runner_v1.py:2159`
- `vllm_ascend/worker/model_runner_v1.py:2172`

padding 的目的不是改变语义，而是让算子输入形状落到少数固定形状上。固定形状更容易复用已捕获的执行图，减少 Python 调度、kernel launch、图构建等开销。

## 7. attention metadata 是什么

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2249`

模型 forward 不只需要 `input_ids` 和 `positions`，还需要 attention 后端知道：

- 当前有几个 token。
- 每个 request 的 query 起止位置。
- KV cache 中哪些 block 属于哪个 request。
- 当前是 prefill 还是 decode。
- padding 后哪些位置是真 token，哪些是补齐 token。
- logits 应该从哪些 token 位置取。

这些信息被打包成 attention metadata：

```text
attn_metadata = self._build_attention_metadata(...)
```

然后传给 Ascend forward context：

```text
set_ascend_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
    input_ids=input_ids,
    ...
)
```

这一步很关键：Qwen3.5 的 attention 模块最终调用的是 Ascend 后端 attention kernel，kernel 必须知道 cache、mask、positions 和 batch 布局。

## 8. positions 的作用

动态追踪里 `_preprocess` 返回过：

```text
positions = Tensor(shape=(3, 8), dtype=torch.int64, device=npu:0)
```

为什么是 3 行？因为这个模型配置里 rope 使用了 mRoPE：

```text
mrope_section = [11, 11, 10]
mrope_interleaved = true
```

对于纯文本，三行位置通常等价或由文本位置展开；对于多模态模型，三行可对应时间、高度、宽度等位置轴。Qwen3.5 patch 里 full attention 会拿：

```text
cos_sin = self.rotary_emb.cos_sin_cache[positions]
```

再把它传给融合拆分 QKV/RMSNorm/RoPE 的算子。

## 9. 本测试的 step 序列

本测试 `max_tokens=5`，因此语义上的用户请求过程是：

```text
Step 0 prefill:
  input = [9419, 11, 821, 803, 369]
  output next = 498

Step 1 decode:
  input = [498]
  output next = 7525

Step 2 decode:
  input = [7525]
  output next = 3855

Step 3 decode:
  input = [3855]
  output next = 1089

Step 4 decode:
  input = [1089]
  output next = 321
```

最终拼接：

```text
[9419, 11, 821, 803, 369] + [498, 7525, 3855, 1089, 321]
```

## 10. 初学者检查点

1. prefill 和 decode 的输入 token 数为什么不同？
2. KV cache 保存 K/V，而不是保存最终文本，这句话为什么重要？
3. padding 到 8 会不会让模型多生成 3 个 token？
4. attention metadata 为什么不能省略？
5. 为什么 decode 阶段仍然能看到整个历史？

答案要点：

- prefill 建历史，decode 追加新 token。
- K/V 是每层 attention 需要的中间表示，不是用户可读文本。
- padding 只是计算形状对齐，真实 token 数仍由 metadata 和 mask 控制。
- kernel 需要 metadata 才知道 cache、mask、位置和 batch 布局。
- decode 用新 Q 查询历史 K/V cache。
