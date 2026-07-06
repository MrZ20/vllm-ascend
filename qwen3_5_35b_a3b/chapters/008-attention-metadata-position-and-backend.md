# 008. Attention metadata、positions 和后端选择

本章回答：为什么模型 forward 前要构造一大堆 metadata？为什么 trace 里会看到 `positions`、`query_start_loc`、`slot_mapping`、`block_table` 这些不像“神经网络层”的东西？

先说大白话：attention 不是只需要 Q、K、V 三个矩阵。推理时还必须知道每个 token 属于哪个 request、在序列里的第几个位置、历史 K/V 存在 KV cache 的哪个槽位、哪些 token 可以互相看见。attention metadata 就是把这些“索引和规则”整理成 kernel 能直接使用的数据。

## 1. `positions` 从哪里来

在 `_prepare_inputs()` 中，位置计算的核心公式是：

```text
position = num_computed_tokens + query_pos
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:834`
- `vllm_ascend/worker/model_runner_v1.py:839`

对本次 prompt 来说，prefill 前：

```text
num_computed_tokens = 0
query_pos = [0, 1, 2, 3, 4]
positions = [0, 1, 2, 3, 4]
```

decode 第一步时，prompt 的 5 个 token 已经算过：

```text
num_computed_tokens = 5
query_pos = [0]
positions = [5]
```

decode 第二步：

```text
num_computed_tokens = 6
query_pos = [0]
positions = [6]
```

所以 position 不是“当前 batch 中第几个 token”，而是“这个 request 的序列位置”。

## 2. 为什么有时 `positions` 是二维或三维概念

本模型配置里有：

```text
rope_parameters = {
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "partial_rotary_factor": 0.25
}
```

runner 里如果 `uses_mrope` 为真，会在 `_prepare_inputs()` 中调用：

```text
_calc_mrope_positions(scheduler_output)
```

并在 dummy run / graph capture 或 forward 时使用：

```python
positions = self.mrope_positions.gpu[:, :num_tokens_padded]
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:1017`
- `vllm_ascend/worker/model_runner_v1.py:3578`

这就是为什么你可能会看到类似：

```text
positions shape = [3, N]
```

其中 `N` 是本步 token 数，padding 后可能是 8 或 1；前面的 `3` 对应 M-RoPE 的三个 position 分量。纯文本输入时，这些位置分量仍然服务于 RoPE 计算；多模态输入时，它们还能表达文本、图像、视频等不同轴上的位置关系。

本测试是纯文本，但模型配置仍然带 M-RoPE 参数，所以不要把 `positions` 简化成永远 `[N]`。

## 3. RoPE 到底对 Q/K 做什么

RoPE 的作用是把位置信息注入 Q 和 K。它不是把 position 加到 hidden state 上，而是对 Q/K 的部分维度做旋转。

二维成对维度的简化公式：

```text
q_even' = q_even * cos(theta_pos) - q_odd * sin(theta_pos)
q_odd'  = q_even * sin(theta_pos) + q_odd * cos(theta_pos)
```

K 也做同样旋转：

```text
k_even' = k_even * cos(theta_pos) - k_odd * sin(theta_pos)
k_odd'  = k_even * sin(theta_pos) + k_odd * cos(theta_pos)
```

这样 attention score：

```text
score(i, j) = Q_i' dot K_j'
```

会自然包含 token i 和 token j 的相对位置信息。

Ascend runner 在 forward 前调用：

```text
update_cos_sin(positions)
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2286`
- `vllm_ascend/worker/model_runner_v1.py:3586`

这一步会让后续 fused RoPE / attention kernel 能拿到当前 batch 所需的 cos/sin。

## 4. `query_start_loc` 表示什么

attention kernel 需要知道一个扁平 token batch 里，每个 request 的 query 从哪里开始。

假设一个 batch 里有三个 request，本步 token 数分别是：

```text
[5, 1, 3]
```

那么：

```text
query_start_loc = [0, 5, 6, 9]
```

含义是：

```text
request 0 的 token 范围 = [0, 5)
request 1 的 token 范围 = [5, 6)
request 2 的 token 范围 = [6, 9)
```

本次只有一个 request：

```text
prefill query_start_loc 逻辑上是 [0, 5]
decode  query_start_loc 逻辑上是 [0, 1]
```

如果 graph capture 要 padding 到 8，运行时还可能插入 dummy request 或 padding 边界，让 kernel 看到固定形状。这不代表真实 prompt 变长了。

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:979`
- `vllm_ascend/worker/model_runner_v1.py:1010`

## 5. `block_table` 和 `slot_mapping` 为什么必要

decode 阶段不能每次重新计算全部历史 token，所以需要 KV cache。

KV cache 可以理解成：

```text
每层 attention 都有一块缓存
缓存里存历史 token 的 K 和 V
```

但缓存不是无限长数组随便 append。vLLM 用 block 管理缓存。于是需要两个映射：

```text
block_table: request -> 它占用了哪些 KV block
slot_mapping: 当前 token -> 写入哪个 KV cache slot
```

当 prefill 处理 5 个 prompt token 时，attention 计算出这些 token 的 K/V，并根据 `slot_mapping` 写入 KV cache。

当 decode 处理第 6 个位置的 token 时，新 token 的 K/V 写入新的 slot，同时 attention kernel 通过 `block_table` 找到历史 5 个 token 的 K/V。

这就是为什么 decode 输入只有 1 个 token，却仍然能“看见”完整 prompt。

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:810`
- `vllm_ascend/worker/model_runner_v1.py:858`
- `vllm_ascend/worker/model_runner_v1.py:3193`
- `vllm_ascend/worker/model_runner_v1.py:3194`

## 6. causal mask 在 metadata 里怎么体现

自回归模型要求第 i 个 token 只能看见自己和之前的 token，不能偷看未来。

数学上 full attention 是：

```text
score = Q K^T / sqrt(d_head) + mask
prob = softmax(score)
out = prob V
```

causal mask 是：

```text
mask[i, j] = 0      if j <= i
mask[i, j] = -inf   if j > i
```

真实推理 kernel 通常不会真的构造一个巨大的 `[seq_len, seq_len]` mask 矩阵，因为那太浪费。它通过 metadata 里的：

```text
seq_lens
query_start_loc
num_computed_tokens
is_prefilling
causal=True
```

在 kernel 内部推导哪些位置合法。

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:3175`
- `vllm_ascend/worker/model_runner_v1.py:3188`
- `vllm_ascend/worker/model_runner_v1.py:3196`
- `vllm_ascend/worker/model_runner_v1.py:3197`

## 7. full attention 和 linear attention 的 metadata 差异

本模型有：

```text
30 层 linear_attention
10 层 full_attention
```

full attention 的核心依赖是历史 K/V，所以它强依赖 KV cache block、slot、seq length。

linear attention 或 Gated DeltaNet 一类状态式注意力，不一定使用同一种完整 K/V 访问方式。它可能维护的是递推状态或压缩状态。尽管如此，runner 仍然要告诉后端：

```text
当前是 prefill 还是 decode
本步有多少 token
每个 request 的位置和长度
padding 到了多少
哪些层共享同一组 metadata
```

`_build_attention_metadata()` 会按 KV cache group 和 attention group 构造 metadata，并把同组 layer 绑定到对应 metadata。

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:3046`
- `vllm_ascend/worker/model_runner_v1.py:3284`
- `vllm_ascend/worker/model_runner_v1.py:3324`

## 8. `set_ascend_forward_context` 的作用

构造完 metadata 后，runner 进入：

```python
set_ascend_forward_context(...)
```

它把当前 forward 的关键信息放到 Ascend forward context 里：

```text
attn_metadata
vllm_config
num_tokens_padded
num_actual_tokens
aclgraph_runtime_mode
batch_descriptor
model_instance
input_ids
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2299`

为什么需要全局 context？因为后续 attention、MoE、量化 kernel 可能不是由 Python 显式传入所有参数，而是在调用栈深处通过 context 读取当前 batch 的运行信息。

这也是硬件插件常见的设计：模型 forward 看起来像普通 PyTorch module 调用，但底层算子需要额外运行时上下文。

## 9. padding 后怎么保证不采样假 token

本次 prompt 真实长度是 5，但 prefill 被 padding 到 8：

```text
真实 token 数 = 5
num_tokens_padded = 8
```

模型 forward 可能在固定形状 `[8, hidden_size]` 上运行。但最终只从真实 token 对应的位置取 hidden state 做 logits。

runner 中有：

```text
logits_indices
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2375`

所以 padding token 的存在是为了满足运行时形状，不是为了让模型多生成 token。

## 10. 本章小结

你可以把 attention metadata 记成一句话：

```text
metadata 告诉 kernel：这些扁平 token 属于谁、在第几个位置、历史 K/V 在哪里、哪些位置可以看、padding 到哪里结束。
```

没有 metadata，Q/K/V 的矩阵乘本身仍然能算，但它不知道 request 边界、cache 位置和 causal 规则，就会把不同 request 或不同位置混在一起，生成结果自然错误。

下一章会解释这些 metadata 和 padding 为什么又会跟 TP、内存 profiling、KV cache 分配、graph capture 绑在一起。
