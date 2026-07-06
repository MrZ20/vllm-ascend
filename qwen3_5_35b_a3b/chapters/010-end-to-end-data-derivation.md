# 010. 从输入到输出的完整数据推导

本章回答一个最朴素的问题：你输入 `"Hello, my name is"`，模型为什么能生成 `" [Your Name], and"`？

先说大白话：模型不是直接“理解一句话然后写一句话”。它每次只做一件事：根据当前已有 token，给词表里每一个可能的下一个 token 打分，然后选一个 token 接到末尾。这个动作重复 5 次，就得到 5 个新 token。

本次实测结果：

```text
prompt text = "Hello, my name is"
prompt token ids = [9419, 11, 821, 803, 369]
generated token ids = [498, 7525, 3855, 1089, 321]
final token ids = [9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321]
final text = "Hello, my name is [Your Name], and"
```

## 1. 文本先变成 token ID

源码入口：

- `tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py:24`
- `tests/e2e/conftest.py:1003`
- `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:523`

测试文件里只有一个 prompt：

```python
EXAMPLE_PROMPTS = [
    "Hello, my name is",
]
```

`VllmRunner.generate_greedy()` 会调用 `get_inputs()`。如果输入是字符串，`get_inputs()` 创建的是：

```text
TextPrompt(prompt="Hello, my name is", multi_modal_data=None)
```

进入 vLLM offline engine 后，tokenizer 把字符串切成 token ID。本次动态追踪看到：

```text
prompt_token_ids = [9419, 11, 821, 803, 369]
```

你可以把 token ID 理解成词表编号，但不要把它简单等同于英文单词。一个 token 可能是一个词、一个空格加词、一段标点，甚至是词的一部分。模型真正吃进去的是整数 ID，不是 Python 字符串。

## 2. token ID 变成 embedding 向量

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:228`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:329`

模型配置事实：

```text
vocab_size = 248320
hidden_size = 2048
```

embedding 表可以看成一个大矩阵：

```text
E shape = [248320, 2048]
```

每个 token ID 是这个表的一行索引：

```text
X_0 = E[token_ids]
```

对于 prompt 的 5 个 token：

```text
token_ids shape = [5]
X_0 shape = [5, 2048]
```

在实际 Ascend runner 里，为了匹配图捕获大小，prefill 这一步被 padding 到 8 个 token。动态追踪里看到：

```text
NPUModelRunner._preprocess(... num_tokens_padded=8)
inputs_embeds = Tensor(shape=(8, 2048), dtype=torch.bfloat16, device=npu:0)
```

这里的 `8` 不是 prompt 真的有 8 个 token，而是运行时为了复用固定形状图，把 5 个真实 token 对齐到了 `cudagraph_capture_sizes=[1, 2, 4, 8]` 里的 `8`。

## 3. embedding 进入 40 层模型

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:240`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:90`

模型配置事实：

```text
num_hidden_layers = 40
layer_types_count = {"linear_attention": 30, "full_attention": 10}
```

也就是说，输入 embedding 会依次经过 40 个 decoder layer。每层大致做：

```text
hidden -> RMSNorm -> attention/linear_attention -> residual
       -> RMSNorm -> MoE -> residual
```

更贴近本仓库 patch 的执行顺序：

```text
if residual is None:
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
else:
    hidden_states, residual = input_layernorm(hidden_states, residual)

hidden_states = self_attention_or_linear_attention(hidden_states)
hidden_states, residual = post_attention_layernorm(hidden_states, residual)
hidden_states = mlp(hidden_states)  # 对 qwen3_5_moe_text 来说，这是 Sparse MoE
return hidden_states, residual
```

这里的 `mlp` 不是普通 dense MLP，而是 `Qwen3NextSparseMoeBlock`。原因在上游模型初始化中：

```text
if config.model_type == "qwen3_5_moe_text":
    self.mlp = Qwen3NextSparseMoeBlock(...)
```

本模型的 `model_type` 正是：

```text
qwen3_5_moe_text
```

## 4. 每层 full attention 的数学过程

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:225`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:41`

如果当前 layer 是 `full_attention`，hidden state 会先通过融合的 QKV projection：

```text
qkv, _ = self.qkv_proj(hidden_states)
```

然后 Ascend patch 对 q/k/v 做拆分、Q/K RMSNorm、RoPE：

```text
q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(...)
attn_output = self.attn(q, k, v)
gate = sigmoid(gate)
attn_output = attn_output * gate
output[:], _ = self.o_proj(attn_output)
```

核心数学公式：

```text
Q = X W_Q
K = X W_K
V = X W_V
score = Q K^T / sqrt(d_head) + causal_mask
prob = softmax(score)
attn_out = prob V
```

本模型：

```text
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
```

这说明它使用 GQA，也就是 Q head 多，KV head 少。多个 Q head 共享一组 K/V head，能减少 KV cache 体积和带宽。

## 5. 每层 MoE 的数学过程

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:101`
- `vllm_ascend/ops/fused_moe/experts_selector.py:30`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py:122`

模型配置事实：

```text
num_experts = 256
num_experts_per_tok = 8
moe_intermediate_size = 512
shared_expert_intermediate_size = 512
```

对每个 token 的 hidden state `x`，router 先打分：

```text
router_logits = x W_router
router_logits shape = [N, 256]
```

然后选 top-8 个专家：

```text
topk_weights shape = [N, 8]
topk_ids shape = [N, 8]
```

每个专家本质是一个 SwiGLU MLP：

```text
Expert_e(x) = W_down_e( silu(W_gate_e x) * (W_up_e x) )
```

一个 token 的 MoE 输出：

```text
MoE(x) = sum_{i=1..8} topk_weights_i * Expert_{topk_ids_i}(x)
```

这就是 `A3B` 的核心直觉：模型总参数很多，因为有 256 个专家；但每个 token 只走 8 个专家，所以单 token 激活的参数量远小于总参数量。

## 6. 最后一层 hidden state 变成 logits

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2375`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:353`

40 层结束后，runner 不需要所有 token 的 logits。对于自回归生成，只需要“最后一个可采样位置”的 hidden state。

代码里是：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

数学上：

```text
h_last shape = [1, 2048]
W_vocab shape = [248320, 2048]
logits = h_last @ W_vocab^T
logits shape = [1, 248320]
```

动态追踪证据：

```text
NPUModelRunner._sample(logits=Tensor(shape=(1, 248320), dtype=torch.bfloat16, device=npu:0))
Sampler.sample(logits=Tensor(shape=(1, 248320), dtype=torch.float32, device=npu:0))
```

`248320` 就是词表大小。此时模型给每个可能 token 一个分数。

## 7. Greedy 选择下一个 token

源码入口：

- `tests/e2e/conftest.py:1083`
- `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:240`
- `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:243`

测试调用：

```python
SamplingParams(temperature=0.0, max_tokens=5)
```

`temperature=0.0` 表示 greedy。greedy 的核心公式：

```text
next_token = argmax(logits)
```

也就是不随机、不抽样，直接选分数最大的 token。

本次连续 5 次选择得到：

```text
[498, 7525, 3855, 1089, 321]
```

解码回文本是：

```text
" [Your Name], and"
```

所以最终文本是：

```text
"Hello, my name is" + " [Your Name], and"
= "Hello, my name is [Your Name], and"
```

## 8. 一次生成循环的完整闭环

把所有阶段串起来：

```text
Python 字符串
  -> tokenizer
token IDs [5]
  -> embedding
hidden states [5, 2048]
  -> padding / positions / attention metadata
hidden states [8, 2048]
  -> 40 layers: attention/linear_attention + MoE
hidden states [8, 2048]
  -> logits_indices 取最后可采样位置
sample hidden [1, 2048]
  -> lm_head
logits [1, 248320]
  -> argmax
next token id
  -> append 到 request
  -> 下一轮 decode
```

prefill 产生第 1 个新 token；后续 decode 每轮产生 1 个新 token。`max_tokens=5`，所以这个闭环总共为用户请求产生 5 个新 token。

## 9. 初学者检查点

如果你能回答下面问题，说明本章已经读通：

1. 模型为什么输出的是 token，而不是直接输出字符串？
2. 为什么 logits 的形状是 `[1, 248320]`？
3. 为什么 prompt 有 5 个 token，trace 里却看到 `inputs_embeds shape=(8, 2048)`？
4. 为什么 `A3B` 和 MoE 有关系？
5. greedy sampling 和 softmax 随机采样有什么区别？

答案要点：

- 字符串只是人类接口，模型内部处理 token ID。
- `248320` 是词表大小，每个候选 token 一个分数。
- 运行时为了匹配图捕获和 kernel 形状做了 padding。
- MoE 总专家很多，但每个 token 只激活 top-k 专家。
- greedy 直接取最大 logit，不按概率随机抽。
