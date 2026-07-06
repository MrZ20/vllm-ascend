# 004. Qwen3.5 forward 和 decoder layer 源码

本章从 `NPUModelRunner._model_forward()` 进入真实模型，解释 Qwen3.5 一层是怎么跑的。

## 1. 上游模型决定“层长什么样”

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:130
```

Qwen3.5 decoder layer 根据 `layer_type` 选择：

```python
if self.layer_type == "linear_attention":
    self.linear_attn = QwenGatedDeltaNetAttention(...)
elif self.layer_type == "full_attention":
    self.self_attn = Qwen3NextAttention(...)
```

本模型配置：

```text
30 层 linear_attention
10 层 full_attention
```

所以不是每层都走标准 full attention。

## 2. MoE 替代普通 dense MLP

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:147
```

关键判断：

```python
if config.model_type == "qwen3_5_moe_text":
    self.mlp = Qwen3NextSparseMoeBlock(...)
```

本模型：

```text
model_type = qwen3_5_moe_text
```

所以 decoder layer 的 FFN 部分不是普通 MLP，而是 sparse MoE。

## 3. Ascend patch 替换 layer forward

源码：

```text
vllm_ascend/patch/worker/patch_qwen3_5.py:195
```

关键代码：

```python
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
```

这意味着：

```text
上游 qwen3_5.py 定义结构
Ascend patch 接管关键 forward 实现
```

读源码时要同时看这两个文件，不然会误以为运行时走的是上游原始 forward。

## 4. Ascend decoder layer 的执行顺序

源码：

```text
vllm_ascend/patch/worker/patch_qwen3_5.py:90
```

核心顺序：

```python
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

if self.layer_type == "linear_attention":
    self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
elif self.layer_type == "full_attention":
    self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)

hidden_states = self_attention_output
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
return hidden_states, residual
```

用公式写就是：

```text
u = RMSNorm(x)
a = AttentionOrLinearAttention(u)
x1 = residual_add(a, x)
v = RMSNorm(x1)
m = MoE(v)
out = residual_add(m, x1)
```

代码中 residual add 被融合在 RMSNorm 模块的返回值里，所以看起来不像单独的 `x + ...`。

## 5. full attention patch 做了什么

源码：

```text
vllm_ascend/patch/worker/patch_qwen3_5.py:41
```

关键代码：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(...)
attn_output = self.attn(q, k, v)
gate = torch.sigmoid(gate)
attn_output = attn_output * gate
output[:], _ = self.o_proj(attn_output)
```

逐步解释：

```text
1. qkv_proj: hidden state 一次投影出 Q/K/V/gate 所需数据。
2. triton_split_qkv_rmsnorm_mrope: 拆 Q/K/V，做 Q/K RMSNorm，做 M-RoPE。
3. self.attn(q,k,v): 调 Ascend attention 后端，使用 KV cache 和 metadata。
4. sigmoid(gate): attention 输出门控。
5. o_proj: attention 输出投影回 hidden_size。
```

标准 attention 公式：

```text
score = Q K^T / sqrt(d_head) + causal_mask
prob = softmax(score)
attn_out = prob V
```

这里又多了一个输出门控：

```text
attn_out = attn_out * sigmoid(gate)
```

## 6. linear attention 分支

如果 `layer_type == "linear_attention"`，代码走：

```python
self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
```

这个分支对应 Qwen Gated DeltaNet / linear attention。它不是简单的 `QK^T` 全量注意力，而是更偏状态递推的注意力形态。

对源码阅读来说，先记住：

```text
full_attention 分支依赖 Q/K/V + KV cache
linear_attention 分支依赖 GDN 状态式计算
二者都返回 [N, hidden_size]，后面统一进入 MoE
```

## 7. MoE block 从哪里进入

decoder layer 最后：

```python
hidden_states = self.mlp(hidden_states)
```

由于 `self.mlp = Qwen3NextSparseMoeBlock`，所以进入上游：

```text
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:195
```

关键代码：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)
```

这里产生下一章的核心数据：

```text
router_logits shape = [N, 256]
```

其中 `256` 是专家数。

后续 `self.experts(...)` 会进入 Ascend FusedMoE 和 W8A8 动态量化路径。
