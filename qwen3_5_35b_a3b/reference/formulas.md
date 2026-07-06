# 公式速查

这份文件只放本次推理链路里真正会用到的公式。符号默认含义：

- `N`：本步 token 数。
- `H`：hidden size，本模型是 `2048`。
- `V`：词表大小，本模型是 `248320`。
- `E`：专家数，本模型是 `256`。
- `K`：每个 token 选择的专家数，本模型是 `8`。
- `d_head`：attention 每个 head 的维度，本模型是 `256`。

## Token Embedding

输入 token ID：

```text
ids = [9419, 11, 821, 803, 369]
```

查 embedding 表：

```text
X_0 = Embedding(ids)
X_0 shape = [N, H]
```

本次 prefill：`N=5`，图捕获对齐后实际 forward 可 padding 到 `8`。

## RMSNorm

RMSNorm 不减均值，只按均方根缩放：

```text
rms(x) = sqrt(mean(x_i^2) + eps)
RMSNorm(x)_i = x_i / rms(x) * gamma_i
```

对一行 hidden state `x in R^H`：

```text
y = x / sqrt((1/H) * sum_j x_j^2 + eps) * gamma
```

本模型 `eps=1e-6`。

## Full Attention

投影：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

按 head reshape 后：

```text
Q shape = [N, num_q_heads, d_head]
K shape = [N, num_kv_heads, d_head]
V shape = [N, num_kv_heads, d_head]
```

本模型总 `num_q_heads=16`，`num_kv_heads=2`，`d_head=256`。这是 GQA：多个 Q head 共享较少的 KV head。

RoPE 旋转：

```text
RoPE(q_pos) = rotate(q, position)
RoPE(k_pos) = rotate(k, position)
```

注意力分数：

```text
score = Q K^T / sqrt(d_head) + causal_mask
prob = softmax(score)
attn_out = prob V
```

causal mask 让当前位置不能看未来 token。

## KV Cache

prefill 时，所有 prompt token 的 K/V 写入 cache：

```text
K_cache[0:T] = K_prompt
V_cache[0:T] = V_prompt
```

decode 第 `t` 步只计算新 token 的 `q_t, k_t, v_t`：

```text
K_cache[t] = k_t
V_cache[t] = v_t
out_t = softmax(q_t K_cache[:t+1]^T / sqrt(d_head) + mask) V_cache[:t+1]
```

所以 decode 不需要重新计算整个 prompt 的 Q/K/V。

## Dense SwiGLU MLP

传统 gated MLP 常见形式：

```text
gate = X W_gate
up = X W_up
act = silu(gate) * up
out = act W_down
```

其中：

```text
silu(x) = x * sigmoid(x)
```

## MoE Router

每个 token 的 hidden state `x in R^H` 经过 router：

```text
router_logits = x W_router
router_logits shape = [N, E]
```

如果评分函数是 softmax：

```text
router_prob = softmax(router_logits)
```

选择 top-k：

```text
topk_ids = topk(router_prob, K).indices
topk_weights = topk(router_prob, K).values
```

如果启用 renormalize：

```text
topk_weights = topk_weights / sum(topk_weights)
```

本模型 `E=256`，`K=8`，所以：

```text
topk_ids shape = [N, 8]
topk_weights shape = [N, 8]
```

## MoE Expert 输出

每个专家本质是一个 SwiGLU MLP：

```text
Expert_e(x) = W_down_e( silu(W_gate_e x) * (W_up_e x) )
```

一个 token 的 MoE 输出是被选中专家的加权和：

```text
MoE(x) = sum_{i=1..K} topk_weights_i * Expert_{topk_ids_i}(x)
```

## W8A8 Dynamic Quant

对 activation `x` 做动态 int8 量化：

```text
s_x = max(abs(x)) / 127
q_x = clamp(round(x / s_x), -128, 127)
```

权重离线量化后保存为：

```text
q_w = int8 weight
s_w = weight_scale
```

int8 matmul 近似浮点 matmul：

```text
y ~= (q_x @ q_w) * s_x * s_w
```

在代码里，`s_x` 对应 `pertoken_scale`，`s_w` 对应 `weight_scale`。

## LM Head 和 Logits

最后一层输出 hidden state：

```text
h_last shape = [1, H]
```

词表投影：

```text
logits = h_last W_vocab^T
logits shape = [1, V]
```

本次实测：

```text
logits shape = [1, 248320]
```

## Softmax 和 Greedy

概率：

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

greedy sampling：

```text
next_token = argmax_i logits_i
```

本测试 `temperature=0.0`，因此不走随机抽样，直接选最大 logit。
