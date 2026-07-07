# 公式速查

这份文件只放本次 Qwen3.5-35B-A3B W8A8 推理链路里会反复用到的公式和形状。它不是论文推导，而是源码阅读时的“公式地图”。

## 符号

| 符号 | 含义 | 本模型数值 |
|---|---|---:|
| `N` | 当前 step 的真实 token 数 | prefill 为 `5`，decode 为 `1` |
| `N_pad` | padding 后的执行 token 数 | prefill 可为 `8` |
| `H` | hidden size | `2048` |
| `V` | vocabulary size | `248320` |
| `L` | decoder layer 数 | `40` |
| `QH` | query head 数 | `16` |
| `KVH` | key/value head 数 | `2` |
| `D` | head dim | `256` |
| `E` | MoE expert 数 | `256` |
| `K_top` | 每个 token 选择的 expert 数 | `8` |
| `I_moe` | MoE intermediate size | `512` |

本次真实输入：

```text
prompt = "Hello, my name is"
prompt_token_ids = [9419, 11, 821, 803, 369]
generated_token_ids = [498, 7525, 3855, 1089, 321]
```

## Token Embedding

tokenizer 把字符串变成 token ID。模型看到的是整数，不是字符串。

```text
ids = [9419, 11, 821, 803, 369]
Embedding table E_tok shape = [V, H]
X_0 = E_tok[ids]
X_0 shape = [N, H]
```

本次 prefill：

```text
真实 shape = [5, 2048]
图执行 shape 可 padding 到 [8, 2048]
```

padding 只改变硬件执行形状，不改变真实 token 数。

## Position

runner 里位置计算的核心公式：

```text
position = num_computed_tokens + query_pos
```

prefill：

```text
num_computed_tokens = 0
query_pos = [0, 1, 2, 3, 4]
positions = [0, 1, 2, 3, 4]
```

decode 第一步：

```text
num_computed_tokens = 5
query_pos = [0]
positions = [5]
```

如果使用 M-RoPE，`positions` 可能是：

```text
positions shape = [3, N_pad]
```

这里的 `3` 对应多维位置分量。

## RMSNorm

RMSNorm 不减均值，只按均方根缩放。

对一行 hidden state `x in R^H`：

```text
rms(x) = sqrt((1 / H) * sum_j x_j^2 + eps)
RMSNorm(x)_i = x_i / rms(x) * gamma_i
```

本模型：

```text
eps = 1e-6
H = 2048
```

RMSNorm 的作用是控制每个 token 的向量尺度，让后续 attention、MoE 和量化更稳定。

## Residual

Transformer layer 常见结构：

```text
u = RMSNorm(x)
a = Attention(u)
x1 = x + a
v = RMSNorm(x1)
m = MLP_or_MoE(v)
y = x1 + m
```

在本仓库 patch 里，residual 有时被 RMSNorm 模块融合处理，所以源码中不一定能看到显式的 `x + a`。

## RoPE

RoPE 把 position 注入 Q/K。简化到一对维度：

```text
q_even' = q_even * cos(theta_pos) - q_odd * sin(theta_pos)
q_odd'  = q_even * sin(theta_pos) + q_odd * cos(theta_pos)
```

K 同理：

```text
k_even' = k_even * cos(theta_pos) - k_odd * sin(theta_pos)
k_odd'  = k_even * sin(theta_pos) + k_odd * cos(theta_pos)
```

attention score 用旋转后的 Q/K：

```text
score(i, j) = Q_i' dot K_j' / sqrt(D)
```

本模型配置中：

```text
mrope_section = [11, 11, 10]
partial_rotary_factor = 0.25
rope_theta = 10000000
```

## Q/K/V 和 GQA

full attention 的线性投影：

```text
Q = X W_Q
K = X W_K
Vv = X W_V
```

reshape 后：

```text
Q shape  = [N, QH, D]
K shape  = [N, KVH, D]
Vv shape = [N, KVH, D]
```

本模型：

```text
QH = 16
KVH = 2
D = 256
```

这叫 GQA。多个 query heads 共享较少的 KV heads，用来降低 KV cache 和带宽压力。

## Causal Full Attention

prefill 中第 `i` 个 token 只能看见 `j <= i`：

```text
score = Q K^T / sqrt(D) + causal_mask
prob = softmax(score)
attn_out = prob Vv
```

causal mask：

```text
mask[i, j] = 0     if j <= i
mask[i, j] = -inf  if j > i
```

decode 时只输入新 token，但会读历史 KV cache：

```text
out_t = softmax(q_t K_cache[:t+1]^T / sqrt(D) + mask) V_cache[:t+1]
```

## KV Cache

prefill 写入所有 prompt token 的 K/V：

```text
K_cache[0:T] = K_prompt
V_cache[0:T] = V_prompt
```

decode 第 `t` 步写入新 token：

```text
K_cache[t] = k_t
V_cache[t] = v_t
```

KV cache 的粗略容量与这些量成正比：

```text
num_layers_with_kv * max_tokens * KVH * D * 2 * bytes_per_element
```

其中 `2` 表示 K 和 V 两份缓存。

## Linear Attention / Gated DeltaNet

本模型有 `30` 层 `linear_attention` 和 `10` 层 `full_attention`。Gated DeltaNet 的完整公式较复杂，源码阅读时先抓住它与 full attention 的区别：

```text
full attention:
  当前 token 通过 QK^T 显式访问历史 K/V。

linear/state attention:
  当前 token 更新或读取某种递推状态 S。
```

抽象形式：

```text
S_t = update(S_{t-1}, x_t)
y_t = read(S_t, x_t)
```

在本目录里，GDN 重点放在源码路径和状态管理，不把它展开成论文级完整推导。

## SwiGLU MLP

普通 gated MLP：

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

MoE expert 使用同类结构。Ascend FusedMoE 中常把 `gate_proj` 和 `up_proj` 打包成 `w13`。

## MoE Router

每个 token 的 hidden state `x in R^H` 经过 router：

```text
router_logits = x W_router
router_logits shape = [N, E]
```

如果 scoring function 是 softmax：

```text
router_prob = softmax(router_logits)
```

选择 top-k：

```text
topk_ids = topk(router_prob, K_top).indices
topk_weights = topk(router_prob, K_top).values
```

如果启用 renormalize：

```text
topk_weights = topk_weights / sum(topk_weights, dim=-1, keepdim=True)
```

本模型：

```text
E = 256
K_top = 8
topk_ids shape = [N, 8]
topk_weights shape = [N, 8]
```

## MoE Expert Combine

每个 expert 是一个 SwiGLU MLP：

```text
Expert_e(x) = W_down_e( silu(W_gate_e x) * (W_up_e x) )
```

单个 token 的 MoE 输出：

```text
MoE(x) = sum_i topk_weights_i * Expert_{topk_ids_i}(x)
```

向量化后：

```text
hidden_states [N, H]
  -> router_logits [N, E]
  -> topk_ids/topk_weights [N, K_top]
  -> expert outputs [N, K_top, H]
  -> weighted sum [N, H]
```

## FusedMoE Dispatch / Combine

FusedMoE 为了减少小算子和 Python 循环，会把 token 按 expert 重排：

```text
原顺序 token: [t0, t1, t2, ...]
按 expert 分组: expert0_tokens, expert1_tokens, ...
```

执行：

```text
dispatch_input = group_by_expert(hidden_states, topk_ids)
expert_output = grouped_mlp(dispatch_input)
output = combine_to_original_order(expert_output, topk_weights)
```

核心不变：结果仍然是每个 token 的 `[H]` 向量。

## W8A8 Dynamic Quant

激活动态量化：

```text
s_x = max(abs(x)) / 127
q_x = clamp(round(x / s_x), -128, 127)
```

权重离线量化：

```text
q_w = int8 weight
s_w = weight_scale
z_w = weight_offset
```

int8 matmul 近似浮点 matmul：

```text
y_int = q_x @ q_w^T
y ~= y_int * s_x * s_w
```

如果使用 offset，概念上还要把 zero-point/offset 修正纳入反量化。源码中 `pertoken_scale` 对应动态激活 scale，`weight_scale` / `weight_offset` 来自量化权重。

本测试中不是所有层都是 W8A8：

```text
MoE expert 代表性权重 = W8A8_DYNAMIC
attention / linear attention / lm_head 代表性权重 = FLOAT
```

## Tensor Parallel

Column parallel：

```text
W = [W_0, W_1]
Y_0 = X W_0
Y_1 = X W_1
Y = concat(Y_0, Y_1)
```

Row parallel：

```text
X = [X_0, X_1]
W = [W_0; W_1]
Y_0 = X_0 W_0
Y_1 = X_1 W_1
Y = all_reduce_sum(Y_0 + Y_1)
```

本测试：

```text
tensor_parallel_size = 2
enable_expert_parallel = False
```

所以使用 TP，不使用 EP all-to-all。

## Logits

最后一层输出 hidden state：

```text
h_sample shape = [1, H]
```

词表投影：

```text
logits = h_sample W_vocab^T
logits shape = [1, V]
```

本次实测：

```text
logits shape = [1, 248320]
```

## Softmax、Temperature 和 Greedy

softmax：

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

temperature：

```text
p_i = softmax(logit_i / temperature)
```

greedy：

```text
next_token = argmax_i logits_i
```

因为：

```text
argmax(softmax(logits)) = argmax(logits)
```

所以 greedy 不需要显式计算 softmax。`temperature=0.0` 在本测试中进入 greedy 路径。

## 本次完整数据闭环

```text
"Hello, my name is"
  -> [9419, 11, 821, 803, 369]
  -> Embedding [5, 2048]
  -> padding/graph [8, 2048]
  -> 40 layers
  -> sample_hidden_states [1, 2048]
  -> logits [1, 248320]
  -> argmax
  -> [498, 7525, 3855, 1089, 321]
  -> " [Your Name], and"
```
