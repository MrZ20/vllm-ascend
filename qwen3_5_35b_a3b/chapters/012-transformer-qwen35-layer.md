# 012. Qwen3.5 Transformer 层内部

本章回答：一次 `_model_forward()` 进去之后，Qwen3.5 的每一层到底在做什么？

先说大白话：一层 decoder layer 会先把当前 token 的表示做一次归一化，然后让 token 从历史上下文里取信息，再把结果送进 MoE 专家网络做非线性变换，最后把这个更新后的表示交给下一层。40 层重复这个过程，表示就从“词表里的一个 token”逐渐变成“带上下文语义的向量”。

## 1. 本模型层结构事实

远端量化模型 `config.json` 的 `text_config`：

```text
model_type = qwen3_5_moe_text
hidden_size = 2048
num_hidden_layers = 40
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
rms_norm_eps = 1e-6
hidden_act = silu
layer_types_count = {"linear_attention": 30, "full_attention": 10}
```

第一批 layer type：

```text
["linear_attention", "linear_attention", "linear_attention", "full_attention",
 "linear_attention", "linear_attention", "linear_attention", "full_attention",
 ...]
```

也就是基本上每 4 层里 3 层 linear attention，1 层 full attention。

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:112`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:233`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:90`

## 2. Decoder layer 初始化决定走哪条路径

上游初始化逻辑：

```text
if self.layer_type == "linear_attention":
    self.linear_attn = QwenGatedDeltaNetAttention(...)
elif self.layer_type == "full_attention":
    self.self_attn = Qwen3NextAttention(...)
```

然后决定 MLP 类型：

```text
if config.model_type == "qwen3_5_moe_text":
    self.mlp = Qwen3NextSparseMoeBlock(...)
elif config.model_type == "qwen3_5_text":
    self.mlp = Qwen3NextMLP(...)
```

本模型是 `qwen3_5_moe_text`，所以：

```text
self.mlp = Sparse MoE
```

这句话非常关键。很多初学者看到 “Transformer = attention + MLP”，会以为 MLP 是一个固定 dense FFN。但在 MoE 模型里，MLP 位置被“router + 多专家 FFN”替代。

## 3. Ascend patch 的 forward 顺序

Ascend 侧 monkeypatch 替换了 Qwen3.5 的 decoder layer forward：

源码：`vllm_ascend/patch/worker/patch_qwen3_5.py:90`

结构简化：

```text
if residual is None:
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
else:
    hidden_states, residual = input_layernorm(hidden_states, residual)

if layer_type == "linear_attention":
    linear_attn(hidden_states, output=self_attention_output)
elif layer_type == "full_attention":
    self_attn(hidden_states, output=self_attention_output, positions=positions)

hidden_states = self_attention_output

if layer_scale:
    hidden_states = hidden_states * (attn_layer_scale + 1)

hidden_states, residual = post_attention_layernorm(hidden_states, residual)
hidden_states = mlp(hidden_states)

if layer_scale:
    hidden_states = hidden_states * (ffn_layer_scale + 1)

return hidden_states, residual
```

这是一种 pre-norm 结构：先 norm，再 attention/MoE。pre-norm 对深层 Transformer 更稳定，因为 residual 主干更容易保留梯度和信息。

## 4. RMSNorm 做了什么

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:165`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:98`

RMSNorm 公式：

```text
rms(x) = sqrt(mean(x_i^2) + eps)
RMSNorm(x)_i = x_i / rms(x) * gamma_i
```

对于 hidden size `H=2048`：

```text
rms(x) = sqrt((x_1^2 + x_2^2 + ... + x_2048^2) / 2048 + 1e-6)
```

RMSNorm 和 LayerNorm 的区别：

- LayerNorm 通常会减均值再除标准差。
- RMSNorm 不减均值，只除均方根。

为什么要 norm？

每层的矩阵乘、attention、MoE 都会改变向量尺度。如果尺度越来越大或越来越小，后面的 softmax、激活函数和量化 kernel 都会更难稳定。RMSNorm 把每个 token 的向量尺度拉回一个可控范围。

## 5. residual 是什么

residual 是“保留上一层输入的主干”。抽象公式：

```text
y = x + F(norm(x))
```

本 patch 里 RMSNorm 接口会同时处理 `hidden_states` 和 `residual`。你可以先理解成：

```text
residual 保存旧 hidden
hidden_states 是当前要送进子层的归一化输入
```

残差连接的意义：

- 让模型每一层只学习“增量更新”。
- 避免深层网络把早期信息洗掉。
- 让梯度更容易跨层传播。

## 6. Full Attention 详细流程

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:225`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:41`

### 6.1 QKV projection

输入：

```text
hidden_states shape = [N, 2048]
```

代码：

```text
qkv, _ = self.qkv_proj(hidden_states)
```

这一步把 hidden state 一次投影成 Q、K、V，以及可选 gate 所需的数据。Qwen3NextAttention 初始化中：

```text
self.total_num_heads = config.num_attention_heads      # 16
self.total_num_kv_heads = config.num_key_value_heads   # 2
self.head_dim = config.head_dim                        # 256
```

### 6.2 GQA

GQA 是 grouped-query attention。它的特征是：

```text
Q heads = 16
KV heads = 2
```

也就是说，每 8 个 Q head 共享 1 个 KV head。好处：

- KV cache 更小。
- decode 读 KV cache 的带宽更低。
- 相比 MQA，保留更多 KV head，精度通常更好。

### 6.3 Q/K RMSNorm 和 RoPE

Ascend patch 中：

```text
q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(...)
```

这个融合算子做了几件事：

1. 从 qkv 大张量里拆出 Q、K、V、gate。
2. 对 Q 和 K 做 RMSNorm。
3. 根据 positions 查 `cos_sin_cache`。
4. 对 Q/K 应用 RoPE。

RoPE 的直觉：把位置信息编码成旋转。对每个位置 `pos`，把 Q/K 向量的一部分维度按正弦余弦旋转。这样点积 `q_i dot k_j` 会携带相对位置信息。

标准二维旋转形式：

```text
[x_1', x_2'] = [x_1 cos(theta) - x_2 sin(theta),
                x_1 sin(theta) + x_2 cos(theta)]
```

Qwen3.5 配置：

```text
rope_theta = 10000000
partial_rotary_factor = 0.25
mrope_section = [11, 11, 10]
mrope_interleaved = true
```

这说明它不是简单对全部 head_dim 做普通 RoPE，而是对部分维度和 mRoPE section 做旋转。

### 6.4 causal attention

核心公式：

```text
score = Q K^T / sqrt(256) + causal_mask
prob = softmax(score)
attn_out = prob V
```

causal mask 的作用：第 `t` 个 token 只能看 `<=t` 的 token，不能偷看未来。

### 6.5 attention output gate

本模型：

```text
attn_output_gate = true
```

Ascend patch 中：

```text
gate = torch.sigmoid(gate)
attn_output = attn_output * gate
```

这是一种门控机制。它让模型可以逐维控制 attention 输出通过多少，类似“这部分上下文信息要不要强烈写入当前 hidden state”。

### 6.6 output projection

代码：

```text
output[:], _ = self.o_proj(attn_output)
```

它把多 head attention 输出重新投影回 hidden size：

```text
attn_out shape = [N, 2048]
```

## 7. Linear Attention 在这里的角色

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:129`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:114`

Qwen3.5 的 30 层是 `linear_attention`，使用的是 `QwenGatedDeltaNetAttention`。

先说直觉：full attention 在每个 token 上都要和历史 token 做相关性计算，长度为 `T` 时，注意力矩阵规模接近 `T x T`。linear attention / gated delta net 一类结构会把历史压缩成状态，让每步更新更接近线性复杂度。

full attention 的历史访问像：

```text
当前 q_t 直接和所有 k_1...k_t 做点积
```

linear/state attention 的思路更像：

```text
state_t = update(state_{t-1}, x_t)
out_t = read(state_t, x_t)
```

本仓库 Ascend patch 对 GDN 也做了替换：

```text
_GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
```

所以 Qwen3.5 的 `linear_attention` 不只是普通 PyTorch 代码，而会进入 Ascend 适配过的 GDN attention 实现。

读到这里先抓住三点：

- `linear_attention` 是为长上下文效率服务的。
- 它不等同于 full attention 的 `softmax(QK^T)V`。
- 但在 decoder layer 外观看，它仍然接收 hidden states，输出同形状 hidden states。

## 8. MLP 位置实际是 Sparse MoE

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:138`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:176`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:195`

`Qwen3NextSparseMoeBlock` 初始化里：

```text
self.gate = ReplicatedLinear(hidden_size, num_experts)
self.shared_expert_gate = ReplicatedLinear(hidden_size, 1)
self.shared_expert = Qwen3NextMLP(...)
self.experts = FusedMoE(...)
```

forward 中：

```text
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)
```

这说明每层 attention 后的 FFN 位置会进入 MoE：

```text
hidden_states -> router -> top-8 experts -> weighted sum
```

本模型每层 MoE：

```text
num_experts = 256
num_experts_per_tok = 8
```

## 9. 一层的完整公式近似

忽略一些工程细节，一层可以写成：

```text
u = RMSNorm(x)
a = AttentionOrLinearAttention(u)
x_attn = x + a

v = RMSNorm(x_attn)
m = MoE(v)
y = x_attn + m
```

因为代码里 residual 的携带方式更复杂，真实实现不是逐字等于这 5 行，但数学结构可以这么理解：每层通过 attention 更新上下文信息，通过 MoE 更新非线性变换能力，并用 residual 保持信息主干。

## 10. 本章和真实运行的连接

动态追踪看到：

```text
NPUModelRunner._model_forward(num_tokens_padded=8, ...)
```

这一步内部会调用：

```text
Qwen3_5ForCausalLM.forward
  -> Qwen3_5Model.forward
    -> 40 x Qwen3_5DecoderLayer.forward
```

每个 `Qwen3_5DecoderLayer.forward` 已被 `vllm_ascend/patch/worker/patch_qwen3_5.py` 替换成 Ascend 版本。

所以你看日志里只有 `_model_forward`，不代表模型内部只有一个函数。它里面展开的是 40 层，每层又包含 attention/linear_attention、MoE、量化/非量化线性层、NPU kernel、通信和 cache 操作。

## 11. 初学者检查点

1. 为什么说 Qwen3.5 这一层不是单纯的 “attention + dense MLP”？
2. GQA 的 `16 Q heads / 2 KV heads` 对 KV cache 有什么影响？
3. RoPE 为什么要作用在 Q/K 上，而不是直接作用在 logits 上？
4. `linear_attention` 为什么对长上下文有意义？
5. RMSNorm 和 residual 分别解决什么问题？

答案要点：

- MoE 模型里 MLP 位置换成 router + experts。
- KV heads 少，KV cache 和读取带宽更小。
- attention 分数来自 Q/K 点积，位置关系要进入 Q/K 才能影响注意力。
- 它用状态式/线性复杂度思想减少长序列成本。
- RMSNorm 控制尺度，residual 保留信息主干。
