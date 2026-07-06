# 005. FusedMoE 和 W8A8 源码流程

本章进入模型最关键的稀疏计算：MoE。

目标是看懂这条链：

```text
hidden_states
  -> router_logits
  -> topk_weights/topk_ids
  -> token dispatch
  -> expert MLP
  -> token combine
  -> final_hidden_states
```

## 1. router logits 从哪里来

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:210
```

关键代码：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)
```

本模型：

```text
num_experts = 256
num_experts_per_tok = 8
```

所以：

```text
hidden_states shape = [N, 2048]
router_logits shape = [N, 256]
```

`router_logits[i, e]` 表示第 i 个 token 对第 e 个 expert 的未归一化偏好分数。

## 2. `select_experts()` 选择 top-k expert

源码：

```text
vllm_ascend/ops/fused_moe/experts_selector.py:30
```

返回值：

```text
topk_weights shape = [num_tokens, top_k]
topk_ids shape = [num_tokens, top_k]
```

本模型：

```text
top_k = 8
```

所以：

```text
topk_weights shape = [N, 8]
topk_ids shape = [N, 8]
```

数学过程可以写成：

```text
scores = softmax(router_logits)
topk_ids = topk(scores, k=8).indices
topk_weights = topk(scores, k=8).values
```

如果 `renormalize=True`，还会做：

```text
topk_weights = topk_weights / sum(topk_weights)
```

不同模型可能使用不同 scoring function，比如 softmax、sigmoid、sqrtsoftplus；具体由参数决定。

## 3. W8A8_DYNAMIC MoE apply 入口

源码：

```text
vllm_ascend/quantization/methods/w8a8_dynamic.py:155
```

类：

```python
AscendW8A8DynamicFusedMoEMethod
```

关键字段：

```python
quant_type = QuantType.W8A8
```

它说明该 MoE quant method 使用 W8A8 路径。

## 4. W8A8 MoE 权重长什么样

源码：

```text
vllm_ascend/quantization/methods/w8a8_dynamic.py:188
```

权重：

```python
w13_weight: [num_experts, 2 * intermediate_size_per_partition, hidden_size]
w2_weight:  [num_experts, hidden_size, intermediate_size_per_partition]
```

为什么叫 `w13`？

SwiGLU MLP 通常有：

```text
gate_proj: hidden -> intermediate
up_proj:   hidden -> intermediate
down_proj: intermediate -> hidden
```

为了减少 kernel 次数，`gate_proj` 和 `up_proj` 常被打包成一个矩阵，也就是 `w13`。

SwiGLU 公式：

```text
a, b = split(x @ W13^T)
h = silu(a) * b
out = h @ W2^T
```

## 5. `apply()` 内部再次得到 top-k

源码：

```text
vllm_ascend/quantization/methods/w8a8_dynamic.py:270
```

关键代码：

```python
topk_weights, topk_ids = select_experts(
    hidden_states=x,
    router_logits=router_logits,
    top_k=top_k,
    ...
)
```

得到：

```text
topk_ids: 每个 token 选哪 8 个 expert
topk_weights: 每个 expert 输出按多大权重合并
```

注意：这里的 router 结果是 MoE 后续 dispatch 的基础，不能在后面随便重新算。

## 6. 进入通信方法 `fused_experts`

源码：

```text
vllm_ascend/quantization/methods/w8a8_dynamic.py:326
```

关键代码：

```python
final_hidden_states = moe_comm_method.fused_experts(
    fused_experts_input=build_fused_experts_input(
        hidden_states=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1=w1,
        w2=w2,
        quant_type=self.quant_type,
        ...
    )
)
```

这里把所有 MoE 必要数据打包：

```text
hidden_states
topk_weights
topk_ids
expert weights
quant_type
scales
activation
swiglu_limit
```

## 7. `moe_comm_method.fused_experts()` 四阶段

源码：

```text
vllm_ascend/ops/fused_moe/moe_comm_method.py:122
```

核心流程：

```python
token_dispatch_input = build_token_dispatch_input(...)
token_dispatch_output = self.token_dispatcher.token_dispatch(...)

mlp_compute_input = build_mlp_compute_input(...)
mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

routed_out = self.token_dispatcher.token_combine(...)
```

四阶段：

```text
1. build dispatch input
   根据 topk_ids 计算 token 应该去哪些 expert。

2. token_dispatch
   把 token 按 expert 重排，让同一个 expert 的 token 连续。

3. unified_apply_mlp
   对每个 expert 执行 W8A8 grouped matmul + SwiGLU + down projection。

4. token_combine
   把 expert 输出按 topk_weights 加权合回原 token 顺序。
```

最终输出：

```text
routed_out shape = [N, 2048]
```

## 8. W8A8 dynamic linear 的基本算子

普通 W8A8 linear 入口：

```text
vllm_ascend/quantization/methods/w8a8_dynamic.py:48
```

关键代码：

```python
quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
```

含义：

```text
当前激活 x 根据本次真实数值动态量化成 int8
同时得到 per-token scale
```

公式：

```text
x_int8 = round(x / scale_x)
y_int = x_int8 @ W_int8^T
y ≈ y_int * scale_x * scale_w
```

MoE W8A8 只是把这个思想扩展到 grouped expert MLP：

```text
不同 expert 有不同 W_int8 和 scale_w
不同 token 有自己的动态 activation scale
```

## 9. 权重加载后处理

源码：

```text
vllm_ascend/quantization/methods/w8a8_dynamic.py:353
```

关键操作：

```python
layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
layer.w13_weight.data = torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)
layer.w2_weight.data = torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)
```

这一步不是数学模型结构变化，而是为了让 NPU grouped matmul 使用更合适的内存布局。

所以你看到 NZ、transpose、scale flatten 时，要把它们归类为：

```text
硬件执行布局优化
```

不是新的 Transformer 层。

## 10. 本章数据流总结

```text
hidden_states [N, 2048]
  -> gate
router_logits [N, 256]
  -> select_experts(top_k=8)
topk_ids [N, 8], topk_weights [N, 8]
  -> token_dispatch
expert-local token batch
  -> W8A8 grouped MLP
expert outputs
  -> token_combine(topk_weights)
final_hidden_states [N, 2048]
```

这就是 A3B 稀疏激活的核心：总专家很多，但每个 token 只经过 8 个 expert。
