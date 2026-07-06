# 014. W8A8 Dynamic 量化机制

本章回答：`W8A8` 到底怎么计算？本测试里的 `quantization="ascend"` 又怎样落到 vLLM Ascend 的源码？

先说大白话：量化就是把原来用 bfloat16/float16 表示的数，用 int8 近似表示。权重提前离线量化好，运行时激活值根据当前 token 动态求 scale 再量化。矩阵乘主要在 int8 上做，最后再用 scale 把结果还原到目标 dtype。这样能减少权重存储、显存带宽和部分计算压力。

## 1. 本模型的量化事实

远端量化模型目录：

```text
/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp
```

权重文件：

```text
quant_model_weights-00001-of-00010.safetensors
...
quant_model_weights-00010-of-00010.safetensors
```

量化描述文件：

```text
quant_model_description.json
```

解析结果：

```text
model_quant_type = W8A8_DYNAMIC
version = 1.0.0
group_size = 0
language_or_mtp_value_count = {
  "FLOAT": 1367,
  "W8A8_DYNAMIC": 92250
}
```

代表性条目：

```text
model.language_model.layers.0.mlp.experts.0.gate_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.gate_proj.weight_offset = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.down_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.linear_attn.out_proj.weight = FLOAT
model.language_model.layers.3.self_attn.o_proj.weight = FLOAT
lm_head.weight = FLOAT
```

结论：这个模型不是“所有层都 int8”。主要 MoE expert 权重是 W8A8_DYNAMIC，部分 attention、linear attention、norm、lm_head 等仍是 FLOAT。

## 2. W8A8 是什么

`W8A8` 可以拆成：

```text
W8 = weight 8-bit
A8 = activation 8-bit
```

也就是：

- 权重用 int8 存储和计算。
- 激活值也在相关 kernel 中量化成 int8。

但本模型更准确的名字是：

```text
W8A8_DYNAMIC
```

`DYNAMIC` 的重点在 activation：运行时根据当前输入动态求 per-token scale，而不是完全使用离线固定 activation scale。

## 3. 最基础的量化公式

对浮点数 `x` 做对称 int8 量化：

```text
s = max(abs(x)) / 127
q = round(x / s)
q = clamp(q, -128, 127)
```

反量化近似：

```text
x ~= q * s
```

对于矩阵乘：

```text
Y = X W
```

量化后：

```text
X ~= Q_X * S_X
W ~= Q_W * S_W
Y ~= (Q_X @ Q_W) * S_X * S_W
```

这里 `Q_X` 和 `Q_W` 是 int8，`S_X` 和 `S_W` 是 scale。

## 4. 动态 activation scale

源码入口：

- `vllm_ascend/quantization/methods/w8a8_dynamic.py:75`

dynamic linear 的 apply：

```text
quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
output = torch_npu.npu_quant_matmul(
    quantized_x,
    layer.weight,
    layer.weight_scale,
    pertoken_scale=pertoken_scale,
    output_dtype=x.dtype,
)
```

`pertoken_scale` 这个名字很重要。它表示每个 token 的激活量化 scale。

假设输入 `x shape = [N, H]`，动态量化一般可以理解为每行一个 scale：

```text
S_X shape ~= [N]
Q_X shape = [N, H]
```

每个 token 的激活分布可能不同，所以动态求 scale 能比一个全局固定 scale 更好地保留数值范围。

## 5. 权重是怎样保存的

源码入口：

- `vllm_ascend/quantization/methods/w8a8_dynamic.py:61`
- `vllm_ascend/quantization/methods/w8a8_dynamic.py:188`

普通 linear 权重：

```text
weight dtype = int8
weight shape = [output_size, input_size]
weight_scale shape = [output_size, 1]
weight_offset shape = [output_size, 1]
```

MoE expert 权重：

```text
w13_weight shape = [num_experts, 2 * intermediate_size_per_partition, hidden_size]
w2_weight shape = [num_experts, hidden_size, intermediate_size_per_partition]
w13_weight_scale shape = [num_experts, 2 * intermediate_size_per_partition, 1]
w2_weight_scale shape = [num_experts, hidden_size, 1]
```

`w13` 是 `gate_proj` 和 `up_proj` 的打包权重；`w2` 是 `down_proj`。

本模型：

```text
num_experts = 256
hidden_size = 2048
moe_intermediate_size = 512
```

所以每层每个专家有三块主要权重：

```text
gate_proj: [2048, 512]
up_proj:   [2048, 512]
down_proj: [512, 2048]
```

打包到 `w13/w2` 后交给 grouped matmul。

## 6. process_weights_after_loading 做什么

源码入口：

- `vllm_ascend/quantization/methods/w8a8_dynamic.py:353`

关键操作：

```text
layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
layer.w13_weight.data = torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)
layer.w2_weight.data = torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)
layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(...)
layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(torch.float32)
```

这一步不是在“改变模型数学意义”，而是在把权重整理成 NPU kernel 更喜欢的内存布局和 scale 形状。

其中 `ACL_FORMAT_FRACTAL_NZ` 是 Ascend 上常见的高性能矩阵乘布局。矩阵乘硬件不一定喜欢普通行优先 ND 排布，转换到 NZ 能提升特定算子的访问效率。

## 7. MoE W8A8 apply 的完整链路

源码入口：

- `vllm_ascend/quantization/methods/w8a8_dynamic.py:214`
- `vllm_ascend/quantization/methods/w8a8_dynamic.py:326`

量化 MoE apply 做：

```text
topk_weights, topk_ids = select_experts(...)
topk_weights = topk_weights.to(self.in_dtype)

final_hidden_states = moe_comm_method.fused_experts(
    fused_experts_input=build_fused_experts_input(
        hidden_states=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1=w13_weight,
        w2=w2_weight,
        quant_type=QuantType.W8A8,
        w1_scale=w13_weight_scale,
        w2_scale=w2_weight_scale,
        ...
    )
)
```

注意：量化 MoE 不是只做矩阵乘。它先路由，再构造 fused experts 输入，再进入 dispatch/MLP/combine。

## 8. MoE MLP 里的动态量化和融合

源码入口：

- `vllm_ascend/ops/fused_moe/moe_mlp.py:88`
- `vllm_ascend/ops/fused_moe/moe_mlp.py:131`
- `vllm_ascend/ops/fused_moe/moe_mlp.py:296`
- `vllm_ascend/ops/fused_moe/moe_mlp.py:343`

如果 dispatch 阶段没有提供 `dynamic_scale`，MLP 会先做：

```text
hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(hidden_states)
```

然后第一段专家矩阵乘和 SwiGLU 可能融合：

```text
hidden_states, swiglu_out_scale, _ =
    DeviceOperator.npu_grouped_matmul_swiglu_quant(
        x=hidden_states,
        weight=w1,
        weight_scale=w1_scale,
        x_scale=pertoken_scale,
        ...
    )
```

如果没有走这条融合路径，就分开做：

```text
gate_up = npu_grouped_matmul(...)
act = npu_swiglu(gate_up)
act_int8, swiglu_out_scale = npu_dynamic_quant(act)
```

再做 down projection：

```text
hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
    hidden_states=hidden_states,
    weight=w2,
    weight_scale=w2_scale,
    per_token_scale=swiglu_out_scale,
    ...
)
```

这里的 `swiglu_out_scale` 是激活函数之后再次量化的 scale。也就是说，一个专家 MLP 里可能有两次 activation quant：

```text
输入 hidden -> quant -> gate/up matmul
SwiGLU 输出 -> quant -> down matmul
```

## 9. 为什么 scale 很重要

假设一个 activation 行：

```text
x = [0.01, -0.03, 2.40, ...]
```

如果不用 scale，直接把 float 强行 cast 到 int8，会丢掉大量小数信息。

有 scale：

```text
s = max(abs(x)) / 127
q = round(x / s)
```

最大值映射到 127 附近，其他值按比例映射。反量化时：

```text
x_hat = q * s
```

误差来自 round 和 clamp：

```text
error = x - x_hat
```

量化设计的目标不是完全无误差，而是在硬件效率和模型精度之间找到可接受平衡。

## 10. `weight_offset` 是什么

量化可以是对称的，也可以是不对称的。

对称量化常见形式：

```text
x ~= q * scale
```

不对称量化可能有 zero point / offset：

```text
x ~= (q - offset) * scale
```

本模型的量化描述里 expert 的 `weight_offset` 也是 `W8A8_DYNAMIC` 条目。具体 kernel 是否使用 offset，取决于对应 quant scheme 和算子路径。源码里可以看到某些路径会把 offset 作为 antiquant offset 传入，某些 W8A8 dynamic grouped 路径主要使用 weight scale。

初学时先抓住：`weight` 是 int8 主体，`weight_scale/weight_offset` 是还原数值范围所需的辅助参数。

## 11. 静态 W8A8 和动态 W8A8 的区别

源码入口：

- `vllm_ascend/quantization/methods/w8a8_static.py:33`
- `vllm_ascend/quantization/methods/w8a8_dynamic.py:48`

静态 W8A8 linear：

```text
x = torch.ops.vllm.quantize(
    x,
    layer.aclnn_input_scale,
    layer.aclnn_input_scale_reciprocal,
    layer.aclnn_input_offset,
)
output = torch_npu.npu_quant_matmul(x, weight, deq_scale, ...)
```

动态 W8A8 linear：

```text
quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
output = torch_npu.npu_quant_matmul(
    quantized_x,
    weight,
    weight_scale,
    pertoken_scale=pertoken_scale,
)
```

区别：

- 静态：activation scale 多来自离线校准或固定配置。
- 动态：每次运行按当前输入动态计算 activation scale。

本模型 `model_quant_type=W8A8_DYNAMIC`，所以讲义重点放在动态路径。

## 12. 为什么 attention/lm_head 可能还是 FLOAT

量化描述显示：

```text
model.language_model.layers.0.linear_attn.out_proj.weight = FLOAT
model.language_model.layers.3.self_attn.o_proj.weight = FLOAT
lm_head.weight = FLOAT
```

这很常见，原因可能包括：

- 某些层对精度更敏感。
- 某些算子还没有对应的高效量化 kernel。
- 某些结构如 linear attention 的特定 projection 或状态更新不适合这版量化方案。
- lm_head 词表投影维度大且直接影响 token 排名，保留浮点可能更稳。

所以看到模型名 `w8a8` 时，不要自动理解为“所有矩阵都是 int8”。要看 `quant_model_description.json` 或运行时 quant config。

## 13. `quantization="ascend"` 的作用

测试参数：

```python
quantization="ascend"
```

这会让 vLLM Ascend 的量化配置解析和方法注册生效。相关源码：

- `vllm_ascend/quantization/modelslim_config.py:66`
- `vllm_ascend/quantization/method_adapters.py:201`
- `vllm_ascend/quantization/methods/registry.py`

ModelSlim 配置里有模型模块映射：

```text
"qwen3_5_moe": {
  "qkv_proj": ["q_proj", "k_proj", "v_proj"],
  "gate_up_proj": ["gate_proj", "up_proj"],
  "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
  "in_proj_ba": ["in_proj_b", "in_proj_a"],
  "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
}
```

这告诉 loader：原始 checkpoint 里的多个权重如何映射到 vLLM 的 packed module。

## 14. 本章完整链路

```text
quant_model_description.json
  -> Ascend ModelSlim config
  -> quant method registry
  -> AscendFusedMoEMethod
  -> AscendW8A8DynamicFusedMoEMethod
  -> process_weights_after_loading
     -> int8 weight layout / scale reshape / NZ format
  -> apply
     -> select_experts
     -> build_fused_experts_input(quant_type=W8A8)
     -> dispatch
     -> dynamic quant activation
     -> grouped int8 matmul
     -> SwiGLU
     -> dynamic quant activation
     -> grouped int8 matmul
     -> combine
```

## 15. 初学者检查点

1. W8A8 中的 W 和 A 分别是什么？
2. 为什么动态量化需要 `pertoken_scale`？
3. 为什么 int8 matmul 后还要 scale？
4. 为什么这个模型里 `lm_head.weight` 是 FLOAT？
5. `W8A8_DYNAMIC` 和 `W8A8` 静态量化的核心区别是什么？

答案要点：

- W 是权重，A 是激活。
- 每个 token 的激活范围不同，需要单独 scale。
- int8 只是缩放后的整数表示，要用 scale 还原数值范围。
- 保留浮点可能是精度或算子支持选择。
- 动态量化运行时求 activation scale；静态量化使用预先给定的 activation scale。
