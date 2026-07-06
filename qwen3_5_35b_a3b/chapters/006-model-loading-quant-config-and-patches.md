# 006. 模型加载、量化配置和 Ascend patch

本章回答：`quantization="ascend"` 到底让 vLLM Ascend 做了什么？为什么模型叫 W8A8，但日志里又能看到一些权重还是 `FLOAT`？

先说大白话：模型目录里不只有权重。它至少包含三类信息：模型结构配置、权重张量、量化说明。vLLM 先读配置，知道模型长什么样；再读量化说明，知道每一层该用浮点还是 int8 量化路径；最后加载权重，并把上游 Qwen3.5 的部分 forward 替换成 Ascend 适配实现。

## 1. 模型加载阶段读了哪些文件

本次真实模型目录是：

```text
/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp
```

关键文件分四类。

第一类是模型结构：

```text
config.json
```

它告诉 vLLM：

```text
model_type = qwen3_5_moe_text
hidden_size = 2048
num_hidden_layers = 40
num_attention_heads = 16
num_key_value_heads = 2
num_experts = 256
num_experts_per_tok = 8
vocab_size = 248320
```

第二类是量化说明：

```text
quant_model_description.json
```

它告诉 vLLM Ascend 每个权重对应哪种量化方式。本次解析结果里有：

```text
model_quant_type = W8A8_DYNAMIC
W8A8_DYNAMIC 条目 = 92250
FLOAT 条目 = 1367
```

第三类是权重分片：

```text
quant_model_weights-00001-of-00010.safetensors
...
quant_model_weights-00010-of-00010.safetensors
```

第四类是 tokenizer 文件。它们负责把字符串变成 token ID，再把 token ID 解码回字符串。

## 2. 配置决定“模型骨架”

`config.json` 里的参数决定 tensor 形状。比如：

```text
hidden_size = 2048
vocab_size = 248320
```

所以 embedding 后的 hidden state 是：

```text
inputs_embeds shape = [N, 2048]
```

LM head 后的 logits 是：

```text
logits shape = [B, 248320]
```

本次真实追踪中：

```text
prefill inputs_embeds shape = (8, 2048)
sample logits shape = (1, 248320)
```

这不是巧合，而是配置数字直接推导出来的。

再看 attention：

```text
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
```

因此 Q 的总通道数大致是：

```text
16 * 256 = 4096
```

K/V 的总通道数大致是：

```text
2 * 256 = 512
```

这说明模型使用 GQA，也就是 Query head 比 Key/Value head 多。GQA 可以减少 KV cache 体积，因为缓存 K/V 时只需要缓存较少的 KV head。

## 3. `quantization="ascend"` 如何进入量化系统

测试文件传入：

```python
quantization="ascend"
```

vLLM Ascend 里对应常量是：

```python
ASCEND_QUANTIZATION_METHOD = "ascend"
```

源码入口：

- `vllm_ascend/utils.py:48`
- `vllm_ascend/quantization/modelslim_config.py:456`

`AscendModelSlimConfig` 通过注册机制成为 `"ascend"` 量化方式的配置类。初始化时，它会加载和保存 `quant_model_description.json` 里的量化描述。

关键路径是：

```text
quantization="ascend"
  -> AscendModelSlimConfig
  -> maybe_update_config()
  -> 读取 quant_model_description.json
  -> get_quant_method(layer, prefix)
  -> 为每个 layer 选择 AscendLinearMethod / AscendFusedMoEMethod / Unquantized...
```

源码入口：

- `vllm_ascend/quantization/modelslim_config.py:749`
- `vllm_ascend/quantization/modelslim_config.py:793`

## 4. 为什么 W8A8 模型里还有 `FLOAT`

`quant_model_description.json` 不是只写一句“全模型 W8A8”。它是按权重名字逐项描述的。

本次代表性条目：

```text
model.language_model.layers.0.mlp.experts.0.gate_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.down_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.linear_attn.out_proj.weight = FLOAT
model.language_model.layers.3.self_attn.o_proj.weight = FLOAT
lm_head.weight = FLOAT
```

这说明本模型的关键 MoE expert 权重走 W8A8 dynamic 路径，但部分 attention、linear attention、lm_head 保留浮点。

源码里也有对应判断：

```text
如果某层被标成 FLOAT:
    选择 AscendUnquantizedLinearMethod 或 AscendUnquantizedFusedMoEMethod
否则:
    根据 W8A8_DYNAMIC 创建对应 quant scheme
```

源码入口：

- `vllm_ascend/quantization/modelslim_config.py:793`
- `vllm_ascend/quantization/modelslim_config.py:839`

所以以后不要说“W8A8 模型每个矩阵乘都是 int8”。更准确的说法是：

```text
这个模型包的量化描述是 W8A8_DYNAMIC，
其中 MoE expert 等大量权重使用 W8A8_DYNAMIC，
但仍有一部分模块按 FLOAT 路径执行。
```

## 5. 权重量化的基本数据结构

以一个线性层为例，浮点矩阵乘是：

```text
y = x W^T + b
```

W8A8 dynamic 的核心思想是把它拆成：

```text
x_float -> x_int8, x_scale
W_float -> W_int8, W_scale
y_int = x_int8 @ W_int8^T
y_float ≈ y_int * x_scale * W_scale
```

其中 `W_int8` 和 `W_scale` 来自模型权重文件和量化描述；`x_scale` 是动态的，因为它取决于当前真实输入激活。

本测试中的 MoE expert 权重条目会包含类似：

```text
weight
weight_scale
weight_offset
```

这些不是额外的模型层，而是为了把 int8 数值还原到合理浮点范围所需的量化元数据。

更细的 W8A8 公式在 `chapters/014-w8a8-dynamic-quantization.md`。

## 6. Ascend patch 是什么时候生效的

vLLM Ascend 是硬件插件，不直接把所有模型文件复制一份。它通常通过 patch 改写上游模型的关键 forward。

worker patch 入口：

```text
vllm_ascend/patch/worker/__init__.py
```

其中非 310P 设备会导入：

```python
import vllm_ascend.patch.worker.patch_qwen3_5
```

源码入口：

- `vllm_ascend/patch/worker/__init__.py:49`

`patch_qwen3_5.py` 最后直接替换上游类的方法：

```python
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
```

源码入口：

- `vllm_ascend/patch/worker/patch_qwen3_5.py:195`
- `vllm_ascend/patch/worker/patch_qwen3_5.py:196`

这意味着：你看上游 `qwen3_5.py` 能理解模型结构，但真实 Ascend 运行时，某些 forward 已经被替换为 Ascend 适配版本。

## 7. patch 为什么必要

上游 vLLM 的 Qwen3.5 模型是通用实现，而 Ascend 上需要处理：

- NPU attention kernel；
- Q/K RMSNorm 和 RoPE 的融合；
- Gated DeltaNet / linear attention 的 NPU 实现；
- Ascend FusedMoE；
- W8A8 quant matmul；
- 图捕获和 padding 形状；
- NPU tensor layout，例如部分场景里的 NZ 格式。

这些实现如果全部塞回上游模型文件，会破坏硬件插件边界。所以 vLLM Ascend 采用 patch：结构仍复用上游模型，执行路径在 worker 侧替换成 Ascend 友好的版本。

## 8. 本章小结

这一章对应的心智模型是：

```text
config.json 决定模型结构和 tensor 形状
quant_model_description.json 决定每层走 FLOAT 还是 W8A8_DYNAMIC
safetensors 保存真实权重和量化元数据
Ascend patch 替换关键 forward，让上游模型能在 NPU 上高效执行
```

到这里，模型还没有处理你的 prompt。但模型骨架、权重、量化方法、forward 实现都已经准备好。下一章才进入 request、scheduler 和真正的推理循环。
