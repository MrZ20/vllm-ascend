# 术语表

## prompt

给模型的输入文本。本测试中是：`Hello, my name is`。

## token

文本被 tokenizer 切分后的基本单位，用整数 ID 表示。本次 prompt 变成了 `[9419, 11, 821, 803, 369]`。

## tokenizer

负责文本和 token ID 互相转换的组件。

## prefill

推理的第一阶段。模型一次性读取 prompt 的所有 token，并建立内部 cache。

## decode

prefill 之后的重复阶段。模型通常每次处理一个新 token。

## logits

下一个 token 的原始分数。本次运行中 logits 形状是 `(1, 248320)`。

## vocabulary

模型可选择的全部 token 集合。本次运行中词表大小是 248,320。

## greedy sampling

确定性生成方式：选择分数最高的 token。在 vLLM 中，本测试通过 `temperature=0.0` 进入 greedy。

## KV cache

缓存历史 token 的 attention 状态。它让 decode 不需要每一步重新计算完整 prompt。

## tensor parallelism

把大 tensor 计算拆到多个 rank/device 上。本测试使用 `tensor_parallel_size=2`。

## expert parallelism

把 MoE experts 拆到多个设备上。本测试显式关闭。

## MoE

mixture of experts。router 会把每个 token 送到选中的 expert 子网络。

## W8A8

权重和激活都使用 8-bit 表示的量化计算方式。

## graph capture

运行时优化：把重复执行流程录下来，后续直接复用。
