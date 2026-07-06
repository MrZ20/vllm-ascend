# 004 - 模型和运行时概念

## Qwen3.5-35B-A3B

`35B` 表示大致总参数规模。`A3B` 表示模型是稀疏激活的：对某一个 token 来说，不是所有参数都会参与计算，而是只激活其中一部分。这通常来自 `MoE`，也就是 mixture of experts。

小白心智模型：

- Dense model：每个 token 都走同一条大 FFN 路径。
- MoE model：router 为每个 token 选择少量 expert 路径。
- A3B：每个 token 实际激活的参数量大约比 35B 总参数少很多。

## W8A8

`W8A8` 表示权重和激活在支持的 kernel 中使用 8-bit 表示。

小白心智模型：

- 高精度就像用更大的量杯存每个数。
- 8-bit 量化像用更小的量杯存很多数。
- 这样可以节省显存和带宽，但需要专门的硬件 kernel 保证速度和可用精度。

本测试里，`quantization="ascend"` 要求 vLLM Ascend 使用 Ascend 专用量化路径。

## TP=2

`tensor_parallel_size=2` 表示使用 2 个 tensor parallel rank。

运行证据：

```text
Worker_TP0
Worker_TP1
```

小白心智模型：一个很大的矩阵计算被拆给两个 worker，各算一部分，需要时再通信合并。

## EP 关闭

`enable_expert_parallel=False` 表示这个测试不使用 expert parallelism。

小白心智模型：

- TP 拆 tensor 计算。
- EP 拆 expert。
- 本测试使用 TP，但不使用 EP。

## graph capture

测试配置：

```python
cudagraph_capture_sizes=[1, 2, 4, 8]
```

运行证据：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 4/4
Capturing CUDA graphs (decode, FULL): 4/4
```

虽然日志里写的是 `CUDA graphs`，但在 Ascend 场景里这里可以理解为类似的图捕获机制：把常见 batch/token 形状的执行流程录下来，后续直接复用，减少调度开销。

## MTP

模型名以 `mtp` 结尾，说明模型包中带有 multi-token prediction 支持。本测试调用的是普通 greedy generation，`max_tokens=5`。动态追踪中 `_sample(..., spec_decode_metadata=None)`，说明这次采样路径没有使用 speculative decoding metadata。
