# 001 - 测试文件和工作流

## 测试文件写了什么

目标文件定义了一句 prompt：

```python
EXAMPLE_PROMPTS = [
    "Hello, my name is",
]
```

然后用这组参数启动 `VllmRunner`：

```python
VllmRunner(
    "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
    max_model_len=4096,
    tensor_parallel_size=2,
    enable_expert_parallel=False,
    quantization="ascend",
    gpu_memory_utilization=0.9,
    distributed_executor_backend="mp",
    cudagraph_capture_sizes=[1, 2, 4, 8],
)
```

源码位置：`tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py:24`。

## 它到底验证什么

断言是：

```python
assert outputs[0][1]
```

这句话不检查回答是否聪明，也不检查语义是否完美。它只验证：引擎能加载模型、跑完 prompt、生成文本，并返回非空字符串。

## 为什么 shell 工作流这么长

shell 工作流不是模型逻辑本身，而是运行环境准备：

- SSH：进入有 Ascend NPU 的机器。
- Docker：进入装好 CANN、torch-npu、vLLM 和 vLLM Ascend 的容器。
- 环境变量：控制离线加载、日志等级和 worker 启动方式。
- `npu-smi info`：查看哪些 NPU 空闲。
- `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`：限制当前进程能看到哪些 NPU。
- `pytest -sv ...`：启动真正的 Python 测试。

模型逻辑从 pytest 调用 `VllmRunner.generate_greedy()` 开始。

## 真实运行结果

目标 pytest 已通过：

```text
1 passed, 17 warnings in 228.68s (0:03:48)
```
