# 运行手册

## 完整工作流

```bash
ssh a3-node0
ssh node1
docker exec -it zsl_m2m_0612_1 bash
cd /vllm-workspace/vllm-ascend
export VLLM_LOGGING_LEVEL=ERROR VLLM_USE_MODELSCOPE=True HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
pytest -sv tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py
```

## 每条命令在做什么

| 命令 | 作用 | 小白解释 |
| --- | --- | --- |
| `ssh a3-node0` | 进入 A3 机器入口节点。 | 先到有 Ascend 硬件的机器。 |
| `ssh node1` | 进入目标 worker 节点。 | 选择实际运行容器的服务器。 |
| `docker exec -it zsl_m2m_0612_1 bash` | 进入预构建运行环境。 | 这个容器里有 Python、CANN、torch-npu、vLLM 和 vLLM Ascend。 |
| `cd /vllm-workspace/vllm-ascend` | 进入待测仓库。 | 让 pytest 使用这个 checkout。 |
| `export VLLM_LOGGING_LEVEL=ERROR` | 降低日志噪声。 | 只看更重要的日志。 |
| `export VLLM_USE_MODELSCOPE=True` | 使用 ModelScope 相关模型解析路径。 | 从本地 ModelScope 缓存找模型。 |
| `export HF_HUB_OFFLINE=1` | 强制离线。 | 测试时不尝试下载模型。 |
| `export VLLM_WORKER_MULTIPROC_METHOD=spawn` | 使用安全的 worker 启动方式。 | 子进程重新启动 Python，避免继承不安全的 NPU 状态。 |
| `export TORCH_DEVICE_BACKEND_AUTOLOAD=0` | 关闭自动后端加载。 | 避免意外加载后端带来副作用。 |
| `npu-smi info` | 查看卡的使用情况。 | 找空闲 NPU。 |
| `export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` | 限制可见 NPU。 | 告诉进程只能看到这些芯片。 |
| `pytest -sv ...` | 运行目标测试。 | 开始真正验证。 |

## 预期行为

这个测试不会启动 HTTP server，而是在 pytest 进程里构造一个本地 `vllm.LLM` 离线推理引擎。它加载 `Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp`，运行一句 prompt，用 greedy 方式最多生成 5 个 token，只要返回文本非空就通过。
