# 005. 运行环境、进程和设备选择

本章回答一个容易被忽略但非常关键的问题：为什么同一段 Python 推理代码，必须先 SSH、进 Docker、设置环境变量、挑 NPU 卡，才能正确跑起来？

先说大白话：模型的数学计算发生在 NPU 上，但在模型真正算第一个 token 之前，系统要先决定“模型文件从哪里找、worker 怎么启动、每个 worker 用哪张卡、分布式通信怎么建起来”。这些决定不改变 prompt 的语义，却会决定推理能不能启动、使用哪些硬件、以及后面看到的 tensor 位于哪个 device。

## 1. 这一步的输入和输出

输入是你的运行环境：

```bash
export VLLM_LOGGING_LEVEL=ERROR VLLM_USE_MODELSCOPE=True HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
pytest -sv tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py
```

输出不是模型文本，而是一组运行时事实：

```text
模型文件查找方式 = 离线缓存 + ModelScope
worker 启动方式 = multiprocessing spawn
可见 NPU = 物理卡 4,5,6,7 映射成进程内逻辑卡
tensor parallel rank 数 = 2
expert parallel = false
```

这一步之后，真实模型输入仍然只是：

```text
"Hello, my name is"
```

也就是说，环境准备阶段没有生成 token，也没有跑 Transformer。它只是把“能跑模型的机器形态”搭好。

## 2. 每个环境变量在控制什么

### `VLLM_LOGGING_LEVEL=ERROR`

控制 vLLM 日志等级。它让日志更安静，但不会改变模型计算。

这类变量只影响“你看到多少日志”，不应该影响生成结果。调试教学时我们会额外加追踪脚本，因为正常测试为了稳定通常不会打印每一层 tensor。

### `VLLM_USE_MODELSCOPE=True`

告诉相关模型文件查找逻辑优先走 ModelScope。vLLM Ascend 的量化工具里，`get_model_file()` 会根据这个开关选择 `modelscope.hub.file_download.model_file_download` 或 Hugging Face 下载接口。

本次设置它的原因是模型名是：

```text
Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp
```

远端容器里实际缓存目录是：

```text
/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp
```

所以这里不是从公网重新下载模型，而是在离线缓存中解析模型文件。

### `HF_HUB_OFFLINE=1`

这个变量让 Hugging Face hub 处于离线模式。上游 vLLM 在初始化参数时会检测这个状态：如果离线，就把模型 ID 替换成本地模型路径。

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/engine/arg_utils.py:760`
- `/Users/lonng/Mrz20/vllm/vllm/engine/arg_utils.py:767`

这一步很重要，因为后续加载 `config.json`、tokenizer、权重分片时，不能依赖网络。

### `VLLM_WORKER_MULTIPROC_METHOD=spawn`

这个变量决定 worker 进程的启动方式。

`spawn` 的含义是：子进程不是简单复制当前 Python 进程，而是重新启动一个 Python 解释器，再导入必要模块。这样更干净，但也有一个限制：入口代码必须是一个真实可导入的文件。

这解释了一个动态追踪中的现象：用 `python -` 从标准输入跑脚本时，worker 进程会尝试重新导入主程序，但 `<stdin>` 不是一个稳定的文件路径，所以会失败。后来把追踪逻辑保存成：

```text
qwen3_5_35b_a3b/trace_qwen_run.py
```

再运行就可以被 `spawn` 子进程导入。

### `TORCH_DEVICE_BACKEND_AUTOLOAD=0`

这个变量用于限制 PyTorch 后端自动加载行为，减少容器中后端初始化顺序带来的不确定性。它不直接参与 Transformer 数学，也不改变 logits，但会影响运行时什么时候加载设备后端。

这里要注意：这类变量通常属于“运行稳定性变量”，不是模型算法变量。

### `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`

这行最容易误解。它不是说测试一定会同时用 4 张卡，而是说当前进程家族“只能看见”物理卡 4、5、6、7。

在进程内部，这些物理卡通常会重新编号成逻辑卡：

```text
进程内 npu:0 -> 物理卡 4
进程内 npu:1 -> 物理卡 5
进程内 npu:2 -> 物理卡 6
进程内 npu:3 -> 物理卡 7
```

本测试设置：

```python
tensor_parallel_size=2
```

所以真正创建的是两个 tensor parallel rank。日志中看到的是：

```text
Worker_TP0
Worker_TP1
```

因此“可见 4 张卡”和“本测试 TP=2”并不矛盾：前者是可用设备池，后者是本次模型并行实际使用的 rank 数。

源码入口：

- `vllm_ascend/worker/worker.py:440`：读取 `torch.npu.device_count()`
- `vllm_ascend/worker/worker.py:445`：逻辑 device id 映射到可见 device id
- `vllm_ascend/worker/worker.py:450`：`torch.npu.set_device(device)`

## 3. pytest 进程、engine 进程、worker 进程

本测试不是一个单进程脚本直接调用 `model.forward()`。它大致是：

```text
pytest 主进程
  -> VllmRunner 创建 LLM
    -> vLLM engine
      -> multiprocessing worker
        -> Worker_TP0 绑定一张 NPU
        -> Worker_TP1 绑定一张 NPU
```

`distributed_executor_backend="mp"` 表示这里走 multiprocessing，不走 Ray。

为什么需要 worker？因为 tensor parallel rank 要分别在不同设备上执行自己的那部分计算，并通过 HCCL 通信交换必要结果。

## 4. `HCCL_BUFFSIZE=1024` 是什么

测试文件里有：

```python
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
```

`HCCL` 是 Ascend 场景中的集合通信库，类似 GPU 场景里常见的 NCCL。TP=2 时，大矩阵被拆到两个 rank 上，某些层需要 `all-reduce`、`all-gather` 这类通信。

`HCCL_BUFFSIZE` 控制通信 buffer 大小。它不是 Transformer 公式的一部分，但会影响分布式通信能否以期望方式运行。

## 5. 这一步和模型输出有什么关系

从数学上看，环境阶段不会改变：

```text
prompt_token_ids = [9419, 11, 821, 803, 369]
```

也不会直接改变：

```text
logits = model(hidden_states)
```

但它决定了这些计算在哪里发生：

```text
inputs_embeds.device = npu:0
logits.device = npu:0
TP rank = 0/1
```

如果设备选择错了，可能会发生：

- NPU 内存不足；
- worker 绑定到正在被别人使用的卡；
- HCCL 初始化失败；
- 模型文件找不到；
- `spawn` 子进程无法导入追踪脚本。

所以“先选空卡再跑测试”不是形式主义，而是大模型推理在共享 NPU 机器上必须完成的资源隔离。

## 6. 本章小结

这章只建立一个判断标准：

```text
运行环境阶段 = 决定文件、进程、设备、通信
模型推理阶段 = 决定 token、hidden state、logits、输出
```

你以后看日志时要先分清楚：某条日志是在准备运行环境，还是已经进入真实 prompt 的 forward。比如图捕获、dummy run、内存 profiling 都发生在真实 prompt 之前，不能把它们误认为模型正在回答 `"Hello, my name is"`。
