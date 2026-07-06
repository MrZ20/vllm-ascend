# 001. 从 pytest 到 offline engine

本章讲第一段源码跳转：

```text
pytest 测试文件
  -> VllmRunner
  -> LLM.generate()
  -> OfflineInferenceMixin
```

这段代码还没有进入 NPU，也没有进入 Transformer。它的任务是把一个 Python 字符串变成 vLLM engine 里的 request。

## 1. pytest 文件只定义了很少的业务输入

源码：

```text
tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py:24
```

关键内容：

```python
EXAMPLE_PROMPTS = [
    "Hello, my name is",
]
```

测试函数里创建 `VllmRunner`：

```python
with VllmRunner(
    "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
    max_model_len=4096,
    tensor_parallel_size=2,
    enable_expert_parallel=False,
    quantization="ascend",
    gpu_memory_utilization=0.9,
    distributed_executor_backend="mp",
    cudagraph_capture_sizes=[1, 2, 4, 8],
) as vllm_model:
    outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, max_tokens=5)
```

这段代码的输入是：

```text
prompt 字符串
模型名
并行配置
量化配置
生成参数 max_tokens=5
```

输出是：

```text
outputs: list[tuple[list[int], str]]
```

最后测试只断言：

```python
assert outputs[0][1]
```

也就是输出文本非空。

## 2. `VllmRunner.__init__()` 创建上游 `LLM`

源码：

```text
tests/e2e/conftest.py:968
```

`VllmRunner` 把 pytest 参数直接传给：

```python
self.model = LLM(...)
```

关键参数进入上游 vLLM：

```text
model = Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp
tensor_parallel_size = 2
max_model_len = 4096
quantization = ascend
distributed_executor_backend = mp
cudagraph_capture_sizes = [1, 2, 4, 8]
```

这一步开始加载配置、初始化 engine、创建 worker。它属于“运行系统搭建”，不是处理 prompt。

## 3. `generate_greedy()` 创建采样参数

源码：

```text
tests/e2e/conftest.py:1083
```

关键代码：

```python
greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
```

本次：

```text
temperature = 0.0
max_tokens = 5
```

`temperature=0.0` 的意义是：不随机采样，后面 sampler 会走 greedy，即取 logits 最大的 token。

## 4. `get_inputs()` 把字符串包成 `TextPrompt`

源码：

```text
tests/e2e/conftest.py:1003
```

如果 prompt 是字符串：

```python
text_prompt_kwargs["prompt"] = prompt
inputs.append(TextPrompt(**text_prompt_kwargs))
```

本次得到：

```text
TextPrompt(prompt="Hello, my name is", multi_modal_data=None)
```

这里仍然没有模型计算，也没有 logits。它只是统一输入格式。

## 5. `VllmRunner.generate()` 调上游 `LLM.generate()`

源码：

```text
tests/e2e/conftest.py:1035
```

关键代码：

```python
req_outputs = self.model.generate(inputs, sampling_params=sampling_params)
```

从这里进入上游 vLLM。

## 6. offline engine 添加 request

上游源码：

```text
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:326
```

`_run_completion()` 做两件事：

```python
self._add_completion_requests(...)
return self._run_engine(...)
```

也就是：

```text
先把 prompt 加进 engine
再循环执行 engine
```

继续看：

```text
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:523
```

`_render_and_add_requests()` 遍历 prompts，每个 prompt 调：

```python
request_id = self._add_request(prompt, params[i], ...)
```

再看：

```text
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:552
```

`_add_request()` 里如果是 `SamplingParams`：

```python
params.output_kind = RequestOutputKind.FINAL_ONLY
request_id = str(next(self.request_counter))
return self.llm_engine.add_request(...)
```

`FINAL_ONLY` 表示最终只返回完整输出，不每一步都把中间 token 返回给 pytest。

## 7. engine 循环开始

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:594
```

核心循环：

```python
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
```

这就是生成式推理的外层循环。每一次 `step()` 通常会完成：

```text
调度一批 token
执行模型
采样下一个 token
更新 request 状态
```

本测试 `max_tokens=5`，所以这个循环会持续到生成 5 个新 token 或触发停止条件。

## 8. 本章数据变化

到本章末尾，数据从：

```text
"Hello, my name is"
```

变成了一个 engine request：

```text
request_id = "0-..."
prompt = TextPrompt(...)
sampling_params = temperature=0.0, max_tokens=5
output_kind = FINAL_ONLY
```

真正的 tokenization 和 scheduler 处理在下一段代码里继续。
