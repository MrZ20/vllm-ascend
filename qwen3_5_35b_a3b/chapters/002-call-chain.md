# 002 - 函数跳转图

本章根据动态追踪解释：从测试代码到 Ascend worker，中间到底跳了哪些函数。

## 1. 测试调用 `generate_greedy`

`tests/e2e/conftest.py:1083` 定义：

```python
def generate_greedy(...):
    greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = self.generate(prompts, greedy_params, ...)
```

运行证据：

```text
VllmRunner.generate_greedy(prompts=['Hello, my name is'], max_tokens=5)
SamplingParams(temperature=0.0, max_tokens=5)
```

小白解释：这里告诉 vLLM“最多生成 5 个新 token，并且不要随机采样”。

## 2. `VllmRunner.generate` 构造 vLLM 输入

`tests/e2e/conftest.py:1035` 先调用 `get_inputs`，再调用 `self.model.generate`。

运行证据：

```text
VllmRunner.get_inputs -> {'multi_modal_data': None, 'prompt': 'Hello, my name is'}
LLM.generate(...)
```

小白解释：原始字符串被包装成 vLLM 统一认识的输入格式。

## 3. vLLM 把输入加入 engine

上游 `offline_utils.py:326` 执行 completion；`offline_utils.py:523` 渲染并添加 request；`offline_utils.py:552` 创建 request ID 并调用 engine `add_request`。

运行证据：

```text
prompt_token_ids=[9419, 11, 821, 803, 369]
LLMEngine.add_request -> '0-b59b0dc7'
```

小白解释：vLLM 把文本变成 token 数字，并给这次任务分配一个 request ID。

## 4. engine 一直 step 到完成

`offline_utils.py:573` 在还有未完成 request 时循环调用 `llm_engine.step()`。

运行证据：

```text
LLMEngine.step #1 -> []
LLMEngine.step #2 -> []
...
LLMEngine.step #5 -> [RequestOutput(...)]
```

小白解释：不是每个 step 都马上返回最终文本。前几个 step 在 worker 侧跑模型和采样，最后一个 step 才拿到完整输出。

## 5. Ascend worker 执行模型

`vllm_ascend/worker/model_runner_v1.py:1970` 是 worker 侧主要入口。

运行证据：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=5))
NPUModelRunner._prepare_inputs(... head=[5])
NPUModelRunner._preprocess(... num_tokens_padded=8)
```

小白解释：第一个真实 step 收到 5 个 prompt token，准备 NPU 张量，并 padding 到 8 这个更适合图执行的大小。

## 6. worker 采样一个 token

`model_runner_v1.py:2421` 从 `execute_model` 保存的状态里取出 logits；`model_runner_v1.py:2475` 调用 `_sample`；`model_runner_v1.py:2634` 调用 sampler。

运行证据：

```text
_sample(logits=Tensor(shape=(1, 248320), dtype=torch.bfloat16, device=npu:0))
Sampler.sample(logits=Tensor(shape=(1, 248320), dtype=torch.float32, device=npu:0))
```

小白解释：模型给每个可能的下一个 token 打分，然后 sampler 选出一个 token。

## 7. 最终输出被拼起来

`tests/e2e/conftest.py:987` 把 prompt token ID 和生成 token ID 拼接，也把 prompt 文本和生成文本拼接。

运行证据：

```text
prompt token ID: [9419, 11, 821, 803, 369]
生成 token ID: [498, 7525, 3855, 1089, 321]
最终文本: 'Hello, my name is [Your Name], and'
```
