# 007. Request、Engine 和 Scheduler 生命周期

本章回答：`VllmRunner.generate_greedy(["Hello, my name is"], max_tokens=5)` 之后，代码是怎么一步步进入模型 forward 的？

先说大白话：vLLM 不是拿到 prompt 就立刻调用一次 `model.forward()` 然后结束。它会把 prompt 包装成 request，放进 engine，由 scheduler 每一步决定“这次该算哪些 token”。第一次算完整 prompt，叫 prefill；之后每次只算新生成的一个 token，叫 decode。

## 1. 测试入口创建了什么

测试文件里真正的业务输入只有：

```python
EXAMPLE_PROMPTS = [
    "Hello, my name is",
]
```

然后：

```python
outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, max_tokens=5)
```

`generate_greedy()` 会创建：

```python
SamplingParams(temperature=0.0, max_tokens=5)
```

含义是：

```text
最多生成 5 个新 token
temperature=0.0，走 greedy，不随机
```

源码入口：

- `tests/e2e/conftest.py:1083`

## 2. prompt 被包装成 `TextPrompt`

`VllmRunner.get_inputs()` 看到输入是字符串，于是构造：

```text
TextPrompt(prompt="Hello, my name is", multi_modal_data=None)
```

源码入口：

- `tests/e2e/conftest.py:1003`

这里还没有 tokenization。它只是把普通 Python 字符串变成 vLLM 能识别的 prompt 对象。

## 3. `LLM.generate()` 把输入交给 offline engine

`VllmRunner.generate()` 调用：

```python
self.model.generate(inputs, sampling_params=sampling_params)
```

源码入口：

- `tests/e2e/conftest.py:1035`

后面进入上游 vLLM offline 执行路径：

```text
LLM.generate
  -> OfflineInferenceMixin._run_completion
  -> _render_and_add_requests
  -> _add_request
  -> LLMEngine.add_request
```

源码地图里对应：

- `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:326`
- `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:523`
- `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:552`

动态追踪中 request ID 是：

```text
0-b59b0dc7
```

request ID 的作用是让 scheduler、worker、output processor 都能指向同一条生成任务。

## 4. tokenizer 在 request 阶段产生 token ID

动态追踪看到：

```text
prompt_token_ids = [9419, 11, 821, 803, 369]
```

这说明：

```text
"Hello, my name is" -> 5 个 token
```

这 5 个 token 是后续 prefill 的真实工作量。

要注意：tokenizer 发生在模型 forward 之前。模型 forward 吃的是整数 token ID 或 embedding，不直接吃字符串。

## 5. engine 循环不是只跑一次

offline engine 会不断执行：

```text
while engine has unfinished requests:
    llm_engine.step()
```

在 vLLM v1 engine 里，`LLMEngine.step()` 大致做：

```text
1. 从 EngineCore 拿输出
2. output_processor 处理上一步输出
3. 把完成的 request abort/free
4. 返回 RequestOutput
```

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py:296`

EngineCore 的核心 step 更接近模型执行：

```text
if scheduler.has_requests():
    scheduler_output = scheduler.schedule(...)
    future = model_executor.execute_model(scheduler_output)
    model_output = future.result()
    if model_output is None:
        model_output = model_executor.sample_tokens(...)
    scheduler.update_from_output(scheduler_output, model_output)
```

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:479`
- `/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:490`

## 6. `SchedulerOutput` 是 worker 的任务单

`SchedulerOutput` 里最关键的字段之一是：

```python
total_num_scheduled_tokens: int
```

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/v1/core/sched/output.py:181`

它表示当前 step 所有 request 一共要跑多少个 token。

本次动态追踪：

```text
prefill:
  SchedulerOutput(total_num_scheduled_tokens=5)

decode:
  SchedulerOutput(total_num_scheduled_tokens=1)
```

为什么第一次是 5？因为 prompt 有 5 个 token，还没有任何历史 cache。

为什么后面是 1？因为每次 decode 只需要把上一步新生成的 token 喂进去，历史 token 的 K/V 已经在 cache 里。

## 7. worker 收到任务单后做什么

Ascend worker 侧入口是：

```text
NPUModelRunner.execute_model(scheduler_output)
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:1970`

它的关键步骤是：

```text
1. _update_states(scheduler_output)
2. 读取 total_num_scheduled_tokens
3. _prepare_inputs(...)
4. _determine_batch_execution_and_padding(...)
5. _build_attention_metadata(...)
6. _preprocess(...)
7. set_ascend_forward_context(...)
8. _model_forward(...)
9. compute_logits(...)
10. 保存 ExecuteModelState
```

采样并不直接在 `execute_model()` 里返回。它把 logits 等状态保存到：

```text
self.execute_model_state
```

然后 engine 再调用：

```text
NPUModelRunner.sample_tokens(...)
```

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2399`
- `vllm_ascend/worker/model_runner_v1.py:2421`

这种拆分的好处是 forward 和 sampling 可以在 engine 调度层更灵活地组织，尤其是异步调度、structured output、speculative decoding 等复杂场景。

## 8. 一次完整生成的循环

本次 `max_tokens=5`，所以从 token 角度看是：

```text
初始 prompt:
  [9419, 11, 821, 803, 369]

step 0 prefill:
  输入 5 个 prompt token
  输出第 1 个新 token = 498

step 1 decode:
  输入 token 498
  输出第 2 个新 token = 7525

step 2 decode:
  输入 token 7525
  输出第 3 个新 token = 3855

step 3 decode:
  输入 token 3855
  输出第 4 个新 token = 1089

step 4 decode:
  输入 token 1089
  输出第 5 个新 token = 321
```

最终：

```text
generated token ids = [498, 7525, 3855, 1089, 321]
final text = "Hello, my name is [Your Name], and"
```

注意最后一个生成 token `321` 已经产生了，但因为 `max_tokens=5` 到达上限，engine 不会再用它继续 decode 第 6 个 token。

## 9. 为什么 scheduler 这么重要

如果只有一个 prompt，你可能会觉得 scheduler 多余。但 vLLM 的核心能力是服务大量并发 request。

scheduler 要同时决定：

- 哪些新 request 可以开始 prefill；
- 哪些老 request 继续 decode；
- 每个 request 本步跑几个 token；
- KV cache block 怎么分配；
- 是否需要释放已经结束的 request；
- 是否可以把多个 request 拼成一个 batch；
- 是否要为了 graph capture 做 padding。

所以 scheduler 不是“模型的一部分”，但它决定模型每一步吃到的数据形状。

本次例子里只有一个 request，所以形状很简单：

```text
prefill: total_num_scheduled_tokens = 5
decode:  total_num_scheduled_tokens = 1
```

真实服务中，可能是：

```text
request A prefill 20 tokens
request B decode 1 token
request C decode 1 token
request D prefill 128 tokens
```

这些会被 scheduler 合成一个 worker step。

## 10. 本章小结

这一章的核心链路是：

```text
Python 字符串
  -> TextPrompt
  -> tokenized request
  -> LLMEngine.add_request
  -> scheduler.schedule
  -> SchedulerOutput
  -> NPUModelRunner.execute_model
  -> NPUModelRunner.sample_tokens
  -> RequestOutput
```

下一章会继续进入 `execute_model()` 内部，解释 `positions`、`query_start_loc`、`block_table`、`slot_mapping` 和 attention metadata 为什么是 attention kernel 正确运行的必要条件。
