# 002. 从 EngineCore 到 NPUModelRunner

本章讲第二段源码跳转：

```text
LLMEngine.add_request()
  -> EngineCore
  -> scheduler.schedule()
  -> model_executor.execute_model()
  -> NPUModelRunner.execute_model()
```

这段是 vLLM 的核心调度层。它决定当前 step 到底要让模型处理几个 token。

## 1. `LLMEngine.add_request()` 处理输入

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py:218
```

关键逻辑：

```python
request = self.input_processor.process_inputs(...)
self.input_processor.assign_request_id(request)
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

这里发生几件事：

```text
1. raw prompt 被 process_inputs 处理
2. request_id 写进 request
3. output_processor 记录这个 request 的输出状态
4. EngineCore 接收 request，等待调度
```

动态追踪中看到 tokenizer 结果：

```text
prompt_token_ids = [9419, 11, 821, 803, 369]
```

所以进入调度层时，prompt 已经从字符串变成 token ID。

## 2. `LLMEngine.step()` 不直接调模型

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py:296
```

`LLMEngine.step()` 主要做：

```text
从 EngineCore 获取输出
交给 output_processor
处理停止字符串和完成状态
返回 RequestOutput
```

模型执行更深一层，在 `EngineCore.step()`。

## 3. `EngineCore.step()` 是调度和执行的核心

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:479
```

核心代码：

```python
if not self.scheduler.has_requests():
    return {}, False

scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)

engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

用中文拆开：

```text
1. scheduler 看看有没有未完成请求。
2. scheduler 决定本步要跑哪些 token。
3. model_executor 把这个任务发给 worker/model runner。
4. 如果 execute_model 只完成 forward，返回 None，就继续调用 sample_tokens。
5. scheduler 根据采样结果更新 request 状态。
```

## 4. `SchedulerOutput` 是这一层最重要的数据结构

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/core/sched/output.py:181
```

关键字段：

```python
num_scheduled_tokens: dict[str, int]
total_num_scheduled_tokens: int
scheduled_new_reqs
scheduled_cached_reqs
finished_req_ids
num_common_prefix_blocks
kv_connector_metadata
```

本测试最关键的是：

```text
prefill:
  total_num_scheduled_tokens = 5

decode:
  total_num_scheduled_tokens = 1
```

为什么？

```text
prefill 要处理完整 prompt 的 5 个 token
decode 每步只处理上一步新生成的 1 个 token
```

## 5. `model_executor.execute_model()` 怎么到 Ascend

`EngineCore` 调的是抽象的：

```python
self.model_executor.execute_model(scheduler_output, non_block=True)
```

由于当前平台是 Ascend，worker 侧实际会进入：

```text
vllm_ascend/worker/model_runner_v1.py:1970
NPUModelRunner.execute_model()
```

这就是上游 vLLM 和 vLLM Ascend 的交接点。

上游 vLLM 不关心 NPU 细节，只交出一个 `SchedulerOutput`。vLLM Ascend 接手后负责：

```text
把 request token 整理成 NPU tensor
构造 attention metadata
决定 graph padding
设置 Ascend forward context
调用真实模型
计算 logits
执行 Ascend sampler
```

## 6. 控制流和数据流对应关系

控制流：

```text
EngineCore.step()
  -> scheduler.schedule()
  -> model_executor.execute_model()
  -> model_executor.sample_tokens()
  -> scheduler.update_from_output()
```

数据流：

```text
request token ids
  -> SchedulerOutput
  -> input_ids / positions / metadata
  -> hidden states
  -> logits
  -> sampled token
  -> request 状态更新
```

到本章末尾，我们已经到达 `NPUModelRunner.execute_model()`，下一章开始逐行拆这个函数。
