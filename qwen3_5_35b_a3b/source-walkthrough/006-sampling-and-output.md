# 006. 从 logits 到 RequestOutput

本章讲最后一段源码：模型已经算出 logits 后，怎么选出 token，并最终回到 pytest 的 `outputs`。

## 1. `execute_model()` 只把 logits 保存起来

上一章到 `NPUModelRunner.execute_model()` 结尾：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    ...
)
return None
```

源码：

```text
vllm_ascend/worker/model_runner_v1.py:2399
```

这说明 forward 已经完成，但采样还没完成。

## 2. EngineCore 看到 None 后调用 `sample_tokens()`

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:497
```

关键逻辑：

```python
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

这和 `execute_model_state` 对上了：

```text
execute_model 先保存 logits
sample_tokens 再读取 logits
```

## 3. logits 的形状

本次动态追踪：

```text
logits shape = [1, 248320]
```

含义：

```text
1        = 当前只有一个位置需要采样
248320   = vocab_size
```

`logits[0, j]` 表示“下一个 token 是词表第 j 个 token”的未归一化分数。

logits 不是概率。概率需要：

```text
prob = softmax(logits)
```

但 greedy 不一定真的需要显式 softmax，因为：

```text
argmax(softmax(logits)) = argmax(logits)
```

## 4. temperature=0 走 greedy

测试里：

```python
SamplingParams(temperature=0.0, max_tokens=5)
```

greedy 的选择公式：

```text
next_token_id = argmax(logits[-1])
```

如果是随机采样，通常会有：

```text
logits = logits / temperature
prob = softmax(logits)
next_token = multinomial(prob)
```

但本测试不是随机采样，所以输出是确定性的。

## 5. 本次生成的 token

动态追踪结果：

```text
generated token ids = [498, 7525, 3855, 1089, 321]
```

循环视角：

```text
prefill:
  prompt [9419, 11, 821, 803, 369]
  -> token 498

decode 1:
  token 498
  -> token 7525

decode 2:
  token 7525
  -> token 3855

decode 3:
  token 3855
  -> token 1089

decode 4:
  token 1089
  -> token 321
```

达到 `max_tokens=5` 后停止。

## 6. scheduler 更新 request 状态

采样后回到：

```text
EngineCore.step()
```

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:504
```

关键代码：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

这一步会：

```text
把新 token 接到 request
更新已生成 token 数
判断是否达到 max_tokens
决定 request 是否 finished
释放或保留相关状态
```

## 7. offline engine 收集最终输出

源码：

```text
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:594
```

关键代码：

```python
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
    for output in step_outputs:
        if output.finished:
            outputs.append(output)
```

由于 `_add_request()` 设置了：

```text
RequestOutputKind.FINAL_ONLY
```

所以 pytest 最终只拿到完整的 `RequestOutput`。

## 8. `VllmRunner._finalize_generate_outputs()` 拼结果

源码：

```text
tests/e2e/conftest.py:987
```

关键逻辑：

```python
prompt_ids = req_output.prompt_token_ids
output_ids = list(sample.token_ids)
req_sample_output_ids.append(prompt_ids + output_ids)
req_sample_output_strs.append((prompt_str or "") + output_str)
```

所以返回给测试的 token ids 是：

```text
prompt_token_ids + generated_token_ids
```

本次：

```text
[9419, 11, 821, 803, 369] + [498, 7525, 3855, 1089, 321]
= [9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321]
```

文本是：

```text
"Hello, my name is" + " [Your Name], and"
= "Hello, my name is [Your Name], and"
```

## 9. 最终测试为什么只检查非空

测试断言：

```python
assert outputs[0][1]
```

它不检查固定字符串，只检查生成文本非空。

原因通常是：

```text
这个 e2e 用例主要验证模型能在 Ascend + TP=2 + W8A8 + EP=false 路径跑通
不是验证语言内容必须逐字一致
```

真正的准确性测试通常会比较 logits、固定输出、困惑度或特定 benchmark；本用例更像“路径可运行性”测试。
