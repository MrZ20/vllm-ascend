# 003 - prompt 如何变成输出

## 第 1 步：文本

输入是：

```text
Hello, my name is
```

这是人能读懂的文本。模型不能直接在文本上计算，它只能处理数字。

## 第 2 步：token ID

运行追踪：

```text
prompt_token_ids=[9419, 11, 821, 803, 369]
```

token ID 是 tokenizer 词表里的整数编号。你可以把它理解成字典索引。模型真正看到的是这些数字，不是原始字符串。

## 第 3 步：prefill

运行追踪：

```text
SchedulerOutput(total_num_scheduled_tokens=5)
_prepare_inputs(... head=[5])
_preprocess(... Tensor(shape=(8, 2048), dtype=torch.bfloat16, device=npu:0))
```

第一次模型 step 会读完整 prompt 的 5 个 token，这个阶段叫 prefill。它会建立这条 request 的初始内部状态和 cache。

为什么 prompt 只有 5 个 token，张量却是 `(8, 2048)`？

- `5` 是真实 prompt 长度。
- `8` 是为了匹配 graph capture size 做的 padding。
- `2048` 是这个运行中每个 TP rank 上的 hidden size。

## 第 4 步：logits

运行追踪：

```text
logits=Tensor(shape=(1, 248320), dtype=torch.bfloat16, device=npu:0)
```

`logits` 是选择下一个 token 前的原始分数。

形状解释：

- `1`：当前只有 1 条活跃 request 要生成下一个 token。
- `248320`：词表大小。模型对每个可能的下一个 token 都给一个分数。

## 第 5 步：greedy sampling

测试使用：

```python
SamplingParams(temperature=0.0, max_tokens=5)
```

sampler 源码里 greedy 选择是：

```python
return logits.argmax(dim=-1).view(-1)
```

源码位置：`/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:239`。

小白解释：选择分数最高的 token。

## 第 6 步：decode 循环

prefill 后，每一步只处理一个新 token：

```text
SchedulerOutput(total_num_scheduled_tokens=1)
_prepare_inputs(... head=[1])
_preprocess(... Tensor(shape=(1, 2048), dtype=torch.bfloat16, device=npu:0))
_sample(logits=Tensor(shape=(1, 248320), ...))
```

这个过程重复，直到生成满 `max_tokens=5` 个新 token。

## 第 7 步：最终文本

运行结果：

```text
prompt token ID:   [9419, 11, 821, 803, 369]
生成 token ID:    [498, 7525, 3855, 1089, 321]
最终 token ID:    [9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321]
最终文本:         Hello, my name is [Your Name], and
```

最终文本就是原始 prompt 加上 5 个生成 token 解码后的文本。
