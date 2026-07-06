# 000 - 整体推理流程

本章只讲一件事：一句文本怎样一步步变成模型输出。

```text
prompt 字符串
  -> tokenizer
  -> token ID
  -> vLLM request
  -> scheduler
  -> Ascend model runner
  -> Qwen3.5 forward
  -> logits
  -> greedy sampler
  -> 新 token ID
  -> 解码后的文本
```

本测试中的 prompt 是：

```python
"Hello, my name is"
```

生成模式是 greedy，因为 `VllmRunner.generate_greedy()` 创建了：

```python
SamplingParams(temperature=0.0, max_tokens=5)
```

greedy 的意思是：每一步都从词表分数里选择最高分 token，不做随机采样。
