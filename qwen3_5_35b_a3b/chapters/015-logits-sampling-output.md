# 015. Logits、Sampling 和最终文本

本章回答：模型最后得到 `[1, 248320]` logits 后，怎么选出一个 token？为什么本测试输出是确定的？

先说大白话：模型最后不是直接吐出文字，而是给词表中 248,320 个 token 分别打一个分数。greedy sampling 就是选分数最高的 token。选中后，把这个 token 接到已有序列末尾，再进入下一轮 decode。

## 1. 本次实测输出

```text
prompt token IDs = [9419, 11, 821, 803, 369]
generated token IDs = [498, 7525, 3855, 1089, 321]
final token IDs = [9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321]
final text = "Hello, my name is [Your Name], and"
```

logits 实测形状：

```text
logits = Tensor(shape=(1, 248320), dtype=torch.bfloat16, device=npu:0)
Sampler.sample(logits=Tensor(shape=(1, 248320), dtype=torch.float32, device=npu:0))
```

`1` 表示当前只有一个活跃序列要采样。`248320` 是词表大小。

## 2. logits 从哪里来

源码入口：

- `vllm_ascend/worker/model_runner_v1.py:2375`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:353`

runner 里：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

模型里：

```text
return self.logits_processor(self.lm_head, hidden_states)
```

数学上：

```text
logits = h @ W_vocab^T
```

本模型：

```text
h shape = [1, 2048]
W_vocab shape = [248320, 2048]
logits shape = [1, 248320]
```

每一个 logit 对应一个候选 token。

## 3. logits 不是概率

logits 是未归一化分数，可以是任意实数：

```text
[-3.2, 0.1, 7.8, 2.0, ...]
```

如果要把它变成概率，需要 softmax：

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

但 greedy sampling 不需要真的计算完整概率，因为：

```text
argmax(softmax(logits)) = argmax(logits)
```

softmax 不改变最大元素的位置。

## 4. `temperature=0.0` 为什么表示 greedy

测试里：

```python
greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
```

源码入口：

- `tests/e2e/conftest.py:1083`
- `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:228`
- `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:240`
- `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:243`

上游 sampler 中，greedy 的核心：

```text
def greedy_sample(logits):
    return logits.argmax(dim=-1).view(-1)
```

sampling 主流程会判断当前请求是否 all greedy。如果是，就直接返回 greedy token。

因此本测试的每一步：

```text
next_token = argmax(logits[0, :])
```

没有随机性，没有 top-p 抽样，没有温度扰动。

## 5. temperature、top-k、top-p 是什么

虽然本测试是 greedy，但理解这几个词有助于知道它“没做什么”。

### temperature

温度缩放：

```text
logits' = logits / temperature
```

- `temperature > 1`：分布更平，随机性更强。
- `0 < temperature < 1`：分布更尖，偏向高分 token。
- `temperature = 0`：通常作为 greedy 特殊情况处理。

### top-k

只保留分数最高的 k 个 token，再采样。

```text
candidate = top_k(logits, k)
```

### top-p

也叫 nucleus sampling。按概率从高到低累计，只保留累计概率达到 `p` 的最小集合。

```text
sort probabilities desc
keep smallest prefix whose sum >= p
```

本测试参数追踪看到：

```text
SamplingParams({'temperature': 0.0, 'max_tokens': 5, 'top_p': 1.0, 'top_k': 0})
```

所以它实际不做随机截断，核心是 greedy。

## 6. 为什么本测试输出确定

满足几个条件：

- `temperature=0.0`。
- 没有启用随机采样。
- prompt 固定。
- 权重固定。
- 推理环境和 kernel 行为稳定。

所以每一步选择同一个最大 logit，输出 token 序列稳定为：

```text
[498, 7525, 3855, 1089, 321]
```

如果改成随机采样，比如 `temperature=0.8, top_p=0.95`，同一个 prompt 可能每次生成不同结果。

## 7. token ID 如何变回文本

tokenizer 有两个方向：

```text
encode: text -> token IDs
decode: token IDs -> text
```

本次：

```text
[498, 7525, 3855, 1089, 321] -> " [Your Name], and"
```

注意第一个生成 token `498` 解码出来包含前导空格。这是常见现象：很多 tokenizer 会把“空格 + 词”编码成一个 token。

最终 `VllmRunner._finalize_generate_outputs()` 做：

```text
req_sample_output_ids.append(prompt_ids + output_ids)
req_sample_output_strs.append(prompt_str + output_str)
```

源码入口：`tests/e2e/conftest.py:987`

所以返回给测试的是完整 token 序列和完整文本，而不是只返回新生成部分。

## 8. request 什么时候结束

本测试指定：

```text
max_tokens = 5
```

所以最多生成 5 个新 token。生成过程还可能因为 stop token、EOS token、stop string 等条件提前结束，但这个测试只关心非空输出。

vLLM offline loop 中：

```text
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
    if output.finished:
        outputs.append(output)
```

源码入口：`/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:573`

本次请求在生成 5 个 token 后 finished，然后 `_run_engine` 返回最终 `RequestOutput`。

## 9. 为什么测试只 assert 非空

测试断言：

```python
assert outputs[0][1]
```

这不是精度测试。它更像 smoke/e2e 测试，验证：

- 模型能被加载。
- Ascend quantization 路径能初始化。
- TP=2 worker 能协同。
- Qwen3.5 patch 没有 forward 崩溃。
- MoE W8A8 dynamic 路径能跑通。
- prefill/decode/graph capture/sampling 能完成。
- 最终能得到非空文本。

如果要做精度测试，通常会固定输出 token 或对比参考实现；这个用例没有做。

## 10. 本章完整链路

```text
hidden_states after final layer
  -> logits_indices
sample_hidden_states [1, 2048]
  -> lm_head / logits_processor
logits [1, 248320]
  -> sampler sees temperature=0.0
next_token = argmax(logits)
  -> append to request output
  -> detokenize generated token IDs
  -> final RequestOutput
  -> VllmRunner._finalize_generate_outputs
  -> pytest assert output text non-empty
```

## 11. 初学者检查点

1. logits 和概率有什么区别？
2. 为什么 greedy 可以直接对 logits 做 argmax？
3. `temperature=0.0` 为什么不是把 logits 除以 0？
4. 为什么生成 token 的第一个文本片段可能带空格？
5. 为什么这个测试通过不代表模型输出质量一定好？

答案要点：

- logits 是原始分数，softmax 后才是概率。
- softmax 不改变最大值位置。
- sampler 把 `temperature=0` 当 greedy 特殊分支处理。
- tokenizer 常把空格和词打包成一个 token。
- 这个测试只验证链路跑通和输出非空，不做质量/准确率断言。
