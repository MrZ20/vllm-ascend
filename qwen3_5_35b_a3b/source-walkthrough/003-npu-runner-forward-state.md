# 003. 逐行拆 `NPUModelRunner.execute_model`

本章是最重要的源码讲解之一。它回答：

```text
SchedulerOutput 进入 Ascend worker 后，怎么变成模型 forward 的输入？
```

源码入口：

```text
vllm_ascend/worker/model_runner_v1.py:1970
```

## 1. 入口检查：不能跳过 sampling

关键代码：

```python
if self.execute_model_state is not None:
    raise RuntimeError("sample_tokens() must be called after execute_model() returns None.")
```

含义：`execute_model()` 和 `sample_tokens()` 是一对。

```text
execute_model() 负责 forward 和 logits
sample_tokens() 负责从 logits 选 token
```

如果上一次 forward 的 logits 还没采样，又进来下一次 forward，状态就乱了，所以直接报错。

## 2. 读取本步 token 数

关键代码：

```python
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
```

本次实测：

```text
prefill = 5
decode = 1
```

这行是连接 scheduler 和 runner 的关键：scheduler 告诉 worker 当前到底有多少真实 token 要跑。

## 3. `_update_states()` 更新 persistent batch 状态

关键代码：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

这里更新的是 runner 内部长期维护的 request 状态，例如：

```text
当前 batch 有哪些 request
每个 request 已经计算多少 token
token_ids_cpu 缓冲里有什么
KV cache block 分配状态
```

这不是 Transformer 数学，但没有它，后面就不知道该从哪里取 token，也不知道 K/V 写到哪里。

## 4. 把每个 request 的本步 token 数变成数组

关键代码：

```python
req_ids = self.input_batch.req_ids
tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
```

本次只有一个 request：

```text
prefill: num_scheduled_tokens_np = [5]
decode:  num_scheduled_tokens_np = [1]
```

如果有多个 request，可能是：

```text
[12, 1, 1, 32]
```

表示一个 batch 里混合了 prefill 和 decode。

## 5. `_prepare_inputs()` 做 token 和 position 准备

关键代码：

```python
logits_indices, spec_decode_metadata, total_num_scheduled_tokens = self._prepare_inputs(...)
```

源码入口：

```text
vllm_ascend/worker/model_runner_v1.py:789
```

它做几件非常关键的事：

```text
1. 计算 req_indices
2. 计算 positions
3. 从 token_ids_cpu 里 gather 本步 input_ids
4. 写 query_start_loc
5. 计算 optimistic seq_lens
6. 准备 logits_indices
```

### position 公式

源码：

```text
vllm_ascend/worker/model_runner_v1.py:834
```

公式：

```text
positions = num_computed_tokens + query_pos
```

本次：

```text
prefill positions = [0, 1, 2, 3, 4]
decode step 1 positions = [5]
decode step 2 positions = [6]
```

## 6. `_determine_batch_execution_and_padding()` 决定图模式和 padding

关键代码：

```python
cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats =
    self._determine_batch_execution_and_padding(...)
```

源码入口：

```text
vllm_ascend/worker/model_runner_v1.py:2956
```

它的输入是真实 token 数：

```text
num_tokens_unpadded = 5
```

输出可能是 padding 后的执行形状：

```text
batch_desc.num_tokens = 8
```

本次为什么是 8？

```text
cudagraph_capture_sizes = [1, 2, 4, 8]
5 不在 capture size 里
下一个合适形状是 8
```

所以：

```text
真实 token = 5
执行形状 = 8
```

## 7. `_build_attention_metadata()` 构造 kernel 需要的索引信息

关键代码：

```python
attn_metadata, spec_decode_common_attn_metadata =
    self._build_attention_metadata(...)
```

源码入口：

```text
vllm_ascend/worker/model_runner_v1.py:3046
```

metadata 包含：

```text
query_start_loc
seq_lens
block_table_tensor
slot_mapping
causal=True
is_prefilling
num_input_tokens
positions
```

这些东西让 attention kernel 知道：

```text
每个 token 属于哪个 request
每个 token 在序列中的位置
历史 K/V 在 KV cache 的哪个 block
哪些 token 可以看见哪些历史
padding 到哪里结束
```

没有 metadata，attention 公式本身能算，但会把 request 边界和 cache 位置搞错。

## 8. `_preprocess()` 产出模型 forward 参数

关键代码：

```python
input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output =
    self._preprocess(...)
```

源码入口：

```text
vllm_ascend/worker/model_runner_v1.py:1396
```

本次实测：

```text
prefill:
  inputs_embeds shape = (8, 2048)
  positions shape 受 M-RoPE 影响可能是 (3, 8)

decode:
  inputs_embeds shape = (1, 2048)
```

这里的 `2048` 来自模型配置 `hidden_size`。

## 9. `set_ascend_forward_context()` 设置 Ascend 上下文

关键代码：

```python
with set_ascend_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
    aclgraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    model_instance=self.model,
    input_ids=input_ids,
):
    hidden_states = self._model_forward(...)
```

这一步把 metadata、padding 数、真实 token 数、图模式等放进全局 forward context。

后面的 attention、MoE、量化算子可以从 context 中读取当前 batch 的运行时信息。

## 10. `_model_forward()` 调真实模型

关键代码：

```python
hidden_states = self._model_forward(
    num_tokens_padded,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
    **model_kwargs
)
```

这才真正进入 Qwen3.5 模型。

输入形状：

```text
prefill: [8, 2048]
decode:  [1, 2048]
```

输出仍然是 hidden states：

```text
hidden_states shape = [num_tokens_padded, 2048]
```

## 11. `compute_logits()` 只对需要采样的位置算词表分数

关键代码：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

本次：

```text
sample_hidden_states shape = [1, 2048]
logits shape = [1, 248320]
```

这里解释了为什么 prefill 有 5 个真实 token、padding 到 8，但 logits 只有 `[1, vocab]`：

```text
生成下一个 token 时，只需要最后一个有效位置的 hidden state
```

## 12. 保存 `ExecuteModelState`

关键代码：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    ...
    positions,
    ...
)
return None
```

这表示 forward 完成，但采样还没完成。

下一步 EngineCore 看到 `model_output is None`，就会调用：

```text
model_executor.sample_tokens(...)
```

进入 sampling 章节。
