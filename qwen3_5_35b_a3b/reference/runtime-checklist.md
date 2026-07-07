# 运行和源码排查检查表

这份表用来快速判断：当前看到的日志、源码变量或 tensor 形状，属于推理链路的哪一层。

## 1. 环境层

检查项：

- 是否在目标容器内：`/vllm-workspace/vllm-ascend`
- 是否设置离线模型变量：`VLLM_USE_MODELSCOPE=True`、`HF_HUB_OFFLINE=1`
- 是否设置 worker 启动方式：`VLLM_WORKER_MULTIPROC_METHOD=spawn`
- 是否设置可见 NPU：`ASCEND_RT_VISIBLE_DEVICES=...`
- 是否确认空卡：`npu-smi info`

常见误解：

- 可见 4 张卡不代表测试会用 4 张卡；本测试 TP=2。
- `spawn` 要求追踪脚本是真实 `.py` 文件，不能用 `python -`。

## 2. 模型加载层

检查项：

- `config.json` 是否来自真实缓存目录。
- `quant_model_description.json` 是否存在。
- 权重是否是 10 个 `quant_model_weights-*.safetensors` 分片。
- `quantization="ascend"` 是否传入 `VllmRunner`。

本次确认事实：

```text
model_type = qwen3_5_moe_text
hidden_size = 2048
vocab_size = 248320
num_experts = 256
num_experts_per_tok = 8
model_quant_type = W8A8_DYNAMIC
```

常见误解：

- W8A8 模型不等于所有层都是 int8；本次量化描述里也有 `FLOAT` 条目。

## 3. Request / Scheduler 层

源码入口：

```text
tests/e2e/conftest.py
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py
/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py
```

检查项：

- `generate_greedy()` 是否创建 `SamplingParams(temperature=0.0, max_tokens=5)`。
- `prompt_token_ids` 是否是 `[9419, 11, 821, 803, 369]`。
- prefill 的 `SchedulerOutput.total_num_scheduled_tokens` 是否为 `5`。
- decode 的 `SchedulerOutput.total_num_scheduled_tokens` 是否为 `1`。

常见误解：

- `SchedulerOutput` 是 worker 的任务单，不是模型输出。

## 4. Runner 输入准备层

源码入口：

```text
vllm_ascend/worker/model_runner_v1.py
```

重点函数：

```text
execute_model()
_prepare_inputs()
_determine_batch_execution_and_padding()
_build_attention_metadata()
_preprocess()
```

检查项：

- `num_scheduled_tokens_np` 是否符合本步真实 token 数。
- `positions` 是否按 `num_computed_tokens + query_pos` 计算。
- `num_tokens_padded` 是否来自图捕获形状。
- `logits_indices` 是否只选需要采样的位置。

常见误解：

- `num_tokens_padded=8` 不表示 prompt 有 8 个 token。
- `positions` 在 M-RoPE 下可能是 `[3, N]`。

## 5. Attention metadata 层

检查项：

- `query_start_loc` 是否描述 request 边界。
- `block_table` 是否指向 request 的 KV cache blocks。
- `slot_mapping` 是否指向当前 token 写入 KV cache 的位置。
- `causal=True` 是否存在。
- `is_prefilling` 是否区分 prefill/decode。

常见误解：

- metadata 不是模型参数；它是 kernel 解释 batch 和 cache 的运行时索引。

## 6. Qwen3.5 模型层

源码入口：

```text
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py
vllm_ascend/patch/worker/patch_qwen3_5.py
```

检查项：

- `qwen3_5_moe_text` 是否使用 `Qwen3NextSparseMoeBlock`。
- Ascend patch 是否替换 `Qwen3_5DecoderLayer.forward`。
- `linear_attention` 与 `full_attention` 是否按 layer type 分支。
- full attention 是否走 QKV split、Q/K RMSNorm、M-RoPE、attention、gate、`o_proj`。

常见误解：

- 不能只看上游 `qwen3_5.py` 的结构，还要看 Ascend patch 后的真实 forward。

## 7. FusedMoE / W8A8 层

源码入口：

```text
vllm_ascend/ops/fused_moe/
vllm_ascend/quantization/methods/w8a8_dynamic.py
```

检查项：

- `router_logits` 形状是否为 `[N, 256]`。
- `topk_ids/topk_weights` 形状是否为 `[N, 8]`。
- 是否进入 `select_experts()`。
- 是否进入 `moe_comm_method.fused_experts()`。
- 是否完成 dispatch、MLP、combine。
- W8A8 dynamic 是否有 `quantized_x`、`pertoken_scale`、`weight_scale`。

常见误解：

- EP=false 不等于不跑 MoE。
- FusedMoE 的 fused 是执行优化，不是改变 MoE 数学定义。

## 8. Sampling / Output 层

检查项：

- `sample_hidden_states` 是否为 `[1, 2048]`。
- `logits` 是否为 `[1, 248320]`。
- `temperature=0.0` 是否走 greedy。
- 生成 token 是否是 `[498, 7525, 3855, 1089, 321]`。
- `VllmRunner._finalize_generate_outputs()` 是否把 prompt token 和 generated token 拼接。

常见误解：

- logits 不是概率；softmax 后才是概率。
- greedy 可以直接 `argmax(logits)`，不必显式 softmax。

## 9. 判断一条日志属于哪一层

| 日志或现象 | 所属层 | 解释 |
|---|---|---|
| `Worker_TP0` / `Worker_TP1` | 并行/worker | TP rank 进程 |
| `Loading safetensors checkpoint shards` | 模型加载 | 权重分片加载 |
| `Capturing CUDA graphs` | 图捕获 | Ascend 场景沿用名称 |
| `_model_forward(num_tokens_padded=8192)` | profiling/warmup | 不是用户 prompt |
| `total_num_scheduled_tokens=5` | scheduler | prefill 真实 token 数 |
| `inputs_embeds=(8, 2048)` | runner/model input | padding 后形状 |
| `logits=(1, 248320)` | sampling 前 | 一个位置的词表分数 |

## 10. 读源码时的最短路径

```text
VllmRunner.generate_greedy
  -> LLM.generate
  -> OfflineInferenceMixin._run_completion
  -> LLMEngine.add_request
  -> EngineCore.step
  -> scheduler.schedule
  -> NPUModelRunner.execute_model
  -> _prepare_inputs
  -> _build_attention_metadata
  -> _model_forward
  -> patched Qwen3_5DecoderLayer.forward
  -> Qwen3NextSparseMoeBlock.forward
  -> W8A8/FusedMoE
  -> compute_logits
  -> sample_tokens
  -> RequestOutput
```
