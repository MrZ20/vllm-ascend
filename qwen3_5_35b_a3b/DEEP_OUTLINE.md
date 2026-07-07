# 深度讲解大纲

这份大纲是深度教学路线。它不只讲“调用了哪些函数”，而是沿着一次真实推理的数据流，把每个关键阶段的技术原理、公式和源码入口串起来。

## 阅读方式

每一章都按同一种结构组织：

1. 先用一句话说这一步在做什么。
2. 再列出输入张量、输出张量、形状和 dtype。
3. 再写核心数学公式。
4. 再落到源码函数和关键行。
5. 最后写本次远端实测证据。

这样读的好处是：你不会只背概念，也不会直接被源码淹没；你可以一直问“这个 tensor 从哪里来，到哪里去，为什么形状变成这样”。

## 编号说明

当前完整阅读路径是：

```text
000-004：基础心智模型
005-009：运行时桥梁层
010-015：核心数学、源码和公式深挖
```

## 0. 本次测试的确定事实

- 测试入口：`tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py`
- 模型：`Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp`
- 实际缓存目录：`/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp`
- 权重分片：`quant_model_weights-00001-of-00010.safetensors` 到 `00010`
- 文本模型类型：`qwen3_5_moe_text`
- 词表大小：`248320`
- hidden size：`2048`
- 层数：`40`
- attention heads：`16`
- KV heads：`2`
- head dim：`256`
- MoE 专家数：`256`
- 每个 token 选择专家数：`8`
- MoE intermediate size：`512`
- shared expert intermediate size：`512`
- layer types：`30` 层 `linear_attention`，`10` 层 `full_attention`
- rope：`mrope_section=[11, 11, 10]`，`partial_rotary_factor=0.25`
- 量化描述：`model_quant_type=W8A8_DYNAMIC`
- 语言模型和 MTP 条目统计：`W8A8_DYNAMIC=92250`，`FLOAT=1367`
- 代表性条目：
  - `model.language_model.layers.0.mlp.experts.0.gate_proj.weight = W8A8_DYNAMIC`
  - `model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale = W8A8_DYNAMIC`
  - `model.language_model.layers.0.mlp.experts.0.down_proj.weight = W8A8_DYNAMIC`
  - `model.language_model.layers.0.linear_attn.out_proj.weight = FLOAT`
  - `model.language_model.layers.3.self_attn.o_proj.weight = FLOAT`
  - `lm_head.weight = FLOAT`

## 1. 运行环境和进程模型

问题：SSH、Docker、环境变量、NPU 选择和 `spawn` 到底影响什么？

讲解内容：

- `VLLM_USE_MODELSCOPE=True`、`HF_HUB_OFFLINE=1` 如何影响模型文件查找。
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` 为什么要求追踪脚本必须是真实文件。
- `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 和 `tensor_parallel_size=2` 为什么不矛盾。
- pytest 主进程、vLLM engine、`Worker_TP0/TP1` 的关系。
- `HCCL_BUFFSIZE=1024` 和 TP 通信的关系。

对应文件：`chapters/005-runtime-environment-and-processes.md`

## 2. 模型加载、量化配置和 patch

问题：`quantization="ascend"` 如何把模型接到 Ascend W8A8 路径？

讲解内容：

- `config.json` 决定模型骨架和 tensor 形状。
- `quant_model_description.json` 决定每层是 `W8A8_DYNAMIC` 还是 `FLOAT`。
- `AscendModelSlimConfig` 如何读取量化描述并选择 quant method。
- 为什么 W8A8 模型里仍有部分 attention 和 `lm_head` 是 `FLOAT`。
- `patch_qwen3_5.py` 如何替换上游 Qwen3.5 forward。

对应文件：`chapters/006-model-loading-quant-config-and-patches.md`

## 3. Request、Engine 和 Scheduler

问题：`generate_greedy()` 之后，代码怎么一步步进入 worker forward？

讲解内容：

- `TextPrompt`、tokenization、request ID。
- `SamplingParams(temperature=0.0, max_tokens=5)`。
- `LLMEngine.add_request()` 和 engine step 循环。
- `SchedulerOutput.total_num_scheduled_tokens` 如何区分 prefill=5 和 decode=1。
- `execute_model()` 与 `sample_tokens()` 为什么分成两个阶段。

对应文件：`chapters/007-request-engine-scheduler-lifecycle.md`

## 4. Attention metadata 和位置系统

问题：为什么 attention kernel 需要 positions、block table、slot mapping？

讲解内容：

- `position = num_computed_tokens + query_pos`。
- M-RoPE 为什么可能让 `positions` 变成 `[3, N]`。
- `query_start_loc` 如何描述扁平 token batch 的 request 边界。
- `block_table` 和 `slot_mapping` 如何定位 KV cache。
- causal mask 如何通过 metadata 进入 kernel。
- full attention 和 linear attention/GDN 对 metadata 的不同需求。

对应文件：`chapters/008-attention-metadata-position-and-backend.md`

## 5. 并行、内存和图捕获

问题：为什么真实 prompt 前有 dummy forward，为什么 5 个 token padding 到 8？

讲解内容：

- TP=2 的 column parallel、row parallel 和 HCCL 通信。
- EP=false 对 MoE 通信路径的影响。
- `gpu_memory_utilization=0.9`、内存 profiling 和 KV cache 可用空间。
- KV cache 为什么让 decode 只需处理 1 个新 token。
- `cudagraph_capture_sizes=[1,2,4,8]` 如何导致 prefill 5 padding 到 8。
- 8192/8/4 dummy forward 为什么不是用户 prompt。

对应文件：`chapters/009-parallelism-memory-and-graph-capture.md`

## 6. 端到端数据推导

问题：`"Hello, my name is"` 为什么会生成 `" [Your Name], and"`？

讲解内容：

- 字符串如何变成 token ID。
- token ID 如何变成 embedding。
- embedding 如何经过 40 层模型。
- 每层如何更新 hidden state。
- 最后一位 hidden state 如何变成 `[1, 248320]` logits。
- greedy sampler 如何取 `argmax`。
- 新 token 如何回到 scheduler，进入下一次 decode。

对应文件：`chapters/010-end-to-end-data-derivation.md`

## 7. Prefill 和 Decode

问题：为什么第一次处理 5 个 token，后面每次只处理 1 个 token？

讲解内容：

- prefill 的输入是完整 prompt。
- decode 的输入是上一步刚生成的新 token。
- KV cache 保存历史 K/V，避免重复计算历史 token。
- vLLM scheduler 用 `SchedulerOutput.total_num_scheduled_tokens` 表示本步真正要跑几个 token。
- 本测试中 prompt 长度是 5，prefill padding 到 8；decode 每步是 1。

对应文件：`chapters/011-prefill-decode-kv-cache.md`

## 8. Qwen3.5 Transformer 层

问题：一层模型内部到底做了什么？

讲解内容：

- residual 和 RMSNorm。
- `linear_attention` 与 `full_attention` 分流。
- `full_attention` 的 Q/K/V、RoPE、GQA、causal mask、softmax。
- `linear_attention` 的状态式注意力位置。
- MoE 替代传统 dense MLP。
- layer scale、输出 residual。

对应文件：`chapters/012-transformer-qwen35-layer.md`

## 9. FusedMoE

问题：A3B、专家、router、FusedMoE 到底是什么？

讲解内容：

- router logits 的含义。
- top-k 选专家。
- `topk_weights/topk_ids` 的形状和作用。
- token dispatch：把 token 按专家重排。
- expert MLP：`gate_proj/up_proj/down_proj` 与 SwiGLU。
- token combine：按路由权重合并专家输出。
- 为什么叫 fused：减少 Python 循环、减少小算子、减少搬运。
- 本测试 `enable_expert_parallel=False`，所以走默认 AllGather 兼容路径，不走 EP all-to-all。

对应文件：`chapters/013-fusedmoe-router-experts.md`

## 10. W8A8 Dynamic 量化

问题：W8A8 不是一句“省显存”就完了，它具体怎么算？

讲解内容：

- 权重量化：`float -> int8 + scale`。
- 激活动态量化：每次运行根据当前 token 的激活值求 scale。
- `npu_dynamic_quant` 产出 `quantized_x` 和 `pertoken_scale`。
- `npu_quant_matmul` 用 int8 输入和 int8 权重做矩阵乘，再按 scale 还原到目标 dtype。
- MoE 中的 grouped quant matmul 如何套用同一思想。
- 本测试中 MoE expert 权重是 W8A8_DYNAMIC，attention/lm_head 代表性条目仍是 FLOAT。

对应文件：`chapters/014-w8a8-dynamic-quantization.md`

## 11. Logits 和 Greedy Sampling

问题：模型怎么从 248320 个候选 token 里选一个？

讲解内容：

- LM head：`logits = h @ W_vocab^T`。
- logits 不是概率，softmax 后才是概率。
- greedy 不需要随机采样，直接 `argmax(logits)`。
- `temperature=0.0` 在 vLLM sampler 中走 greedy 分支。
- 本次 5 个生成 token：`[498, 7525, 3855, 1089, 321]`。

对应文件：`chapters/015-logits-sampling-output.md`

## 12. 源码地图和公式速查

目标：以后你想回头查某个概念，不用重读所有章节。

- `reference/source-map.md`：源码入口表。
- `reference/formulas.md`：公式速查表。
- `reference/glossary.md`：术语表。
- `reference/tensor-shapes.md`：关键 tensor 形状速查表。
- `reference/runtime-checklist.md`：运行和源码排查检查表。
