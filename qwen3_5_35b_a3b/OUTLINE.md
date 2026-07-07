# 教学大纲

这是总览型学习路线，适合快速扫全局。深度路线见 `DEEP_OUTLINE.md`；源码逐跳路线见 `SOURCE_WALKTHROUGH.md`；复习材料见 `reference/`。

## 深度章节入口

- `chapters/005-runtime-environment-and-processes.md`：环境变量、spawn、NPU 可见设备和 worker 进程。
- `chapters/006-model-loading-quant-config-and-patches.md`：模型缓存、`config.json`、`quant_model_description.json` 和 Ascend patch。
- `chapters/007-request-engine-scheduler-lifecycle.md`：`TextPrompt`、request、engine、scheduler、`SchedulerOutput`。
- `chapters/008-attention-metadata-position-and-backend.md`：positions、M-RoPE、`query_start_loc`、`block_table`、`slot_mapping`。
- `chapters/009-parallelism-memory-and-graph-capture.md`：TP、EP=false、内存 profiling、KV cache、图捕获和 padding。
- `chapters/010-end-to-end-data-derivation.md`：从输入 prompt 到输出文本的完整数据推导。
- `chapters/011-prefill-decode-kv-cache.md`：prefill、decode、KV cache 和 padding。
- `chapters/012-transformer-qwen35-layer.md`：Qwen3.5 decoder layer、RMSNorm、full attention、linear attention、MoE。
- `chapters/013-fusedmoe-router-experts.md`：router、top-k、dispatch、grouped matmul、combine。
- `chapters/014-w8a8-dynamic-quantization.md`：W8A8_DYNAMIC 量化公式和源码路径。
- `chapters/015-logits-sampling-output.md`：logits、greedy sampling、token 解码和最终输出。
- `reference/formulas.md`：公式速查。
- `reference/glossary.md`：术语表。
- `reference/tensor-shapes.md`：关键 tensor 形状速查。
- `reference/runtime-checklist.md`：运行和源码排查检查表。
- `reference/source-map.md`：源码地图。
- `SOURCE_WALKTHROUGH.md`：源码逐跳讲解入口。
- `source-walkthrough/000-branch-and-reading-map.md`：两个教学分支和跨仓库源码地图。
- `source-walkthrough/001-test-to-offline-engine.md`：pytest、`VllmRunner`、`LLM.generate()`、offline engine。
- `source-walkthrough/002-engine-core-to-npu-runner.md`：`LLMEngine`、`EngineCore`、scheduler 到 Ascend runner。
- `source-walkthrough/003-npu-runner-forward-state.md`：`NPUModelRunner.execute_model()` 的状态和数据变化。
- `source-walkthrough/004-qwen35-forward-layer.md`：Qwen3.5 patched forward 和 decoder layer。
- `source-walkthrough/005-fusedmoe-w8a8-source-flow.md`：MoE、router、FusedMoE 和 W8A8 源码。
- `source-walkthrough/006-sampling-and-output.md`：logits、greedy sampling 和 `RequestOutput`。

编号说明：`000-004` 是基础章节；`005-009` 是运行时桥梁层；`010-015` 是核心数学、源码和公式深挖。

## 0. 一屏心智模型

目标：先建立整体图，再进入细节。

- 你输入一句 prompt。
- tokenizer 把文本切成 token ID。
- vLLM 创建一个 request，并附带生成参数。
- scheduler 决定当前 step 要跑哪些 token。
- Ascend model runner 准备张量和元数据。
- Qwen3.5 在 NPU 上执行 forward，得到 hidden states。
- LM head 把 hidden states 转成词表上的 logits。
- greedy sampling 选择分数最高的下一个 token。
- vLLM 把新 token 接到已有输出后面，重复直到生成 5 个 token 或遇到停止条件。
- tokenizer 把 token ID 解码回文本。

## 1. 真实工作流

目标：让运行流程可复现。

- SSH 到 A3 机器。
- 进入目标 Docker 容器。
- 进入 `/vllm-workspace/vllm-ascend`。
- 设置离线加载和运行时环境变量。
- 用 `npu-smi info` 查看 NPU 使用情况。
- 设置 `ASCEND_RT_VISIBLE_DEVICES`。
- 运行目标 pytest。
- 记录硬件、可见设备、命令、耗时和通过/失败结果。

## 2. 这个测试到底测什么

目标：拆开很短的 pytest 文件。

- `EXAMPLE_PROMPTS = ["Hello, my name is"]`
- `HCCL_BUFFSIZE=1024`
- 模型：`Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp`
- `max_model_len=4096`
- `tensor_parallel_size=2`
- `enable_expert_parallel=False`
- `quantization="ascend"`
- `distributed_executor_backend="mp"`
- `cudagraph_capture_sizes=[1, 2, 4, 8]`
- 断言：输出文本非空。

## 3. pytest 包装和 `VllmRunner`

目标：解释 pytest 只是 vLLM 离线推理的一层包装。

- `wait_until_npu_memory_free()` 在子进程里检查 NPU 空闲内存。
- `VllmRunner.__init__()` 构造 `vllm.LLM`。
- `generate_greedy()` 创建 `SamplingParams(temperature=0.0, max_tokens=5)`。
- `generate()` 构造 `TextPrompt` 并调用 `LLM.generate()`。
- `_finalize_generate_outputs()` 整理 token ID 和文本。

## 4. prompt 到 request

目标：解释推理前半段。

- 原始字符串。
- tokenization。
- request ID。
- sampling 参数。
- engine 队列。
- scheduler step。
- prefill 和 decode。

## 5. Ascend model runner

目标：解释 worker 侧执行路径。

- `execute_model()` 更新 request 状态并准备模型输入。
- 构造 attention metadata 和 positions。
- `set_ascend_forward_context()` 记录 Ascend 算子需要的运行上下文。
- `_model_forward()` 调用真实模型。
- `compute_logits()` 把选中的 hidden states 转成词表分数。
- `sample_tokens()` 调用 `_sample()`。

## 6. greedy 输出如何选择

目标：解释为什么本测试输出是确定性的。

- `temperature=0.0` 表示 greedy。
- vLLM sampler 执行 `argmax(logits)`。
- 选出的 token 被接到 request 后面。
- 循环最多重复 5 次。

## 7. 从零理解 Qwen3.5 模型名

目标：解释模型名里的词。

- `35B`：总参数规模。
- `A3B`：每个 token 激活的参数量小于总参数量，通常来自 MoE 路由。
- `MoE`：router 在 FFN 阶段选择专家。
- `MTP`：multi-token prediction 支持，本仓库有 Qwen3.5 相关 patch。
- 结合真实 `config.json` 解释层数、专家数、hidden size、M-RoPE 和 attention 类型。

## 8. 从零理解 W8A8 量化

目标：解释大模型为什么能更省内存地运行。

- `W8A8`：权重和激活在支持的 kernel 中使用 8-bit 表示。
- `quantization="ascend"`：要求 vLLM Ascend 使用 Ascend 量化路径。
- 量化减少显存和带宽压力，但依赖硬件适配 kernel。

## 9. Tensor parallelism 和 NPU 可见设备

目标：连接 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 和 `tensor_parallel_size=2`。

- 可见设备定义当前进程能看到哪些 NPU。
- TP=2 表示大矩阵计算由两个 rank 协同完成。
- 本测试关闭 EP，所以没有按专家并行切分专家。
- `distributed_executor_backend="mp"` 表示使用 multiprocessing worker。

## 10. 如何读日志

目标：知道运行时重点看什么。

- 模型缓存和权重分片加载。
- Ascend 后端初始化。
- worker spawn 和分布式连接。
- 内存分析和 KV cache 分配。
- 图捕获大小 1、2、4、8。
- prompt 渲染和处理进度。
- 最终 pytest 断言。

## 11. 源码索引

目标：把概念映射到代码。

- 测试入口文件。
- `tests/e2e/conftest.py`
- 上游 `vllm/entrypoints/llm.py`
- 上游 `vllm/entrypoints/offline_utils.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/ascend_forward_context.py`

## 12. 验证总结

目标：记录真实证据。

- 命令。
- 环境变量。
- 选择的设备。
- 退出状态。
- 输出文本。
- 警告或失败。
- 如果失败，解释失败发生在哪一层。
