# Qwen3.5-35B-A3B W8A8 资料索引

## 知识来源

- [目标 pytest：`tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py`](../tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py)
  用途：确认模型名、TP 大小、量化模式、prompt 和断言。
- [E2E 工具：`tests/e2e/conftest.py`](../tests/e2e/conftest.py)
  用途：理解 `wait_until_npu_memory_free`、`VllmRunner`、`generate_greedy` 和输出整理。
- [Ascend model runner：`vllm_ascend/worker/model_runner_v1.py`](../vllm_ascend/worker/model_runner_v1.py)
  用途：理解输入准备、forward、logits 计算和 token 采样。
- [Ascend worker：`vllm_ascend/worker/worker.py`](../vllm_ascend/worker/worker.py)
  用途：理解 NPU 绑定、内存 profiling、KV cache 可用空间、warmup 和图捕获。
- [Qwen3.5 patch：`vllm_ascend/patch/worker/patch_qwen3_5.py`](../vllm_ascend/patch/worker/patch_qwen3_5.py)
  用途：理解 Qwen3.5 在 Ascend 上的 decoder layer 和 MTP patch。
- [Worker patch 入口：`vllm_ascend/patch/worker/__init__.py`](../vllm_ascend/patch/worker/__init__.py)
  用途：确认 Qwen3.5 patch 如何在 worker 侧被导入并生效。
- [MoE 通信选择：`vllm_ascend/ascend_forward_context.py`](../vllm_ascend/ascend_forward_context.py)
  用途：理解 MoE 通信策略、A3、W8A8 和 EP 相关判断。
- [上游 vLLM 离线接口：`/Users/lonng/Mrz20/vllm/vllm/entrypoints/llm.py`](/Users/lonng/Mrz20/vllm/vllm/entrypoints/llm.py)
  用途：理解 `LLM.generate` 如何进入 vLLM。
- [上游 vLLM 离线执行工具：`/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py`](/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py)
  用途：理解 request 创建、engine step 循环和最终输出收集。
- [上游 vLLM v1 engine：`/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py`](/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py)
  用途：理解 `add_request()`、`step()` 和 output processor。
- [上游 vLLM EngineCore：`/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py`](/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py)
  用途：理解 scheduler、model executor、sampling 和 `update_from_output()` 的关系。
- [上游 SchedulerOutput：`/Users/lonng/Mrz20/vllm/vllm/v1/core/sched/output.py`](/Users/lonng/Mrz20/vllm/vllm/v1/core/sched/output.py)
  用途：理解 `total_num_scheduled_tokens`、新 request、cached request 和 KV connector metadata。
- [上游 vLLM sampler：`/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py`](/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py)
  用途：解释为什么 `temperature=0` 会走 greedy `argmax`。
- [上游 Qwen3.5 模型：`/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py`](/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py)
  用途：理解 Qwen3.5 decoder layer、MoE 模型包装、LM head 和 `compute_logits`。
- [上游 Qwen3 Next 组件：`/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py`](/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py)
  用途：理解 `Qwen3NextSparseMoeBlock`、router、FusedMoE 和 attention 参数。
- [Ascend FusedMoE：`vllm_ascend/ops/fused_moe/fused_moe.py`](../vllm_ascend/ops/fused_moe/fused_moe.py)
  用途：理解 Ascend MoE runner、路由和 fused experts 入口。
- [Ascend 专家选择：`vllm_ascend/ops/fused_moe/experts_selector.py`](../vllm_ascend/ops/fused_moe/experts_selector.py)
  用途：理解 `topk_weights/topk_ids` 如何产生。
- [Ascend MoE 通信：`vllm_ascend/ops/fused_moe/moe_comm_method.py`](../vllm_ascend/ops/fused_moe/moe_comm_method.py)
  用途：理解 dispatch、MLP、combine 四阶段。
- [Ascend MoE MLP：`vllm_ascend/ops/fused_moe/moe_mlp.py`](../vllm_ascend/ops/fused_moe/moe_mlp.py)
  用途：理解 grouped matmul、SwiGLU 和量化 MLP。
- [W8A8 dynamic：`vllm_ascend/quantization/methods/w8a8_dynamic.py`](../vllm_ascend/quantization/methods/w8a8_dynamic.py)
  用途：理解 activation 动态量化、MoE W8A8 apply 和权重后处理。
- [W8A8 static：`vllm_ascend/quantization/methods/w8a8_static.py`](../vllm_ascend/quantization/methods/w8a8_static.py)
  用途：对比静态量化和动态量化。
- [ModelSlim 配置：`vllm_ascend/quantization/modelslim_config.py`](../vllm_ascend/quantization/modelslim_config.py)
  用途：理解 `quant_model_description.json` 如何映射到 Ascend quant method。
- [公式速查：`reference/formulas.md`](reference/formulas.md)
  用途：复习 RMSNorm、attention、MoE、W8A8、logits 和 sampling 公式。
- [源码地图：`reference/source-map.md`](reference/source-map.md)
  用途：快速定位每个概念对应的代码。
- [源码逐跳讲解：`SOURCE_WALKTHROUGH.md`](SOURCE_WALKTHROUGH.md)
  用途：按真实函数调用顺序阅读本测试从 pytest 到输出的代码流程。

## 实战经验来源

- 以仓库里的 e2e pytest 用例作为主要练习闭环，因为它们直接体现 Ascend 硬件上的真实运行假设。
- 需要理解跨文件调用链或做大范围改动时，先用 CodeGraph 建图和查询，避免在多个 model runner 路径里迷路。

## 已验证的远端模型文件

- `/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp/config.json`
  用途：确认真实模型结构参数。
- `/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp/quant_model_description.json`
  用途：确认哪些权重是 `W8A8_DYNAMIC`，哪些仍是 `FLOAT`。
- `/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp/quant_model_weights-00001-of-00010.safetensors`
  用途：确认本测试加载的是 10 个量化权重分片。
