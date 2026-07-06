# 源码地图

这份表把概念、函数和文件位置连起来。阅读章节时，如果你想看真实代码，从这里跳。

## 测试入口

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| prompt | `tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py:24` | `EXAMPLE_PROMPTS = ["Hello, my name is"]` |
| vLLM Runner 参数 | `tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py:32` | 模型名、TP=2、EP=false、quantization=ascend |
| 非空断言 | `tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py:44` | 只检查输出文本非空 |

## pytest 包装

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| 创建 `LLM` | `tests/e2e/conftest.py:968` | `VllmRunner.__init__` 把测试参数传给 `vllm.LLM` |
| 构造文本输入 | `tests/e2e/conftest.py:1003` | `get_inputs` 把字符串包装成 `TextPrompt` |
| 调用推理 | `tests/e2e/conftest.py:1035` | `generate` 调 `self.model.generate` |
| greedy 参数 | `tests/e2e/conftest.py:1083` | `temperature=0.0, max_tokens=5` |
| 拼最终结果 | `tests/e2e/conftest.py:987` | prompt token 和 output token 拼在一起 |

## vLLM 离线引擎

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| 加请求再跑引擎 | `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:326` | `_run_completion` |
| 渲染并添加请求 | `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:523` | `_render_and_add_requests` |
| 设置 final-only 输出 | `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:552` | `_add_request` |
| 引擎循环 | `/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py:573` | `_run_engine` 反复调用 `llm_engine.step()` |
| 离线本地路径替换 | `/Users/lonng/Mrz20/vllm/vllm/engine/arg_utils.py:760` | `HF_HUB_OFFLINE` 时把模型 ID 替换成本地路径 |
| 添加 request | `/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py:218` | `LLMEngine.add_request` |
| v1 engine step | `/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py:296` | 获取 EngineCore 输出并交给 output processor |
| EngineCore 调度执行 | `/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:479` | `scheduler.schedule()` |
| EngineCore 调模型 | `/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py:490` | `model_executor.execute_model(scheduler_output)` |
| SchedulerOutput | `/Users/lonng/Mrz20/vllm/vllm/v1/core/sched/output.py:181` | 定义 `total_num_scheduled_tokens` 等 worker 任务单字段 |

## 运行环境和 worker

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| Ascend quantization 名称 | `vllm_ascend/utils.py:48` | `ASCEND_QUANTIZATION_METHOD = "ascend"` |
| NPU 可见数检查 | `vllm_ascend/worker/worker.py:440` | worker 读取 `torch.npu.device_count()` |
| local rank 到 device | `vllm_ascend/worker/worker.py:445` | 逻辑 rank 映射到可见 device id |
| 设置 NPU device | `vllm_ascend/worker/worker.py:450` | `torch.npu.set_device(device)` |
| requested memory | `vllm_ascend/worker/worker.py:472` | `total_memory * gpu_memory_utilization` |
| dummy profile run | `vllm_ascend/worker/worker.py:558` | 用 dummy forward 估算非 KV 内存 |
| KV cache 可用空间 | `vllm_ascend/worker/worker.py:613` | requested memory 减去权重、激活、图内存等 |
| warmup sizes | `vllm_ascend/worker/worker.py:774` | 编译和 warmup 模型 |
| 图捕获入口 | `vllm_ascend/worker/worker.py:800` | `model_runner.capture_model()` |

## Ascend Model Runner

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| worker 执行入口 | `vllm_ascend/worker/model_runner_v1.py:1970` | `execute_model` |
| prepare inputs | `vllm_ascend/worker/model_runner_v1.py:789` | 计算 positions、input_ids、query_start_loc |
| positions 计算 | `vllm_ascend/worker/model_runner_v1.py:834` | `position = num_computed_tokens + query_pos` |
| input ids gather | `vllm_ascend/worker/model_runner_v1.py:918` | 从 request token 缓冲中取本 step token |
| query_start_loc | `vllm_ascend/worker/model_runner_v1.py:979` | 记录每个 request 的 token 边界 |
| M-RoPE positions | `vllm_ascend/worker/model_runner_v1.py:1017` | 计算多维 RoPE position |
| preprocess 包装 | `vllm_ascend/worker/model_runner_v1.py:1396` | Ascend runner 包装上游 preprocess |
| 本步 token 数 | `vllm_ascend/worker/model_runner_v1.py:2042` | `scheduler_output.total_num_scheduled_tokens` |
| 准备输入 | `vllm_ascend/worker/model_runner_v1.py:2111` | `_prepare_inputs` |
| padding 和图执行决策 | `vllm_ascend/worker/model_runner_v1.py:2133` | `_determine_batch_execution_and_padding` |
| attention metadata | `vllm_ascend/worker/model_runner_v1.py:2249` | `_build_attention_metadata` |
| 预处理出 tensor | `vllm_ascend/worker/model_runner_v1.py:2272` | `_preprocess` |
| forward context | `vllm_ascend/worker/model_runner_v1.py:2312` | `set_ascend_forward_context` |
| 调真实模型 | `vllm_ascend/worker/model_runner_v1.py:2338` | `_model_forward` |
| 算 logits | `vllm_ascend/worker/model_runner_v1.py:2375` | 选 sample hidden states 并 `compute_logits` |
| 采样入口 | `vllm_ascend/worker/model_runner_v1.py:2421` | `sample_tokens` |
| 调 sampler | `vllm_ascend/worker/model_runner_v1.py:2634` | `_sample` |
| 模型调用包装 | `vllm_ascend/worker/model_runner_v1.py:2852` | `_model_forward` 把输入传给 `self.model` |
| 图模式和 padding | `vllm_ascend/worker/model_runner_v1.py:2956` | `_determine_batch_execution_and_padding` |
| 构造 metadata | `vllm_ascend/worker/model_runner_v1.py:3046` | `_build_attention_metadata` |
| common attention metadata | `vllm_ascend/worker/model_runner_v1.py:3174` | seq_lens、block table、slot mapping、causal 等 |
| per-group metadata | `vllm_ascend/worker/model_runner_v1.py:3284` | 按 KV cache group 生成各层 metadata |

## Qwen3.5 模型

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| Decoder layer 初始化 | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:112` | 决定 layer 是 full attention 还是 linear attention |
| MoE 替代 MLP | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:147` | `qwen3_5_moe_text` 使用 `Qwen3NextSparseMoeBlock` |
| embedding 和 layers | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:212` | `Qwen3_5Model` 创建 embedding、40 层、final norm |
| LM head | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:311` | 词表投影层 |
| compute logits | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:353` | hidden state -> logits |
| MoE 参数收集 | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:573` | `set_moe_parameters` |

## Qwen3.5 Ascend Patch

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| worker patch 导入 | `vllm_ascend/patch/worker/__init__.py:49` | 非 310P 导入 `patch_qwen3_5` |
| full attention forward patch | `vllm_ascend/patch/worker/patch_qwen3_5.py:41` | QKV split、Q/K RMSNorm、RoPE、attention、gate、o_proj |
| decoder layer forward patch | `vllm_ascend/patch/worker/patch_qwen3_5.py:90` | RMSNorm -> attention -> RMSNorm -> MoE |
| MTP forward patch | `vllm_ascend/patch/worker/patch_qwen3_5.py:153` | MTP drafter 的 forward |
| monkeypatch 生效 | `vllm_ascend/patch/worker/patch_qwen3_5.py:195` | 替换上游 forward |

## Qwen3 Next 组件

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| Sparse MoE block | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:101` | gate、shared expert、FusedMoE |
| router logits | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:210` | `router_logits, _ = self.gate(hidden_states)` |
| Q/K/V 参数 | `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:225` | `Qwen3NextAttention` |

## FusedMoE

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| 无量化 MoE apply | `vllm_ascend/ops/fused_moe/fused_moe.py:141` | router -> fused experts |
| Ascend MoE runner 初始化 | `vllm_ascend/ops/fused_moe/fused_moe.py:287` | 绑定 Ascend quant method 和通信方法 |
| Ascend MoE forward | `vllm_ascend/ops/fused_moe/fused_moe.py:545` | prepare -> quant apply -> finalize |
| 选择专家 | `vllm_ascend/ops/fused_moe/experts_selector.py:30` | 返回 `topk_weights/topk_ids` |
| NPU top-k gate | `vllm_ascend/ops/fused_moe/experts_selector.py:235` | 使用 Ascend 融合 gating 算子 |
| native top-k gate | `vllm_ascend/ops/fused_moe/experts_selector.py:312` | softmax/sigmoid/sqrtsoftplus + topk |
| dispatch/MLP/combine | `vllm_ascend/ops/fused_moe/moe_comm_method.py:122` | `fused_experts` 主流程 |
| AllGather dispatcher | `vllm_ascend/ops/fused_moe/token_dispatcher.py:342` | EP=false 默认可用路径 |
| grouped MLP | `vllm_ascend/ops/fused_moe/moe_mlp.py:424` | `unified_apply_mlp` |

## W8A8

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| 量化自动检测 | `vllm_ascend/quantization/utils.py:83` | 根据 `quant_model_description.json` 判断 Ascend ModelSlim |
| 自动应用量化 | `vllm_ascend/quantization/utils.py:141` | 没有显式量化时可自动补充 quant config |
| AscendModelSlimConfig | `vllm_ascend/quantization/modelslim_config.py:456` | `ascend` 量化配置类 |
| 读取量化描述 | `vllm_ascend/quantization/modelslim_config.py:749` | `maybe_update_config` 加载 `quant_model_description.json` |
| layer quant method | `vllm_ascend/quantization/modelslim_config.py:793` | 根据 layer 类型和 prefix 选择 quant method |
| dynamic linear | `vllm_ascend/quantization/methods/w8a8_dynamic.py:48` | activation 动态量化 + int8 matmul |
| dynamic MoE | `vllm_ascend/quantization/methods/w8a8_dynamic.py:155` | W8A8_DYNAMIC FusedMoE |
| dynamic MoE apply | `vllm_ascend/quantization/methods/w8a8_dynamic.py:214` | select experts -> fused experts |
| dynamic MoE 权重整理 | `vllm_ascend/quantization/methods/w8a8_dynamic.py:353` | transpose、NZ、scale 展平 |
| static linear | `vllm_ascend/quantization/methods/w8a8_static.py:33` | 静态 input scale 量化路径 |
| quant adapter | `vllm_ascend/quantization/method_adapters.py:201` | vLLM FusedMoE method 包装 |
| ModelSlim mapping | `vllm_ascend/quantization/modelslim_config.py:66` | `quant_model_description.json` 解析入口 |

## Sampling

| 概念 | 文件和行 | 说明 |
|---|---:|---|
| Ascend sampler | `vllm_ascend/sample/sampler.py:45` | Ascend 采样器 |
| greedy sample | `vllm_ascend/sample/sampler.py:105` | Ascend 中 greedy 可直接 argmax |
| 上游 sampler | `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:243` | vLLM sampling 主逻辑 |
| temperature | `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:228` | 温度处理 |
| greedy argmax | `/Users/lonng/Mrz20/vllm/vllm/v1/sample/sampler.py:240` | `logits.argmax(dim=-1)` |
