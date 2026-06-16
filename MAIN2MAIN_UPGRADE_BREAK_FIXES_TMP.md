# main2main 临时说明：vLLM 967c5c3 -> e0871ad

> 临时 review 文件。当前 PR 验证通过后可删除。

## 背景

本轮升级 vLLM commit：

- from: `967c5c3bc38891f4465d3f4e99917ed837bb3833`
- to: `e0871ad2259768add6dc43e2972bd364d0d13086`

当前 CI 先暴露了两个导入/收集阶段 blocker：

1. CPU UT 导入 `vllm_ascend.ops.fused_moe.fused_moe` 失败。
2. A3 e2e conftest 导入 `patch_tool_choice_none_content` 失败。

## 上游 PR -> break -> 本地修改

### 1. vLLM PR #41184: FusedMoE/MoERunner inversion refactor

上游变化：

- `FusedMoE` 从可继承的 class 改成 factory function。
- MoE 权重和 weight loading 迁移到 `RoutedExperts`。
- forward orchestration 迁移到 `MoERunner`。
- `UnquantizedFusedMoEMethod` 从 `fused_moe/layer.py` 移到 `fused_moe/unquantized_fused_moe_method.py`。
- `FusedMoE.make_expert_params_mapping` 对应能力变为 module-level helper。

导致 break：

- 旧代码 `class AscendFusedMoE(FusedMoE)` 在 target vLLM 下不再成立。
- 旧代码从 `vllm.model_executor.layers.fused_moe.layer` 导入 `UnquantizedFusedMoEMethod` 失败。
- 旧量化配置只识别 `isinstance(layer, FusedMoE)`，target vLLM 下真正持有权重的是 `RoutedExperts`，导致 Ascend MoE quant method 可能无法选中。
- routed-experts capture 旧逻辑只绑定到 `FusedMoE` layer，target vLLM 的 `static_forward_context` 注册对象变为 `MoERunner`。

本地修改：

- `vllm_ascend/ops/fused_moe/fused_moe.py`
    - 增加 `UnquantizedFusedMoEMethod` 新旧路径 fallback。
    - 增加 `AscendRoutedExperts`，把旧 `AscendFusedMoE` 里的 Ascend quant、EPLB、shared expert、NPU MoE comm 初始化迁移到 target vLLM 的 `RoutedExperts` 扩展点。
    - 调整 `AscendMoERunner`，按新 `_forward_impl(hidden_states, router_logits, shared_experts_input, input_ids)` 接口从 `self.routed_experts` 调用 Ascend 路径。
    - target vLLM 下 `AscendFusedMoE(...)` 通过上游 factory 注入 `runner_cls=AscendMoERunner` 和 `routed_experts_cls=AscendRoutedExperts`。
    - patch package-level 和 layer-level `FusedMoE` export，避免 OOT class 注册无法替换 plain function。
    - 为 `make_expert_params_mapping` 保留旧静态方法写法。
- `vllm_ascend/quantization/*.py`
    - MoE layer 判断改为兼容旧 `FusedMoE` class 和新 `RoutedExperts`。
- `vllm_ascend/worker/model_runner_v1.py`
    - target vLLM 下把 routed-experts capturer 绑定到 `MoERunner.routed_experts`。
- `_310p` 相关文件
    - 保持 import/量化识别兼容 target vLLM 的拆分，避免 CPU UT 收集阶段失败。

### 2. vLLM PR #45190 / #45171 / #45104: parser/Response API refactor

上游变化：

- parser 调用路径统一，`DelegatingParser._parse_tool_calls` 被移除/替换为 `_extract_tool_calls`。
- Chat Completions / Responses 的非流式与流式 harmony parser 路径重构。

导致 break：

- `vllm_ascend/patch/platform/patch_tool_choice_none_content.py` import 时直接访问 `DelegatingParser._parse_tool_calls`，target vLLM 下属性不存在，e2e conftest 导入失败。

本地修改：

- `patch_tool_choice_none_content.py`
    - 兼容旧 `_parse_tool_calls(request, content, enable_auto_tools)`。
    - 兼容新 `_extract_tool_calls(content, request, enable_auto_tools=False)`。
    - 保留原补丁目标：forced/named tool choice 且 `content is None` 时返回 `([], None)`，避免 reasoning 消耗全部文本后触发 assert。
- 对应 UT 根据当前 vLLM 存在的 parser API 选择调用路径。

## 下一轮 CI 选择

CPU UT：

- `tests/ut` 继续默认全跑。

新增/保留的 NPU e2e probe：

- parser/tool-choice：
    - `tests/e2e/pull_request/one_card/test_guided_decoding.py`
- MoE factory / RoutedExperts / shared expert：
    - `tests/e2e/pull_request/one_card/test_multistream_overlap_shared_expert.py`
    - `tests/e2e/pull_request/two_card/test_shared_expert_dp.py`
    - `tests/e2e/pull_request/two_card/test_deepseek_multistream_moe.py`
- routed experts capture：
    - `tests/e2e/pull_request/two_card/test_moe_routing_replay.py`
- EPLB / quantized MoE：
    - `tests/e2e/pull_request/two_card/test_qwen3_moe_eplb.py`
    - `tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py`
- base MoE / model-runner regression：
    - `tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py`
    - `tests/e2e/pull_request/four_card/test_deepseek_v4.py`
- 与前一轮无冲突、继续覆盖的 probe：
    - `tests/e2e/pull_request/one_card/spec_decode/test_dflash.py`
    - `tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py`
    - `tests/e2e/pull_request/two_card/test_prefix_caching.py`
    - `tests/e2e/pull_request/one_card/test_simple_cpu_offload.py`
    - `tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py`

这些测试通过 `.github/workflows/pr_test.yaml` 的 `MAIN2MAIN_TESTS` 和
`.github/workflows/scripts/main2main_probe_test_config.yaml` 临时选择。

## 本地验证状态

本地已做：

- `python3 -m py_compile` 覆盖本轮修改文件，通过。
- `git diff --check` 通过。

本地未做：

- pytest / e2e 未跑。本地环境没有完整安装 vLLM/torch/NPU 运行环境，行为验证依赖 CI。
