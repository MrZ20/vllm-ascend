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
    - review 后修正：`AscendMoERunner.forward_impl` 的 UT 改为验证新版 `self.routed_experts` 委托路径，不再保留旧 `layer` 参数假设。
    - review 后修正：新版 runner 传入的 `shared_experts_input` 继续传给 Ascend shared expert 路径，避免 routed input transform 场景下 shared expert 误用 routed hidden states。
    - review 后修正：重建 RoutedExperts 权重时补齐上游新增的 `unpadded_hidden_size` 参数，保持和 target vLLM `RoutedExperts.create_weights` 参数集一致。
- `vllm_ascend/quantization/*.py`
    - MoE layer 判断改为兼容旧 `FusedMoE` class 和新 `RoutedExperts`。
- `vllm_ascend/worker/model_runner_v1.py`
    - target vLLM 下把 routed-experts capturer 绑定到 `MoERunner.routed_experts`。
- `_310p` 相关文件
    - 保持 import/量化识别兼容 target vLLM 的拆分，避免 CPU UT 收集阶段失败。
    - review 后修正：旧 vLLM 仍使用 `AscendFusedMoE310(FusedMoE)` subclass；target vLLM 下 `AscendFusedMoE310(...)` 不再抛错，而是委托到已适配的 factory path。这样 310P 不会把新版 package/layer-level `FusedMoE` patch 成一个不可构造的类。

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

## Review 后推翻/删除的内容

- 删除 `.github/workflows/scripts/select_tests.py` 中未被 workflow 使用的 `--skip-default-cpu-ut` 参数及其 UT。CPU UT 在 main2main 中继续默认全跑，不引入额外选择器语义。
- 推翻 `tests/ut/ops/test_fused_moe.py::test_forward_impl_delegates_to_layer` 的旧接口假设，改为覆盖 target vLLM 的 `MoERunner -> routed_experts` 委托。
- 推翻 310P target vLLM 下只“保 import 但构造即 RuntimeError”的写法；target factory 场景必须至少可构造，并与主线 factory patch 保持同一扩展方式。
- 第二轮 CI 后推翻 import-probing 式版本兼容：
    - `vllm_ascend/ops/fused_moe/fused_moe.py`
    - `vllm_ascend/_310p/fused_moe/fused_moe.py`
    - `vllm_ascend/quantization/compressed_tensors_config.py`
    - `vllm_ascend/quantization/modelslim_config.py`
    - `vllm_ascend/quantization/fp8_config.py`
    - `vllm_ascend/_310p/quantization/modelslim_config.py`
    - 对应 UT 中的 MoE spec 选择
    这些位置统一改成 `vllm_version_is("0.22.1")` 的 if-else。原因是上游 PR #41184 带来的语义变化不是单纯模块是否存在，而是 `FusedMoE` 从 class 变为 factory、权重 owner 迁移到 `RoutedExperts`；显式版本分支比 `try/except ImportError` 更符合 main/v0.22.1 双版本维护规范。
    - 分支风格约定：旧/新两套并列实现使用 `if vllm_version_is("0.22.1") ... else ...`；只在 target main 新增的导入、字段或协议补齐使用 `if not vllm_version_is("0.22.1")`。
    - quant config 中保留 v0.22.1 分支的原始 `isinstance(layer, FusedMoE)` 判断，只在 target-main 分支导入并识别 `RoutedExperts`，避免把旧分支也抽象重写，同时规避 `RoutedExperts = None` 后再 import 造成的 mypy `no-redef` / `isinstance(..., None)` 风险。

## 第二轮 CI 失败与修复

CI run: <https://github.com/vllm-project/vllm-ascend/actions/runs/27602442936?pr=10459>

### 1. CPU UT: `AscendFusedMoE.__new__` 触发 target factory

失败：

- `tests/ut/ops/test_fused_moe.py` 多个用例失败。
- 报错：`TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'`。

原因：

- 上游 vLLM PR #41184 后，target main 下 `AscendFusedMoE.__new__(AscendFusedMoE)` 会委托到 `FusedMoE` factory。
- 旧 UT 用 `AscendFusedMoE.__new__` 构造半初始化对象来测试 helper 方法，这在 target main 下不再是“空壳构造”，而是进入真实 factory。

修复：

- 新增测试 helper `_new_uninitialized_ascend_fused_moe()`，用 `nn.Module.__new__(AscendFusedMoE)` 显式绕过 target factory。
- 所有需要半初始化对象的 UT 改用该 helper。
- 注释说明该写法受 vLLM PR #41184 影响，v0.22.1 下等价于旧测试意图。

### 2. A3 two-card: MoE quant method 缺 `is_monolithic`

失败：

- `tests/e2e/pull_request/two_card/test_prefix_caching.py::test_models_prefix_cache_tp2[50-deepseek-ai/DeepSeek-V2-Lite-Chat]`
- 根因：`AttributeError: 'UnquantizedFusedMoEMethod' object has no attribute 'is_monolithic'`。

原因：

- 上游 vLLM PR #41184 的 `MoERunner._apply_quant_method` 会读取 `self.routed_experts.quant_method.is_monolithic` 来决定 modular/monolithic 路径。
- Ascend 适配后的 quant method 需要暴露该 target-main 属性，同时保持 v0.22.1 旧路径不受影响。

修复：

- `AscendUnquantizedFusedMoEMethod` / `AscendUnquantizedFusedMoEMethod310` 保留 `is_monolithic = False`。
- `AscendFusedMoEMethod` adapter 新增 `is_monolithic = False`，覆盖量化 MoE 场景。
- `AscendRoutedExperts.__init__` 在替换为 Ascend quant method 后，额外补齐缺失的 `is_monolithic = False`。原因是 PR #41184 后 `RoutedExperts.__init__` 会先安装上游 quant method，部分 quant config 路径可能返回仍未声明该 target-main 协议的 method；在真正交给 `MoERunner` 前统一补齐，避免 prefix-cache DeepSeek 路径再次读到旧对象。
- 注释说明该属性来自 vLLM PR #41184 的新 `MoERunner` 判断。

### 3. A2 one-card: `AscendFusedMoEMethod.apply` 不接 `topk_weights`

失败：

- `tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py` 中 DeepSeek-V2-Lite-W8A8 相关用例。
- 根因：`TypeError: AscendFusedMoEMethod.apply() got an unexpected keyword argument 'topk_weights'`。

原因：

- 上游 vLLM PR #41184 的 `RoutedExperts.forward_modular()` 会向 `quant_method.apply(...)` 传入预计算路由结果 `topk_weights` / `topk_ids`。
- Ascend quantized MoE adapter 仍是旧签名，不能接收这两个新 keyword。

修复：

- `AscendFusedMoEMethod.apply` 增加可选 `topk_weights` / `topk_ids` 参数。
- adapter 层接收并显式丢弃这两个参数，因为现有 Ascend quantized scheme 仍在自身 `apply` 内完成 `select_experts`；直接透传会打破 v0.22.1 风格的 scheme 签名。
- `AscendUnquantizedFusedMoEMethod.apply` 也增加这两个参数；当 target main 已传入路由结果时复用它们，否则保持旧路径自行 `select_experts`。

### 4. A2 one-card spec decode: `build_attn_metadata` 不接 `causal`

失败：

- `tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py` 中 EAGLE spec decode 相关用例。
- 根因：`TypeError: build_attn_metadata() got an unexpected keyword argument 'causal'`。

原因：

- target main 的 spec decode `speculator.py` 调用 `build_attn_metadata(..., causal=...)`。
- Ascend 覆盖的 `vllm_ascend/worker/v2/attn_utils.py::build_attn_metadata` 仍是旧签名。

修复：

- `build_attn_metadata` 增加默认参数 `causal: bool = True`。
- 当前 Ascend metadata builder 不消费该字段，因此显式 `del causal`，仅用于兼容 target main 调用；v0.22.1 调用方继续不传该参数。

### 5. A3 four-card: `AscendRoutedExperts` 缺旧 Ascend routing 字段

失败：

- `tests/e2e/pull_request/multi_card/test_ep.py::test_ep[deepseek-ai/DeepSeek-V2-Lite-Chat-4-2-True-3-1]`
- 根因：`AttributeError: 'AscendRoutedExperts' object has no attribute 'e_score_correction_bias'`。

原因：

- 上游 vLLM PR #41184 将 target main 的 MoE 执行 owner 从 legacy `FusedMoE` 拆到 `RoutedExperts`。
- Ascend target-main 适配复用了旧 `AscendFusedMoE.forward_impl`，该实现会读取 `renormalize`、`use_grouped_topk`、`topk_group`、`e_score_correction_bias`、`apply_router_weight_on_input` 等旧 `FusedMoE` public 字段。
- 新上游 `RoutedExperts.__init__` 不保证继续以同名 public attribute 暴露这些字段，因此 EP DeepSeek 路径进入 Ascend forward 后直接访问失败。

修复：

- `AscendRoutedExperts.__init__` 调用上游 `RoutedExperts.__init__` 后，显式保存 Ascend 旧 forward 所需的 routing 字段。
- 注释说明这是受 vLLM PR #41184 的 owner 拆分影响：target main 下 `AscendRoutedExperts` 是真实执行对象，必须携带旧 Ascend forward 依赖的状态；v0.22.1 仍走 legacy `AscendFusedMoE` class，不受影响。

### 6. mypy: 条件导入导致 `no-redef` 和变量基类错误

失败：

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- 根因：`fused_moe_make_expert_params_mapping` / `RoutedExperts` 在 v0.22.1 分支先赋值，又在 target-main 分支用同名 import，触发 `no-redef`。
- 继续用 `class AscendRoutedExperts(RoutedExperts)` 会让 mypy 把条件分支里的 `RoutedExperts` 当普通变量，不接受它作为基类。

原因：

- 上游 vLLM PR #41184 只在 target main 引入 `RoutedExperts` owner 和 `fused_moe_make_expert_params_mapping` helper。
- v0.22.1 分支不应该 import target-main 新对象，但 mypy 需要在静态分析时看到真实基类类型。

修复：

- 使用 target-main 专用别名 `_TargetRoutedExperts` / `_target_fused_moe_make_expert_params_mapping`，避免覆盖旧分支名字。
- `TYPE_CHECKING` 分支仅供 mypy 解析真实 target-main 类型；运行时 v0.22.1 仍使用 `torch.nn.Module` fallback，且不会走 target factory 注入路径。
- `AscendRoutedExperts` 改为继承 `_TargetRoutedExperts`，内部显式调用也同步使用该别名，消除 `valid-type` / `Invalid base class`。

### 7. 分支代码与 UT 双版本规范化

检查范围：

- 分支相对 `upstream/main` 的 Python 修改文件。
- 重点检查 target-main 专属逻辑是否裸露、UT 是否只覆盖一个版本。

修复：

- `model_runner_v1.py::_bind_routed_experts_capturer`：
    - v0.22.1 分支显式绑定 legacy `FusedMoE`。
    - target-main 分支显式绑定 `MoERunner.routed_experts`。
    - 去掉 `isinstance(FusedMoE, type)` 和 `hasattr(..., "routed_experts")` 探测。
- `utils.py::register_ascend_customop`：
    - `patch_fused_moe_factory(...)` 只在 `not vllm_version_is("0.22.1")` 下执行。
    - v0.22.1 保留原 CustomOp OOT 注册路径。
- `patch_tool_choice_none_content.py` 及对应 UT：
    - v0.22.1 分支 patch / 测试 `_parse_tool_calls`。
    - target-main 分支 patch / 测试 `_extract_tool_calls`。
    - 不再用 `hasattr` 动态选择 API。

原则：

- 并列旧/新行为必须用 `if vllm_version_is("0.22.1") ... else ...`。
- target-main 新增导入、字段、factory patch、协议补齐才使用 `if not vllm_version_is("0.22.1")`。
- UT 要么在同一个用例内部按版本选择断言/调用，要么拆成版本专属用例并用 `vllm_version_is("0.22.1")` 控制跳过，不能只保证单版本通过。

## 本地验证状态

本地已做：

- `git diff --check` 通过。
- `python3 -m py_compile` 覆盖本轮修改文件，通过。
- `source .venv/bin/activate && bash format.sh` 通过。
- 本地尝试 `python3 -m mypy --ignore-missing-imports ...`，但当前 `.venv` 缺完整 target vLLM/torch-npu 类型环境，仍会展开到大量无关导入文件；该结果不能等价 CI mypy。已根据输出额外清理 quant config / UT 中的 `RoutedExperts = None` 静态类型风险，同时保持 v0.22.1 分支尽量原样。

本地未做：

- pytest / e2e 未跑。本地环境没有完整安装 vLLM/torch/NPU 运行环境，行为验证依赖 CI。
