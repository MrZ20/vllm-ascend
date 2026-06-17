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

## 第三轮 CI 失败与修复

CI run: <https://github.com/vllm-project/vllm-ascend/actions/runs/27609361478?pr=10459>

本轮三处失败实际上只有两个根因。

### 1. CPU UT: 测试引用了已被重命名的 `RoutedExperts` 模块属性

失败：

- `tests/ut/ops/test_fused_moe.py` 收集阶段报错
  `AttributeError: module 'vllm_ascend.ops.fused_moe.fused_moe' has no attribute 'RoutedExperts'`。

原因：

- 第二轮 mypy 修复把 `fused_moe.py` 里的 target-main `RoutedExperts` 别名改成
  `_TargetRoutedExperts`（避免与 v0.22.1 fallback 冲突的 `no-redef`），模块不再导出
  `RoutedExperts`。
- 但 `TestVllmParentInterfaceCompatibility` 仍按旧名字访问 `fused_moe_module.RoutedExperts`。

修复：

- target-main 分支的 parent-interface 用例与 `explicitly_calls_parent` 判断改用
  `fused_moe_module._TargetRoutedExperts`，和源码别名保持一致。
- 该引用只在 `else`（非 v0.22.1）分支求值，v0.22.1 分支不受影响。

### 2. A2/A3 e2e: DeepSeek MoE 走的是上游 plain `MoERunner`/`RoutedExperts`，未命中 Ascend factory

失败（同一根因，两个表现）：

- A3 two-card `test_prefix_caching.py::...[deepseek-ai/DeepSeek-V2-Lite-Chat]`：
  `AttributeError: 'UnquantizedFusedMoEMethod' object has no attribute 'is_monolithic'`。
- A2 one-card `model_runner_v2/test_basic.py::...[vllm-ascend/DeepSeek-V2-Lite-W8A8]`：
  `TypeError: AscendFusedMoEMethod.apply() got an unexpected keyword argument 'shared_experts'`。

两条 traceback 都是
`_moe_forward_shared -> MoERunner._forward_impl(moe_runner.py:821) -> _apply_quant_method`，
即执行的是上游 plain `MoERunner`/`RoutedExperts`，而不是 `AscendMoERunner` /
`AscendRoutedExperts`。说明该 MoE 层根本没走我们注入 `runner_cls` / `routed_experts_cls`
的 factory。

原因（导入顺序）：

- 上游 vLLM PR #41184 把 `FusedMoE` 变成 module-level 函数，OOT `register_oot` 不能再替换它，
  我们改为在 `patch_fused_moe_factory` 里替换 `fused_moe` package 和 `layer` 模块上的
  `FusedMoE` 属性。这种替换只对“替换之后”才发生的 import 生效。
- `worker.py` 中 `adapt_patch()`（line 103）先于 `register_ascend_customop`（line 111）。
  `adapt_patch()` 经 `patch/worker/__init__.py` 导入 `patch_deepseek_mtp`，后者
  `from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM` 把 `deepseek_v2`
  模块提前导入。`deepseek_v2` 顶部 `from ...fused_moe import FusedMoE` 此时绑定的是**原始**
  factory。
- 等到 `patch_fused_moe_factory` 执行时，`deepseek_v2.FusedMoE` 已经指向原始 factory，
  我们的替换够不到它，于是 DeepSeek 构造的是上游 plain runner，触发上面两个错误。
  （Qwen 等模型在 `load_model` 阶段才 import，发生在 patch 之后，因此不受影响。）

修复：

- `patch_fused_moe_factory` 在替换两个模块属性后，遍历 `sys.modules`，把所有
  `vllm.model_executor.models.*` 模块里仍指向原始 factory 的 `FusedMoE` 引用一并重绑到 Ascend
  替换实现。这样提前 import 的 DeepSeek 等模型也会用 `AscendFusedMoE` factory，进而得到
  `AscendMoERunner` + `AscendRoutedExperts`，两个 e2e 报错同时消除。
- 该逻辑位于 `patch_fused_moe_factory` 内部：函数在 `_FUSED_MOE_IS_CLASS`（即
  `vllm_version_is("0.22.1")`）时直接 return，且仅在 `not vllm_version_is("0.22.1")` 时被调用，
  属于 target-main 专属新增逻辑，符合双版本规范。

### 3. A3 four-card: 修好 factory 后暴露 `AscendRoutedExperts` 删除了 `e_score_correction_bias`

失败：

- `tests/e2e/pull_request/four_card/test_deepseek_v4.py`
- 根因：`RuntimeError: Worker failed with error
  "'AscendRoutedExperts' object has no attribute 'e_score_correction_bias'"`，worker 崩溃后
  executor 进入 SIGTERM/SIGKILL 宽限期，导致 CI 一直挂住。

原因：

- 上面第 2 点的 factory 重绑修好后，DeepSeek 才真正走 `AscendRoutedExperts`（之前是 plain
  `RoutedExperts`，更早就在 `is_monolithic` 崩了），于是暴露出本类自身的下一个 bug。
- 上游 `RoutedExperts.__init__`（routed_experts.py:107）把 `e_score_correction_bias` 存为属性；
  deepseek v3/r1/v4 的该字段是 gate 共享的 `nn.Parameter`，会被注册进 `self._parameters`。
- `AscendRoutedExperts.__init__` 为了用 Ascend 的 `local_num_experts/global_num_experts` 重建
  权重，会先 `for param_name in list(self._parameters): delattr(...)` 删掉上游 eager 创建的
  provisional 权重。这个循环把 `e_score_correction_bias` 也一并删了。
- DeepSeek-V2-Lite 的 `e_score_correction_bias` 为 None（不是 Parameter，不在 `_parameters`），
  所以之前没暴露；deepseek_v4 非 None，命中。

修复：

- 删除 provisional 权重的循环里跳过 `e_score_correction_bias`：它是路由字段（gate 共享 bias），
  不是 `create_weights` 会重建的专家权重。保留它与上游 `RoutedExperts` 行为一致（上游同样把它
  注册为参数且不删除）。
- 该逻辑位于 target-main 专属类 `AscendRoutedExperts.__init__` 内，v0.22.1 走 legacy
  `AscendFusedMoE`，不受影响。

### 4. Dense graph: `AscendRoutedExperts.shared_forward_impl` 把 shared input 回传给 routed path

失败：

- `test_qwen3_dense_graph_mode[full_decode_only-False-32-vllm-ascend/DeepSeek-V2-Lite-W8A8]`
- 根因断言：
  `Number of global experts mismatch ... router_experts=2048, expected_experts=64`。

原因：

- 上游 vLLM PR #41184 把 `MoERunner` 拆成 `MoERunner` + `RoutedExperts`，并把 routed hidden
  states 与 shared-expert input 分开传递。
- `AscendRoutedExperts.shared_forward_impl` 已经在本函数内部保存了
  `shared_hidden_states = hidden_states if shared_experts_input is None else shared_experts_input`，
  并在 routed expert 结束后调用 `_forward_shared_experts(shared_hidden_states, ...)`。
- 之前的适配又把 `shared_hidden_states` 作为 `shared_experts_input` 传回
  `AscendFusedMoE.forward_impl`。而该参数在 legacy 兼容路径中会影响 routed 计算输入，导致
  W8A8_DYNAMIC 看到的 logits/输入维度变成 hidden size 2048，而不是 DeepSeek-V2-Lite 的 64 个
  logical experts，于是触发断言。

修复：

- `AscendRoutedExperts.shared_forward_impl` 增加显式双版本分支：
    - `if vllm_version_is("0.22.1")`：保留原调用，继续把 `shared_hidden_states` 作为
    `shared_experts_input` 传入。v0.22.1 没有上游 PR #41184 的 `RoutedExperts` 拆分，正常执行仍走
    legacy `AscendFusedMoE`。
    - `else`：target-main 调用 `self.forward_impl(...)` 时只传
    `hidden_states/router_logits/return_with_event`，不再把 `shared_hidden_states` 回传给 routed path。
- `shared_hidden_states` 仍在后续 `_forward_shared_experts(shared_hidden_states, ...)` 使用，保持
  PR #41184 新增的 routed/shared 输入分离语义。
- 该修复遵循 main2main 双版本规范：并列行为使用 `vllm_version_is("0.22.1")` 显式分支，旧版本分支
  尽量保持原样，新版本分支只处理 PR #41184 引入的新 `RoutedExperts` 语义。

### 4. 内部路由 gate 未生效：`The bias first dim should be same as x second dim`

失败：

- 手动 NPU 跑（路径 `/vllm-workspace/...`）：`RuntimeError: Worker failed with error
  'The bias first dim should be same as x second dim'`，发生在 `determine_available_memory ->
  profile_run` 的 MoE forward，worker 崩溃 -> CI/执行器挂住。

原因（上游 PR #41184 的 gate owner 迁移 + 双版本 `is_internal_router` 语义不一致）：

- 模型用 `self.experts.is_internal_router` 决定：True 时把 `hidden_states` 当 router_logits 占位
  传进去、由 MoE 层内部应用 gate；False 时模型自己先调 gate 再传真正的 router_logits。
- v0.22.1：`self.experts` 就是 `AscendFusedMoE` 实例，模型读到的是
  `AscendFusedMoE.is_internal_router`（语义：gate 存在且有 `weight_fp32`）。只有 deepseek_v4 的
  gate 设了 `precast_fp32_weight` 才有 `weight_fp32`，所以只有它走"层内 fp32 gate"，其它（qwen3、
  deepseek_v2）模型自己应用 gate，行为自洽。
- target main：`self.experts` 变成 `AscendMoERunner`，模型读到的是上游
  `MoERunner.is_internal_router`（语义：`gate is not None`）——对所有带 gate 的 MoE 都为 True。
  于是 qwen3 / deepseek_v2 也走占位路径，但 Ascend 只有在 `weight_fp32` 时才在 `shared_forward_impl`
  里应用 gate，结果 `router_logits` 一直是 `hidden_states`（expert 维 = hidden_size），喂进 kernel
  触发 bias/shape 报错。

修复：

- 给 `AscendMoERunner` 覆写 `is_internal_router`，让 target main 下也采用 Ascend 语义
  （`gate is not None and hasattr(gate, "weight_fp32")`），与 v0.22.1 的
  `AscendFusedMoE.is_internal_router` 对齐。这样：
    - qwen3 / deepseek_v2（无 `weight_fp32`）：runner.is_internal_router=False，模型自己应用 gate，
      传真正的 router_logits，Ascend 直接使用。
    - deepseek_v4（有 `weight_fp32`）：runner.is_internal_router=True，模型传占位，Ascend 在
      `shared_forward_impl` 走 fp32 gate。
- 双版本：`if vllm_version_is("0.22.1")` 分支保留上游 runner 语义
  （`MoERunner.is_internal_router.fget(self)`），仅 target main 改用 Ascend 语义。v0.22.1 的模型本来
  就读 `AscendFusedMoE.is_internal_router`，不受影响。

### 5. CPU UT：310P 测试触发 factory & compressed-tensors MoE target 不匹配

CPU UT 7 个失败（`error.py`），两类根因。

#### 5.1 `tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py`（6 个）

- 报错：`TypeError: FusedMoE() missing 4 required positional arguments`，栈在
  `fused_moe.py:_create_ascend_fused_moe_runner` 的 `return FusedMoE(*args, **kwargs)`。
- 原因：测试用 `AscendFusedMoE310.__new__(AscendFusedMoE310)` 造半初始化对象，但 target main 下
  `AscendFusedMoE310.__new__` 会委托到上游 factory（无参 → 报错）。这和第二轮给
  `test_fused_moe.py` 加 `_new_uninitialized_ascend_fused_moe()` 是同一类问题。
- 修复：`_build_layer` 改用 `torch.nn.Module.__new__(AscendFusedMoE310)` 绕过 factory；v0.22.1 下
  对 nn.Module 对象等价。

#### 5.2 `tests/ut/quantization/test_compressed_tensors_config.py::test_get_moe_quant_method`

- 报错：`KeyError: None`，栈在 `compressed_tensors_config.py:get_scheme_dict` 的
  `self.target_scheme_map[matched_target]`。
- 原因：`find_matched_target` 用 `module.__class__.__name__`（`check_contains=True`）匹配 target。
  上游 PR #41184 后 MoE owner 是 `RoutedExperts`（生产是 `AscendRoutedExperts`），但
  `_add_fused_moe_to_target_scheme_map` 只加了 `"FusedMoE"` target，匹配不到 → `matched_target=None`
  → KeyError。
- 修复：`_add_fused_moe_to_target_scheme_map` 按版本注册 target —— target main 加 `"RoutedExperts"`，
  v0.22.1 保留 `"FusedMoE"`。`check_contains=True` 让 `"RoutedExperts"` 既匹配 UT 的
  `RoutedExperts` mock（精确），也匹配生产的 `AscendRoutedExperts`（子串包含），与 legacy
  `"FusedMoE"` 匹配 `AscendFusedMoE` 的方式一致。

## 第四轮 CI 失败与修复

CI run: <https://github.com/vllm-project/vllm-ascend/actions/runs/27618176532?pr=10459>

### 1. A3 two-card EPLB: `AscendMoERunner` 缺 `local_num_experts`

失败：

- `tests/e2e/pull_request/two_card/test_qwen3_moe_eplb.py`
- 根因：`VllmEplbAdaptor` 仍读取
  `self.model.model.layers[-1].mlp.experts.local_num_experts`，target main 下
  `mlp.experts` 已是 `AscendMoERunner`，真实 EPLB/权重字段在 `runner.routed_experts`。

修复：

- 新增 `eplb/utils.py::get_moe_weight_owner`：
    - `vllm_version_is("0.22.1")`：返回 legacy `mlp.experts`。
    - `else`：返回 target-main `mlp.experts.routed_experts`。
- `VllmEplbAdaptor` 和 `eplb/utils.py` 中读取 `local_num_experts`、`quant_type`、
  `global_expert_map`、`moe_load`、`log2phy_map` 的位置统一通过该 helper。
- 注释说明这是 vLLM PR #41184 将 MoE owner 从 `FusedMoE` 移到 `RoutedExperts` 导致的变更。

### 2. A3 disaggregated encoder: full generator 多出 `chat_template_kwargs`

失败：

- `tests/e2e/pull_request/two_card/test_disaggregated_encoder.py`
- 根因：`patch_minimax_usage_accounting.py` wrapper 仍按旧签名调用
  `OpenAIServingChat.chat_completion_full_generator(..., reasoning_parser)`；target main 上游
  PR #45190/#45171 已把 generator 签名改成接收 `chat_template_kwargs`，不再接收
  `reasoning_parser`。

修复：

- wrapper 增加显式双版本分支：
    - v0.22.1 保留旧调用形状，继续传 `reasoning_parser`。
    - target main 不再传 `reasoning_parser`，改为透传 `chat_template_kwargs` 等 extra kwargs。
- target main 下为 usage accounting 按需用 `reasoning_parser_cls(tokenizer, chat_template_kwargs=...)`
  重建 reasoning parser，保持 MiniMax reasoning tokens 统计能力。
- stream/full 两个 generator wrapper 一起修，避免下一轮流式路径触发同类参数错误。

### 3. 310P MoE: target factory 走到通用 Ascend MoE op

失败：

- `tests/e2e/pull_request/four_card/_310p/test_moe_model_310p.py`
- 根因：310P target-main `AscendFusedMoE310.__new__` 委托到了通用
  `_create_ascend_fused_moe_runner`，于是运行时使用通用 Ascend `AscendRoutedExperts` /
  `AscendUnquantizedFusedMoEMethod`，进入 `torch.ops._C_ascend.moe_gating_top_k`、
  `npu_moe_init_routing_custom` 等 910B/C 路径；310P 环境没有这些 op。

修复：

- 增加 `_create_ascend_fused_moe_runner_310`，target main 下通过 vLLM PR #41184 新增的
  `runner_cls` / `routed_experts_cls` 扩展点注入 `AscendMoERunner310` 和
  `AscendRoutedExperts310`。
- `AscendRoutedExperts310` 在 target main 的 `RoutedExperts` owner 上重建 310P 权重，继续使用
  `AscendUnquantizedFusedMoEMethod310` / `AllGatherCommImpl310`。
- `shared_forward_impl` 按版本处理 vLLM PR #41184 新增的 `shared_experts_input`：v0.22.1 保持
  单输入语义，target main 只在 shared path 使用该输入，routed path 继续使用真实 router logits。

## 第五轮 CI 失败与修复

CI run: <https://github.com/vllm-project/vllm-ascend/actions/runs/27635141589?pr=10459>

### 1. A2 one-card dense graph: runner custom-op 入口仍可能拿到 hidden placeholder

失败：

- Job:
  <https://github.com/vllm-project/vllm-ascend/actions/runs/27635141589/job/81721101298?pr=10459>
- `tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py::test_qwen3_dense_graph_mode`
- 模型：`vllm-ascend/DeepSeek-V2-Lite-W8A8`
- 根因断言：
  `Number of global experts mismatch ... router_experts=2048, expected_experts=64`。

原因：

- 第三轮里已经修过 `AscendRoutedExperts.shared_forward_impl` 不再把
  `shared_experts_input` 回传给 routed path，但 CI 栈显示失败仍从 target-main 的
  `MoERunner._moe_forward_shared -> AscendMoERunner._forward_impl -> AscendMoERunner.forward_impl`
  进入。
- 上游 vLLM PR #41184 后，graph/custom-op 入口会把 `router_logits` 和
  `shared_experts_input` 分开传进 `MoERunner._forward_impl`。当 runner 被视为 internal router 时，
  第二个参数可能仍是 `hidden_states` placeholder。
- Ascend 的 target-main runner 又把执行委托给 `AscendRoutedExperts`，如果不先在 runner 层把
  placeholder 归一成真实 router logits，旧 Ascend `forward_impl` 会把 hidden size 2048 当成 expert
  数，继续触发 W8A8_DYNAMIC 的 expert 数校验。

修复：

- 在 `AscendMoERunner.forward_impl` 增加显式双版本分支：
    - `if vllm_version_is("0.22.1")`：保持旧行为，直接使用传入的 `router_logits`。
    - `else`：调用 target-main helper，在进入 `self.routed_experts.forward_impl` /
      `shared_forward_impl` 前归一化 router logits。
- 新增 `_maybe_apply_target_main_internal_router`，且新增逻辑用
  `if not vllm_version_is("0.22.1")` 包住：
    - 如果 `router_logits.shape[-1]` 已等于 `moe_config.num_logical_experts` /
      `moe_config.num_experts`，说明模型已经应用 gate，直接返回。
    - 如果 runner 持有 gate 且 `router_logits` 仍是 hidden placeholder，则通过 gate 重新算真实
      expert logits。
    - 对 FSE fuse gate 路径保留上游 PR #41184 的 combined gate 处理。
- 注释说明该变更来自上游 PR #41184：router gate 迁移到 `MoERunner`，模型可把
  `hidden_states` 作为 `router_logits` placeholder 传入。
- 新增 UT：
  `tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_forward_impl_normalizes_target_main_router_placeholder`，
  覆盖 shared/no-shared 两条路径，固定 2048 hidden placeholder 被归一成 64 expert logits。

建议补跑：

- `pytest -sv tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_forward_impl_normalizes_target_main_router_placeholder`
- `pytest -sv tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py::test_qwen3_dense_graph_mode`
- `pytest -sv tests/e2e/pull_request/one_card/model_runner_v2/test_basic.py`

### 2. 310P one-card: Mamba prefix cache scheduler 签名不兼容

失败：

- Job:
  <https://github.com/vllm-project/vllm-ascend/actions/runs/27635141589/job/81721101301?pr=10459>
- `tests/e2e/pull_request/one_card/_310p/test_dense_model_310p.py::test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16`
- 根因：
  `TypeError: _mamba_block_aligned_split() takes from 3 to 5 positional arguments but 6 were given`。

原因：

- 上游 vLLM PR #37898 给
  `Scheduler._mamba_block_aligned_split` 增加了
  `num_uncached_common_prefix_tokens` 参数，并新增 Marconi-style prefix-cache admission 逻辑。
- `vllm_ascend/patch/platform/patch_scheduler.py` 覆盖了上游 scheduler 方法，但仍是旧签名，只接
  `request`、`num_new_tokens`、`num_new_local_computed_tokens`、
  `num_external_computed_tokens`。
- target main 的 scheduler 在 hybrid Mamba prefix-cache align 模式下传入第 5 个上下文参数，
  覆盖后的旧函数直接 TypeError。

修复：

- `patch_scheduler.py::_mamba_block_aligned_split` 增加
  `num_uncached_common_prefix_tokens: int = 0`，保证 target-main 调用形状兼容。
- 新增逻辑用 `if not vllm_version_is("0.22.1")` 包住：
    - 当 uncached common prefix 长度大于等于 block size，且当前计划 token 数超过该 common prefix
      长度时，把 `num_new_tokens` 截断到 common prefix 边界。
    - 截断后继续按 block size 对齐，保持 Mamba state cache 和 attention prefix cache 一致。
- v0.22.1 分支保持旧行为：即使测试传入第 5 个参数，也不启用 PR #37898 的 admission 逻辑。
- 新增 UT：
  `tests/ut/patch/platform/test_patch_scheduler.py`
    - target-main 分支验证第 5 个参数生效。
    - v0.22.1 分支验证旧行为不变。

建议补跑：

- `pytest -sv tests/ut/patch/platform/test_patch_scheduler.py`
- `pytest -sv tests/e2e/pull_request/one_card/_310p/test_dense_model_310p.py::test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16`
- `pytest -sv tests/e2e/pull_request/one_card/_310p/test_dense_model_310p.py`

## 本地验证状态

本地已做：

- `git diff --check` 通过。
- `.venv/bin/python -m py_compile` 覆盖本轮修改文件，通过。
- `PATH=.venv/bin:$PATH bash format.sh` 通过。
- 本地尝试 `python3 -m mypy --ignore-missing-imports ...`，但当前 `.venv` 缺完整 target vLLM/torch-npu 类型环境，仍会展开到大量无关导入文件；该结果不能等价 CI mypy。已根据输出额外清理 quant config / UT 中的 `RoutedExperts = None` 静态类型风险，同时保持 v0.22.1 分支尽量原样。

本地未做：

- pytest / e2e 未跑。本地 `.venv` 没有安装 `pytest` / `vllm`，也没有完整 torch-npu/NPU 运行环境，
  行为验证依赖 CI。
