你说得对，`model_runner_v1.py` 的 `vllm_version_is` 是有效使用点，上一轮这里我误判了。下面只看 **main2main 相关代码改动**，不再评价 workflow 和临时文档。

## 总体判断

这次 main2main 代码主线是合理的：核心是在适配 vLLM PR #41184，把旧的 `FusedMoE` 直接持有权重/runner 的结构，迁移到 target-main 的 `MoERunner -> RoutedExperts` 结构；同时保留 v0.22.1 的 legacy `FusedMoE` 路径。#41184 明确说明了 `MoERunner` 和 `FusedMoE` 关系反转、`FusedMoE` 被重命名为 `RoutedExperts`，并且 MoE 参数路径多了一层 `.experts.routed_experts.<foo>`，所以你现在对 MoE / EPLB / quant / weight loading 的适配方向是对的。([GitHub][1])

我认为当前代码 **大体可以继续沿这个方向收敛**，但合入前建议再修 5 个点，主要是让代码更符合你说的规则：**版本分支更显式、patch 范围更可控、重复逻辑更少、v0.22.1 路径尽量不感知 target-main 结构。**

---

## 1. `ops/fused_moe/fused_moe.py`：核心适配方向正确，但 factory patch 要收敛

### 合理的地方

你这里的版本隔离做得比较完整：

* v0.22.1 继续以 `FusedMoE` 作为基类；
* target-main 下让 `AscendFusedMoE.__new__()` 直接返回 `AscendMoERunner`；
* target-main 下新增 `AscendRoutedExperts`，把旧 Ascend FusedMoE 的权重创建、quant method、EPLB 字段迁移到 `RoutedExperts`；
* `AscendFusedMoE.__init__()` 在 target-main 下直接 raise，避免错误地初始化 legacy 类。

这些设计和 #41184 的结构变化是匹配的。你现在的代码也明确用 `vllm_version_is("0.22.1")` 把 legacy 路径和 target-main 路径分开了。([GitHub][2])

`shared_forward_impl()` 的处理也比较关键。target-main 下 `shared_experts_input` 不能再被直接塞进 routed experts 的 `forward_impl`，否则容易把 hidden states 当成 routing logits；你现在在 target-main 下先走 routed path，再单独 `_forward_shared_experts()`，这个设计是合理的。([GitHub][2])

### 建议修改 1：`patch_fused_moe_factory()` 的 `sys.modules` 扫描有点重

你现在 target-main 下除了 patch `fused_moe_layer.FusedMoE` / `fused_moe_pkg.FusedMoE`，还会遍历 `sys.modules`，把已经 import 过的 `vllm.model_executor.models.*` 里的 `FusedMoE` 引用替换掉。这个能解决“模型模块提前捕获旧 FusedMoE 引用”的问题，但从 review 视角看，它仍然偏 monkey patch。([GitHub][3])

建议优先考虑两个更干净的方案：

第一选择：把 `patch_fused_moe_factory()` 调用时机前移到任何 model module import 之前，这样不需要扫 `sys.modules`。

第二选择：如果确实做不到前移，就把当前 broad rebinding 收窄成“已知受影响模块列表”，并加注释说明原因：

```python
# vLLM PR #41184 changes model modules to capture FusedMoE from
# vllm.model_executor.layers.fused_moe. Some model modules may be imported
# before Ascend replaces the factory, so we rebind only known stale captures.
_TARGET_MAIN_FUSED_MOE_CAPTURE_MODULE_PREFIXES = (
    "vllm.model_executor.models.deepseek",
    "vllm.model_executor.models.qwen",
    # add only verified modules here
)
```

如果你最后保留 `sys.modules` 扫描，也建议把函数名改得更明确，比如：

```python
_rebind_stale_fused_moe_captures_for_target_main()
```

这样 reviewer 会更容易接受：这不是随意 patch，而是 #41184 后 import capture 顺序导致的兼容补丁。

### 建议修改 2：`AscendRoutedExperts` 里删除参数的逻辑建议抽 helper

你现在 target-main 下先调用 `_TargetRoutedExperts.__init__()`，再删除 provisional upstream parameters，最后用 Ascend 的 `create_weights()` 重新创建权重。这是能理解的，因为 #41184 后 `RoutedExperts` 自己会先创建一批上游权重。([GitHub][2])

但这段逻辑在通用 NPU 和 310P 版本里都有类似实现，建议抽成小 helper，避免之后两边不一致：

```python
def _clear_upstream_routed_experts_parameters(module: torch.nn.Module) -> None:
    # vLLM PR #41184 makes RoutedExperts eagerly create expert parameters.
    # Ascend recreates expert weights through its quant_method, so remove only
    # upstream provisional expert parameters while preserving router-side bias.
    for name in list(module._parameters):
        if name != "e_score_correction_bias":
            del module._parameters[name]
```

如果想更安全，可以不要删“所有非 `e_score_correction_bias` 参数”，而是只删已知 expert 权重名，例如 `w13_weight`、`w2_weight`、`w1_weight`、`w3_weight` 等。这样未来上游如果给 `RoutedExperts` 新增一个非 expert 参数，不会被你误删。

---

## 2. `model_runner_v1.py`：你这个使用点是合理的，但建议把 import 也放进版本分支

你说得对，`model_runner_v1.py` 里确实有使用点。`_bind_routed_experts_capturer()` 现在按版本区分：

* v0.22.1：静态图里捕获的是 `FusedMoE`；
* target-main：#41184 后静态图里捕获的是 `MoERunner`，真正的 experts 在 `module.routed_experts` 里。

这个逻辑是对的，而且是 main2main 适配的关键点之一。([GitHub][4])

我建议的小改动是：把 `MoERunner` 和 `FusedMoE` 的 import 也分别放进对应版本分支。这样更符合你的规则：v0.22.1 路径不需要知道 target-main 的类结构，target-main 路径也不需要依赖 legacy 捕获逻辑。

建议写法：

```python
def _bind_routed_experts_capturer(self, modules):
    if vllm_version_is("0.22.1"):
        # vLLM v0.22.1 stores Ascend MoE modules as FusedMoE instances.
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        for module in modules:
            if isinstance(module, FusedMoE):
                module.routed_experts_capturer = self.routed_experts_capturer
        return

    # vLLM PR #41184 moves routed expert parameters under
    # MoERunner.routed_experts, so capture the RoutedExperts object.
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    for module in modules:
        if isinstance(module, MoERunner):
            module.routed_experts.routed_experts_capturer = (
                self.routed_experts_capturer
            )
```

这个不是功能修复，而是代码审美和双版本隔离问题。改完后 reviewers 会更难挑刺。

---

## 3. EPLB：`get_moe_weight_owner()` 是正确抽象，建议增强错误信息

`eplb/utils.py` 里新增 `get_moe_weight_owner()` 是非常好的抽象：v0.22.1 返回 `experts` 自身，target-main 返回 `experts.routed_experts`。这正好把 #41184 的参数路径变化集中在一个地方，避免 `vllm_adaptor.py` 到处写版本分支。([GitHub][5])

`vllm_adaptor.py` 里通过这个 helper 访问 `local_num_experts`、`quant_type`、expert weights、`global_expert_map`，方向也合理。([GitHub][6])

建议把 helper 改得更防御一点：

```python
def get_moe_weight_owner(experts):
    if vllm_version_is("0.22.1"):
        # vLLM PR #41184 has not landed in v0.22.1. The FusedMoE module
        # itself owns expert weights and EPLB metadata.
        return experts

    # vLLM PR #41184 moves expert weights and EPLB metadata under
    # MoERunner.routed_experts.
    if not hasattr(experts, "routed_experts"):
        raise TypeError(
            "Expected target-main MoERunner with routed_experts after "
            "vLLM PR #41184, but got "
            f"{type(experts).__name__}."
        )
    return experts.routed_experts
```

这样如果以后上游结构再变，报错会直接指向 #41184 适配层，而不是在很深的位置炸一个 `AttributeError`。

---

## 4. quantization：整体干净，但 `method_adapters.py` 建议加显式版本分支

`modelslim_config.py`、`compressed_tensors_config.py`、`fp8_config.py` 里都用 `_is_fused_moe_layer()` 区分 v0.22.1 的 `FusedMoE` 和 target-main 的 `RoutedExperts`，这是正确做法。`compressed_tensors_config.py` 里根据版本选择 `"FusedMoE"` 或 `"RoutedExperts"` 加入 target scheme map，也符合 #41184 后的类名变化。([GitHub][7])

主要建议在 `quantization/method_adapters.py`。你现在让 `AscendFusedMoEMethod.apply()` 接收 target-main 新增的 `topk_weights/topk_ids`，然后直接 `del` 掉，并注释说 Ascend quantized schemes 仍在内部选择 expert。这个功能上大概率没问题，但不完全符合你“每个修改点都要 vllm_version_is 分隔”的规则。([GitHub][8])

建议改成显式版本分支：

```python
if vllm_version_is("0.22.1"):
    # vLLM PR #41184 has not landed in v0.22.1, so legacy callers
    # should not pass precomputed routing tensors.
    assert topk_weights is None and topk_ids is None
else:
    # vLLM PR #41184 passes precomputed routing tensors from
    # RoutedExperts.forward_modular. Ascend quantized schemes still select
    # experts inside their own apply path, so keep accepting these kwargs
    # but do not forward them.
    del topk_weights, topk_ids
```

如果你不想引入 assert，也可以写成：

```python
if not vllm_version_is("0.22.1"):
    # comment...
    del topk_weights, topk_ids
```

但我更推荐第一种，因为 v0.22.1 路径如果意外传了新参数，应该直接暴露问题。

---

## 5. 310P MoE：结构跟主路径一致，建议修一个小的清晰性问题

`_310p/fused_moe/fused_moe.py` 的 main2main 适配和主路径基本一致：v0.22.1 继续 legacy `FusedMoE310`，target-main 走 `AscendMoERunner310 + AscendRoutedExperts310`。这个和 #41184 是匹配的。([GitHub][9])

`AscendMoERunner310.is_internal_router` 直接返回 `False`，也合理。310P 当前逻辑看起来不支持 internal router，所以不要为了 target-main 强行复制通用 NPU 的 internal-router gate 逻辑。([GitHub][9])

小建议在 `_310p/quantization/modelslim_config.py`：Linear 分支已经用了

```python
packed = getattr(self, "packed_modules_mapping", {})
```

但 MoE 分支里又直接访问 `self.packed_modules_mapping`。建议统一用 `packed`，避免某些 model type 没初始化这个字段时，Linear 分支安全、MoE 分支不安全。([GitHub][10])

建议改成：

```python
packed = getattr(self, "packed_modules_mapping", {})

if isinstance(layer, torch.nn.Linear):
    if self.is_layer_skipped_ascend(prefix, packed):
        return UnquantizedLinearMethod()
    return AscendModelSlimLinearMethod(self)

if _is_fused_moe_layer(layer):
    if self.is_layer_skipped_ascend(prefix, packed):
        return UnquantizedFusedMoEMethod()
    return AscendFusedMoEMethod(self)
```

这属于小问题，但能提升简洁性和一致性。

---

## 6. parser / usage accounting patch：可以保留，属于上游 API 变更适配

`patch_tool_choice_none_content.py` 这部分是合理 patch。上游 #45190 把 Response API 也统一到 shared `parser.parse()` 路径；#45104 也把 Chat Completions Harmony 相关逻辑统一到 `Parser`，删除了 Harmony-specific serving path。所以你现在按版本分别 patch v0.22.1 的 `_parse_tool_calls` 和 target-main 的 `_extract_tool_calls`，方向是对的。([GitHub][11])

`patch_minimax_usage_accounting.py` 里也按版本处理了 usage accounting 需要的 `reasoning_parser`：v0.22.1 从旧参数拿，target-main 用 `reasoning_parser_cls` 和 `chat_template_kwargs` 重新构造。这和 parser API 重构是对应的。([GitHub][12])

这里我只建议补测试，不建议继续重构。因为 parser 本身就是上游服务层 API 变化，没有明显的 extension point，patch 是可以接受的。

---

## 7. `patch_gpt_oss.py`：target-main-only 分支合理，建议加 idempotent guard

`patch_gpt_oss.py` 只在 `not vllm_version_is("0.22.1")` 时生效，用来把旧的 `.mlp.experts.` 权重名映射到 #41184 后的 `.mlp.experts.routed_experts.`。这个 patch 的范围比较窄，注释也能解释清楚，不属于随意 patch。([GitHub][13])

建议加一个防重复 patch 的 guard，避免测试 reload 或多次 import 时 wrapper 套 wrapper：

```python
if not vllm_version_is("0.22.1") and not hasattr(
    GptOssModel, "_ascend_original_load_weights_other"
):
    GptOssModel._ascend_original_load_weights_other = (
        GptOssModel.load_weights
    )
    GptOssModel.load_weights = ascend_load_weights
```

这不是必须项，但能提升健壮性。

---

## 8. v2 worker gating：可以接受，但注释建议弱化

`patch/worker/__init__.py` 里用

```python
_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.22.1")
```

把 v2 model runner patch 设成 target-main-only。这个在你的规则下是可以接受的，因为 v0.22.1 路径保持原样，不去强行兼容新 v2 patch。([GitHub][14])

`worker/v2/attn_utils.py` 里新增 `causal: bool = True` 并 `del causal`，本质是为了兼容 target-main caller 新增参数。由于整个 v2 patch 已经被 gating 到 target-main-only，这里不一定需要再写 `vllm_version_is`。([GitHub][15])

不过注释里 “v2 model runner intentionally NOT compatible with v0.22.1” 语气有点重。建议换成：

```python
# Target-main-only patch set.
# v0.22.1 keeps the legacy v1 runner path unchanged to avoid importing
# target-main-only worker/v2 APIs.
```

这样更符合“双版本适配”的表达，不会让 reviewer 误解成你放弃了 v0.22.1。

---

## 我建议合入前重点改这 5 个点

1. **`method_adapters.py` 加 `vllm_version_is` 分支**：这是最明显的“功能能跑但不完全符合规则”的点。
2. **`model_runner_v1.py` 把 `FusedMoE` / `MoERunner` import 移进版本分支**：你这里逻辑对，但可以更清晰。
3. **`AscendRoutedExperts` / `AscendRoutedExperts310` 抽一个清理 provisional upstream parameters 的 helper**：减少重复，避免未来两边漂移。
4. **`patch_fused_moe_factory()` 收窄或解释 `sys.modules` rebinding**：这是最容易被 reviewer 质疑“随意 patch”的地方。
5. **`_310p/quantization/modelslim_config.py` MoE 分支也使用 `packed = getattr(...)`**：小修，但更稳。

## 建议补的测试

最值得补的是这几类，不需要很重：

* v0.22.1：`FusedMoE` 路径仍然能被 quant config / model runner capturer 识别。
* target-main：`AscendFusedMoE.__new__()` 返回 `AscendMoERunner`，且 `routed_experts` 是 Ascend 版本。
* target-main：`get_moe_weight_owner()` 返回 `experts.routed_experts`，v0.22.1 返回 `experts` 自身。
* target-main：`compressed_tensors_config` 注册 `"RoutedExperts"`，v0.22.1 注册 `"FusedMoE"`。
* target-main：`patch_gpt_oss.py` 的权重名只在存在 `.routed_experts.` 目标参数时重写，不影响 v0.22.1。

最终我对 main2main 代码的判断是：**方向正确，主要架构适配合理；现在不是“需要大改”的状态，而是需要把几个地方收敛得更像正式双版本适配代码，少一点临时 patch 味道。**

[1]: https://github.com/vllm-project/vllm/pull/41184 "[MoE Refactor] FusedMoE/MoERunner inversion refactor by bnellnm · Pull Request #41184 · vllm-project/vllm · GitHub"
[2]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/ops/fused_moe/fused_moe.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/MrZ20/vllm-ascend/m2m_6_12/vllm_ascend/ops/fused_moe/fused_moe.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/MrZ20/vllm-ascend/m2m_6_12/vllm_ascend/worker/model_runner_v1.py "raw.githubusercontent.com"
[5]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/eplb/utils.py "raw.githubusercontent.com"
[6]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/eplb/adaptor/vllm_adaptor.py "raw.githubusercontent.com"
[7]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/quantization/modelslim_config.py "raw.githubusercontent.com"
[8]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/quantization/method_adapters.py "raw.githubusercontent.com"
[9]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/_310p/fused_moe/fused_moe.py "raw.githubusercontent.com"
[10]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/_310p/quantization/modelslim_config.py "raw.githubusercontent.com"
[11]: https://github.com/vllm-project/vllm/pull/45190 "[Refactor][Parser] Unify Response API to use parser.parse() like Chat Completion API by sfeng33 · Pull Request #45190 · vllm-project/vllm · GitHub"
[12]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/patch/platform/patch_minimax_usage_accounting.py "raw.githubusercontent.com"
[13]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/patch/worker/patch_gpt_oss.py "raw.githubusercontent.com"
[14]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/patch/worker/__init__.py "raw.githubusercontent.com"
[15]: https://github.com/MrZ20/vllm-ascend/raw/m2m_6_12/vllm_ascend/worker/v2/attn_utils.py "raw.githubusercontent.com"
