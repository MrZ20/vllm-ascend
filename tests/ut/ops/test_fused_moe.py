#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest_mock import MockerFixture
from vllm.config import set_current_vllm_config

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendFusedMoERouter,
    AscendMoERunner,
    AscendRoutedExperts,
    AscendSharedExpertsExecutor,
    AscendUnquantizedFusedMoEMethod,
    _create_ascend_fused_moe_runner,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, adapt_patch

adapt_patch(True)


def mock_ep_and_mc2_group(mocker):
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.rank = 0
    mock_group.world_size = 4
    mock_group.device_group = "mock_group_ep"
    mock_group.all_to_all = MagicMock(return_value=torch.randn(8, 8))
    return mock_group


def mock_dp_and_tp_group(mocker):
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 2
    mock_group.device_group = "mock_group"
    mock_group.all_gather = MagicMock(return_value=torch.randn(10, 32))
    return mock_group


def mock_npu_format_cast(weight_data, format):
    return weight_data


def build_mlp_compute_input_fixture(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    group_list: torch.Tensor,
    with_quant: bool,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    w1_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w2_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    activation: str = "silu",
    need_trans: bool = True,
    dynamic_eplb: bool = False,
) -> MoEMlpComputeInput:
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=dynamic_scale,
        topk_scales=topk_scales,
        weights=MoEWeights(
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8 if with_quant else QuantType.NONE),
        fusion=fusion,
        activation=activation,
        need_trans=need_trans,
        dynamic_eplb=dynamic_eplb,
    )


@pytest.fixture(autouse=True)
def setup_vllm_config_mock(mocker: MockerFixture):
    mock_hf_config = MagicMock()
    mock_hf_config.model_type = "llama"

    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config

    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.parallel_config = MagicMock(tensor_parallel_size=2)
    mock_vllm_config.scheduler_config = MagicMock(max_num_seqs=4)
    mock_vllm_config.model_config.max_model_len = 2048

    mocker.patch("vllm_ascend.ops.fused_moe.fused_moe.get_current_vllm_config", return_value=mock_vllm_config)


@pytest.fixture
def mock_dist_env(mocker: MockerFixture):
    mock_moe_comm_method = MagicMock()

    def mock_prepare(hidden_states, router_logits, **kwargs):
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=kwargs.get("mc2_mask"),
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    mock_moe_comm_method.prepare.side_effect = mock_prepare

    mock_fused_experts_result = torch.randn(16, 2)
    mock_moe_comm_method.fused_experts.return_value = mock_fused_experts_result

    def mock_finalize(hidden_states, **kwargs):
        return hidden_states

    mock_moe_comm_method.finalize.side_effect = mock_finalize
    dp_metadata = MagicMock(num_tokens_across_dp_cpu=[5, 5])
    mock_weight_prefetch_method = MagicMock()
    mock_forward_context_obj = MagicMock(
        moe_comm_method=mock_moe_comm_method,
        moe_comm_type=MoECommType.MC2,
        max_tokens_across_dp=10,
        dp_metadata=dp_metadata,
        mc2_mask=torch.zeros(16, dtype=torch.bool),
        padded_num_tokens=16,
        with_quant=False,
    )

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_ep_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_mc2_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_tp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.model_executor.layers.fused_moe.layer.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.model_executor.layers.fused_moe.config.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config",
            return_value=MagicMock(enable_multistream_moe=False, expert_map_path=None),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.init_eplb_config",
            return_value=(torch.tensor([0, 1, 2, -1, -1, -1, -1, -1]), None, 0),
        ),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_forward_context", return_value=mock_forward_context_obj),
        patch("vllm_ascend.ascend_forward_context.get_forward_context", return_value=mock_forward_context_obj),
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.MC2CommImpl._get_token_dispatcher", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.AlltoAllCommImpl._get_token_dispatcher", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.AllGatherCommImpl._get_token_dispatcher", return_value=None),
        patch(
            "vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method",
            return_value=mock_weight_prefetch_method,
        ),
    ):
        yield {
            "mock_forward_context_obj": mock_forward_context_obj,
            "mock_moe_comm_method": mock_moe_comm_method,
        }


@pytest.fixture
def default_moe_config():
    return {"num_experts": 8, "top_k": 2, "hidden_size": 512, "intermediate_size": 1024}


@pytest.fixture
def moe_method(mock_dist_env):
    moe = MagicMock()
    moe.moe_parallel_config.return_value = MagicMock(ep_size=4)
    moe.moe_parallel_config.use_ep = False
    moe.moe_parallel_config.dp_size = 1
    return AscendUnquantizedFusedMoEMethod(moe)


def test_ascend_unquantized_skips_upstream_modular_kernel_init():
    method = AscendUnquantizedFusedMoEMethod.maybe_make_prepare_finalize

    assert method(object()) is None


class Device(TypedDict):
    device_id: int
    device_expert: list[int]


class Layer(TypedDict):
    layer_id: int
    device_count: int
    device_list: list[Device]


class MockData(TypedDict):
    moe_layer_count: int
    layer_list: list[Layer]


class MockQuantMethod(nn.Module):
    def __init__(self, shared_experts, num_tokens):
        super().__init__()
        if shared_experts:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32), torch.randn(num_tokens, 10)))
        else:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32)))


def _drop_self(signature: inspect.Signature) -> list[inspect.Parameter]:
    params = list(signature.parameters.values())
    if params and params[0].name == "self":
        return params[1:]
    return params


def _format_signature_mismatch(method_name: str, issues: list[str]) -> str:
    return f"{method_name} signature is not aligned with vLLM parent: " + "; ".join(issues)


def _assert_child_signature_accepts_parent_interface(child_method, parent_method):
    child_params = _drop_self(inspect.signature(child_method))
    parent_params = _drop_self(inspect.signature(parent_method))
    child_by_name = {
        param.name: param
        for param in child_params
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    child_has_var_positional = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in child_params)
    child_has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in child_params)
    issues: list[str] = []

    for parent_param in parent_params:
        if parent_param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not child_has_var_positional:
                issues.append("child is missing *args from parent")
            continue

        if parent_param.kind == inspect.Parameter.VAR_KEYWORD:
            if not child_has_var_keyword:
                issues.append("child is missing **kwargs from parent")
            continue

        child_param = child_by_name.get(parent_param.name)
        if child_param is None:
            if parent_param.kind == inspect.Parameter.KEYWORD_ONLY:
                if not child_has_var_keyword:
                    issues.append(f"missing keyword-only parameter {parent_param.name!r}")
            elif not child_has_var_positional and not child_has_var_keyword:
                issues.append(f"missing parameter {parent_param.name!r}")
            continue

        if parent_param.kind != child_param.kind:
            issues.append(
                f"parameter {parent_param.name!r} has kind {child_param.kind!s}, expected {parent_param.kind!s}"
            )

    parent_param_names = {param.name for param in parent_params}
    for child_param in child_params:
        if child_param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if child_param.name in parent_param_names:
            continue
        if child_param.default is inspect.Parameter.empty:
            issues.append(f"extra parameter {child_param.name!r} must be optional")

    assert not issues, _format_signature_mismatch(parent_method.__qualname__, issues)


def _method_uses_super(method) -> bool:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False

    tree = ast.parse(textwrap.dedent(source))
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
        for node in ast.walk(tree)
    )


class TestVllmParentInterfaceCompatibility:
    @pytest.mark.parametrize(
        "child_cls,parent_cls,method_name",
        [
            (AscendUnquantizedFusedMoEMethod, fused_moe_module.UnquantizedFusedMoEMethod, "__init__"),
            (
                AscendUnquantizedFusedMoEMethod,
                fused_moe_module.UnquantizedFusedMoEMethod,
                "process_weights_after_loading",
            ),
            (AscendUnquantizedFusedMoEMethod, fused_moe_module.UnquantizedFusedMoEMethod, "apply"),
            (AscendMoERunner, fused_moe_module.MoERunner, "_forward_impl"),
        ],
    )
    def test_overridden_method_signature_accepts_parent_interface(self, child_cls, parent_cls, method_name):
        child_method = getattr(child_cls, method_name)
        if not _method_uses_super(child_method):
            pytest.skip(
                f"{child_cls.__name__}.{method_name} does not call "
                "super(), so parent interface alignment is not "
                "required"
            )

        if not hasattr(parent_cls, method_name):
            pytest.fail(
                f"{child_cls.__name__}.{method_name} calls super(), but {parent_cls.__name__} has no {method_name}"
            )

        _assert_child_signature_accepts_parent_interface(
            child_method,
            getattr(parent_cls, method_name),
        )


class TestAscendUnquantizedFusedMoEMethod:
    def _build_layer(self, *, has_bias=True, zero_expert_num=0):
        layer = MagicMock()
        layer.w13_weight = nn.Parameter(torch.randn(2, 3, 4))
        layer.w2_weight = nn.Parameter(torch.randn(2, 4, 3))
        layer.w13_bias = torch.randn(2, 4) if has_bias else None
        layer.w2_bias = torch.randn(2, 3) if has_bias else None
        layer.zero_expert_num = zero_expert_num
        layer.zero_expert_type = "identity" if zero_expert_num > 0 else None
        layer.n_shared_experts = 0
        layer.moe_config = SimpleNamespace(num_logical_experts=None)
        layer.layer_id = 3
        layer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False))
        return layer

    @pytest.mark.parametrize("enable_fused_mc2", [True, False])
    def test_process_weights_after_loading_transposes_and_formats(self, monkeypatch, enable_fused_mc2):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
        layer = self._build_layer()
        original_w13 = layer.w13_weight.detach().clone()
        original_w2 = layer.w2_weight.detach().clone()
        format_cast = MagicMock(side_effect=lambda weight, _: weight)
        maybe_trans_nz = MagicMock(side_effect=lambda weight: weight)

        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_fused_mc2 = enable_fused_mc2
        monkeypatch.setattr(fused_moe_module, "get_ascend_config", lambda: mock_ascend_config)
        monkeypatch.setattr(fused_moe_module.torch_npu, "npu_format_cast", format_cast)
        monkeypatch.setattr(fused_moe_module, "maybe_trans_nz", maybe_trans_nz)

        method.process_weights_after_loading(layer)

        torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2).contiguous())
        torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2).contiguous())
        if enable_fused_mc2:
            assert format_cast.call_count == 2
            maybe_trans_nz.assert_not_called()
        else:
            assert maybe_trans_nz.call_count == 2
            format_cast.assert_not_called()

    @pytest.mark.parametrize("moe_comm_type", [MoECommType.MC2, MoECommType.FUSED_MC2])
    def test_apply_builds_fused_experts_input(self, monkeypatch, moe_comm_type):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=True)
        method.dynamic_eplb = False
        method.tid2eid = None
        layer = self._build_layer(has_bias=True)
        hidden_states = torch.randn(2, 4, dtype=torch.float16)
        topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = FusedExpertsResult(routed_out=torch.ones_like(hidden_states))
        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(moe_comm_type=moe_comm_type, moe_comm_method=moe_comm_method),
        )

        result = method.apply(
            layer=layer,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=True,
            activation="gelu",
            pertoken_scale=torch.ones(2),
            mc2_mask=torch.tensor([True, False]),
        )

        torch.testing.assert_close(result.routed_out, torch.ones_like(hidden_states))
        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        assert fused_input.hidden_states is hidden_states
        torch.testing.assert_close(fused_input.topk_weights, topk_weights.to(hidden_states.dtype))
        assert torch.equal(fused_input.topk_ids, topk_ids)
        assert fused_input.weights.w1_bias is layer.w13_bias
        assert fused_input.weights.w2_bias is layer.w2_bias
        assert fused_input.routing.apply_router_weight_on_input
        assert fused_input.activation == "gelu"
        if moe_comm_type == MoECommType.FUSED_MC2:
            assert fused_input.weights.w1[0] is layer.w13_weight
            assert fused_input.weights.w2[0] is layer.w2_weight
            assert isinstance(fused_input.weights.w1_scale, list)
            assert isinstance(fused_input.weights.w2_scale, list)
        else:
            assert fused_input.weights.w1 is layer.w13_weight
            assert fused_input.weights.w2 is layer.w2_weight
            assert fused_input.weights.w1_scale is None
            assert fused_input.weights.w2_scale is None

    def test_apply_requires_precomputed_topk(self, monkeypatch):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        method.dynamic_eplb = True
        method.tid2eid = None
        layer = self._build_layer(has_bias=False)
        hidden_states = torch.randn(2, 4)

        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.MC2, moe_comm_method=MagicMock()),
        )

        with pytest.raises(ValueError, match="precomputed topk_weights and topk_ids"):
            method.apply(
                layer=layer,
                x=hidden_states,
                topk_weights=None,
                topk_ids=None,
            )


class TestAscendMoERunner:
    @pytest.mark.parametrize(
        "moe_comm_type, flash_comm_v1_enabled, expected",
        [
            (MoECommType.ALLTOALL, False, True),
            (MoECommType.MC2, False, True),
            (MoECommType.FUSED_MC2, False, True),
            (MoECommType.ALLGATHER, False, False),
            (MoECommType.ALLGATHER, True, True),
        ],
    )
    def test_runner_reduction_properties(self, monkeypatch, moe_comm_type, flash_comm_v1_enabled, expected):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        monkeypatch.setattr(fused_moe_module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=moe_comm_type))
        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(moe_comm_type=moe_comm_type, flash_comm_v1_enabled=flash_comm_v1_enabled),
        )

        assert runner.use_dp_chunking is False
        if hasattr(type(runner), "_fused_output_is_reduced"):
            assert runner._fused_output_is_reduced is expected
        if hasattr(runner, "_maybe_reduce_shared_expert_output"):
            assert runner._maybe_reduce_shared_expert_output("shared") == "shared"

    def test_runner_init_wraps_router_with_ascend_router(self):
        source = inspect.getsource(AscendMoERunner.__init__)

        assert AscendMoERunner.router_cls is AscendFusedMoERouter
        assert "self.router_cls" in source

    def test_runner_delegates_eplb_state_to_routed_experts(self):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        routed_experts = MagicMock()
        routed_experts.local_num_experts = 2
        routed_experts.ep_rank = 1
        routed_experts.ep_size = 4
        routed_experts.global_expert_map = torch.tensor([[0, 1]])
        routed_experts.moe_load = torch.ones(2)
        routed_experts.get_log2phy_map.return_value = torch.tensor([1, 0])
        runner.routed_experts = routed_experts

        assert runner.local_num_experts == 2
        assert runner.ep_rank == 1
        assert runner.ep_size == 4
        assert torch.equal(runner.global_expert_map, torch.tensor([[0, 1]]))
        assert torch.equal(runner.moe_load, torch.ones(2))
        assert torch.equal(runner.get_log2phy_map(), torch.tensor([1, 0]))

        runner.clear_moe_load()
        routed_experts.clear_moe_load.assert_called_once()
        runner.update_expert_map("new_map")
        routed_experts.update_expert_map.assert_called_once_with("new_map")


class TestAscendFactoryWrapper:
    def test_create_ascend_fused_moe_runner_injects_runner_and_routed_experts(self, monkeypatch):
        captured_kwargs = {}

        def original_fused_moe(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            n_shared_experts=None,
            tp_size=None,
            dp_size=None,
            pcp_size=None,
            is_sequence_parallel=False,
            enable_eplb=False,
            num_redundant_experts=0,
            prefix="",
            routed_scaling_factor=1.0,
            activation="silu",
            hash_indices_table=None,
            zero_expert_type=None,
            runner_cls=None,
            routed_experts_cls=None,
            runner_args=None,
            routed_experts_args=None,
        ):
            captured_kwargs.update(locals())
            return "runner"

        topology = SimpleNamespace(
            upstream_num_experts=6,
            upstream_num_redundant_experts=2,
            enable_eplb=True,
        )
        monkeypatch.setattr(fused_moe_module, "_ORIGINAL_FUSED_MOE", original_fused_moe)
        monkeypatch.setattr(
            fused_moe_module,
            "_plan_ascend_moe_topology_for_main",
            MagicMock(return_value=topology),
        )

        result = _create_ascend_fused_moe_runner(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            n_shared_experts=1,
            hash="enabled",
            tid2eid={0: 1},
        )

        assert result == "runner"
        assert captured_kwargs["num_experts"] == 6
        assert captured_kwargs["num_redundant_experts"] == 2
        assert captured_kwargs["enable_eplb"] is True
        assert captured_kwargs["runner_cls"] is AscendMoERunner
        assert captured_kwargs["routed_experts_cls"] is AscendRoutedExperts
        assert captured_kwargs["runner_args"]["tid2eid"] == {0: 1}
        assert captured_kwargs["runner_args"]["hash_enabled"] == "enabled"
        assert captured_kwargs["runner_args"]["ascend_topology"] is topology
        assert captured_kwargs["routed_experts_args"]["ascend_topology"] is topology

    def test_patch_fused_moe_factory_has_no_sys_modules_rebind(self):
        source = inspect.getsource(fused_moe_module.patch_fused_moe_factory)

        assert "sys.modules" not in source
        assert "_rebind_stale_fused_moe_factory_captures" not in source


class TestAscendRoutedExperts:
    def test_get_quant_method_returns_ascend_unquantized_before_weights(self):
        routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
        routed_experts.tid2eid = {0: 1}
        moe_config = MagicMock()

        # NOTE(fused_moe refactor): AscendUnquantizedFusedMoEMethod is a vLLM CustomOp whose
        # __init__ reads the current vLLM config (custom_ops list) and get_ascend_config(); provide both.
        vllm_config = MagicMock()
        vllm_config.compilation_config.custom_ops = ["all"]
        with (
            set_current_vllm_config(vllm_config),
            patch("vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config", return_value=MagicMock()),
        ):
            quant_method = routed_experts._get_quant_method("layer", None, moe_config)

        assert isinstance(quant_method, AscendUnquantizedFusedMoEMethod)
        assert quant_method.moe is moe_config
        assert quant_method.tid2eid == {0: 1}


class TestAscendSharedExpertsExecutor:
    def test_shared_experts_split_with_expert_gate(self):
        runner = SimpleNamespace(
            moe_config=SimpleNamespace(hidden_dim=2, in_dtype=torch.float32),
            multistream_overlap_shared_expert=False,
            shared_multistream_overlap_gate=False,
            quant_type=QuantType.NONE,
        )
        hidden_states = torch.tensor([[1.0, -1.0]])
        gate_up = torch.tensor([[2.0, -2.0]])
        down_out = torch.tensor([[3.0, 4.0]])
        gate_out = torch.tensor([[0.0, 2.0]])
        shared_experts = MagicMock()
        shared_experts.gate_up_proj.return_value = (gate_up, None)
        shared_experts.act_fn.side_effect = lambda tensor: tensor + 1
        shared_experts.down_proj.return_value = (down_out, None)
        shared_experts.expert_gate.return_value = (gate_out, None)
        executor = AscendSharedExpertsExecutor(runner=runner, shared_experts=shared_experts)

        part1_out = executor._shared_experts_part1(hidden_states)
        part2_out = executor._shared_experts_part2(hidden_states, part1_out)

        torch.testing.assert_close(part1_out, gate_up)
        torch.testing.assert_close(part2_out, F.sigmoid(gate_out) * down_out)

    def test_runner_forward_keeps_shared_input_separate_from_routed_hidden_states(self):
        source = inspect.getsource(AscendMoERunner._forward_impl)

        assert "shared_hidden_states =" in source
        assert "shared_experts_input is None" in source
        assert "self.shared_experts_executor.forward_shared_experts" in source
