#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config

from vllm_ascend._310p.fused_moe import fused_moe as fused_moe_310_module
from vllm_ascend._310p.fused_moe.fused_moe import (
    AscendFusedMoERouter310,
    AscendMoERunner310,
    AscendRoutedExperts310,
    AscendSharedExpertsExecutor310,
    AscendUnquantizedFusedMoEMethod310,
)
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendFusedMoERouter,
    AscendMoERunner,
    AscendRoutedExperts,
    AscendSharedExpertsExecutor,
    _create_ascend_fused_moe_runner,
)


def _make_router_310() -> AscendFusedMoERouter310:
    router = AscendFusedMoERouter310.__new__(AscendFusedMoERouter310)
    router.top_k = 2
    router.num_logical_experts = 8
    router.use_grouped_topk = False
    router.renormalize = True
    router.topk_group = None
    router.num_expert_group = None
    router.custom_routing_function = None
    router.scoring_func = "softmax"
    router.routed_scaling_factor = 1.0
    router.e_score_correction_bias = None
    router.tid2eid = None
    router.hash_enabled = None
    router.hash_indices_table = None
    router.zero_expert_type = None
    router.capture_fn = MagicMock()
    router._zero_expert_output = None
    return router


def _make_topology(**overrides):
    values = {
        "dynamic_eplb": False,
        "upstream_num_redundant_experts": 0,
        "mix_placement": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestFusedMoE310Architecture:
    def test_factory_injects_310p_runner_and_routed_experts(self, monkeypatch):
        captured_kwargs = {}

        # NOTE(fused_moe refactor): the factory extracts args via inspect.signature() +
        # bind_partial(), so the stand-in must expose the real named params (num_experts, ...)
        # rather than a bare **kwargs (which would nest everything under 'kwargs').
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
            upstream_num_experts=8,
            upstream_num_redundant_experts=0,
            enable_eplb=False,
        )
        monkeypatch.setattr(fused_moe_module, "_ORIGINAL_FUSED_MOE", original_fused_moe)
        monkeypatch.setattr(fused_moe_module, "is_310p", lambda: True)
        monkeypatch.setattr(
            fused_moe_module,
            "_plan_ascend_moe_topology_for_main",
            MagicMock(return_value=topology),
        )

        result = _create_ascend_fused_moe_runner(
            num_experts=8,
            top_k=2,
            hidden_size=16,
            intermediate_size=32,
        )

        assert result == "runner"
        assert captured_kwargs["runner_cls"] is AscendMoERunner310
        assert captured_kwargs["routed_experts_cls"] is AscendRoutedExperts310

    def test_310p_classes_are_specializations(self):
        assert issubclass(AscendFusedMoERouter310, AscendFusedMoERouter)
        assert issubclass(AscendRoutedExperts310, AscendRoutedExperts)
        assert issubclass(AscendSharedExpertsExecutor310, AscendSharedExpertsExecutor)
        assert issubclass(AscendMoERunner310, AscendMoERunner)
        forbidden_name = "Ascend" + "FusedMoE310"
        assert not hasattr(fused_moe_310_module, forbidden_name)

    def test_runner_310_selects_310p_router_and_shared_executor(self):
        assert AscendMoERunner310.router_cls is AscendFusedMoERouter310
        assert AscendMoERunner310.shared_experts_executor_cls is AscendSharedExpertsExecutor310

    def test_routed_experts_310_unquantized_method(self):
        routed_experts = AscendRoutedExperts310.__new__(AscendRoutedExperts310)
        routed_experts.tid2eid = None
        moe_config = MagicMock()

        # NOTE(fused_moe refactor): the unquantized method is a vLLM CustomOp; its __init__
        # reads the current vLLM config (custom_ops list) and get_ascend_config(); provide both.
        vllm_config = MagicMock()
        vllm_config.compilation_config.custom_ops = ["all"]
        with (
            set_current_vllm_config(vllm_config),
            patch("vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config", return_value=MagicMock()),
        ):
            quant_method = routed_experts._get_quant_method("layer", None, moe_config)

        assert type(quant_method) is AscendUnquantizedFusedMoEMethod310
        assert quant_method.moe is moe_config

    def test_unquantized_310_apply_requires_precomputed_topk(self):
        # NOTE(fused_moe refactor): vLLM CustomOp __init__ reads the current vLLM config
        # (custom_ops list) and get_ascend_config(); provide both.
        vllm_config = MagicMock()
        vllm_config.compilation_config.custom_ops = ["all"]
        with (
            set_current_vllm_config(vllm_config),
            patch("vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config", return_value=MagicMock()),
        ):
            method = AscendUnquantizedFusedMoEMethod310(MagicMock())

        with pytest.raises(RuntimeError, match="310P/unquantized"):
            method.apply(
                layer=torch.nn.Module(),
                x=torch.randn(4, 8),
                topk_weights=None,
                topk_ids=None,
            )

    def test_router_310_uses_310p_selector(self, monkeypatch):
        router = _make_router_310()
        hidden_states = torch.randn(4, 8)
        router_logits = torch.randn(4, 8)
        topk_weights = torch.randn(4, 2)
        topk_ids = torch.randint(0, 8, (4, 2), dtype=torch.int64)
        mock_selector = MagicMock(return_value=(topk_weights, topk_ids))
        monkeypatch.setattr(fused_moe_310_module, "select_experts_310", mock_selector)
        monkeypatch.setattr(
            fused_moe_310_module,
            "_EXTRA_CTX",
            SimpleNamespace(in_profile_run=False),
        )

        selected_weights, selected_ids = router._select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=torch.int32,
        )

        mock_selector.assert_called_once()
        router.capture_fn.assert_called_once()
        assert selected_weights.dtype == hidden_states.dtype
        assert selected_ids.dtype == torch.int32

    def test_router_310_rejects_hash_routing(self):
        router = _make_router_310()
        router.hash_enabled = True

        with pytest.raises(RuntimeError, match="hash-based MoE routing"):
            router._select_experts(
                hidden_states=torch.randn(4, 8),
                router_logits=torch.randn(4, 8),
            )

    @pytest.mark.parametrize(
        "moe_config, topology, message",
        [
            (SimpleNamespace(ep_size=2), _make_topology(), "Expert Parallel"),
            (SimpleNamespace(ep_size=1), _make_topology(dynamic_eplb=True), "Dynamic EPLB"),
            (
                SimpleNamespace(ep_size=1),
                _make_topology(upstream_num_redundant_experts=1),
                "Redundant experts",
            ),
            (SimpleNamespace(ep_size=1), _make_topology(mix_placement=True), "mix_placement"),
        ],
    )
    def test_routed_experts_310_rejects_unsupported_topology(self, moe_config, topology, message):
        routed_experts = AscendRoutedExperts310.__new__(AscendRoutedExperts310)
        routed_experts.moe_config = moe_config
        routed_experts.ascend_topology = topology
        routed_experts.local_num_experts = 8
        routed_experts.global_num_experts = 8

        with pytest.raises(RuntimeError, match=message):
            routed_experts._validate_310p_topology()
