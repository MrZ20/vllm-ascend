#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendFusedMoERouter,
    AscendMoERunner,
    AscendRoutedExperts,
    AscendSharedExpertsExecutor,
    AscendUnquantizedFusedMoEMethod,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, _MoECommMethods
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.base import require_topk_weights_and_ids
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import maybe_trans_nz

from .experts_selector import select_experts as select_experts_310
from .moe_comm_method import AllGatherCommImpl310


def _wrap_fused_experts_result(result: FusedExpertsResult | torch.Tensor) -> FusedExpertsResult:
    if isinstance(result, FusedExpertsResult):
        return result
    return FusedExpertsResult(routed_out=result)


def _install_310p_allgather_comm(moe_config) -> None:
    _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl310(moe_config)


class AscendFusedMoERouter310(AscendFusedMoERouter):
    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids

        if self.hash_enabled or self.hash_indices_table is not None or self.tid2eid is not None:
            raise RuntimeError("hash-based MoE routing is not supported on 310P.")

        topk_weights, topk_ids = select_experts_310(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            global_num_experts=self.num_logical_experts,
        )

        if self.zero_expert_type is not None:
            topk_ids, topk_weights, self._zero_expert_output = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=self.num_logical_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )

        if topk_indices_dtype is not None:
            topk_ids = topk_ids.to(topk_indices_dtype)

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        if _EXTRA_CTX.in_profile_run:
            random_matrix = torch.rand(topk_ids.size(0), self.num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        return topk_weights.to(hidden_states.dtype), topk_ids


class AscendUnquantizedFusedMoEMethod310(AscendUnquantizedFusedMoEMethod):
    def process_weights_after_loading(self, layer):
        super(AscendUnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        w13_data = maybe_trans_nz(w13_data)
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        w2_data = maybe_trans_nz(w2_data)
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> FusedExpertsResult:
        del log2phy, global_redundant_expert_num, mc2_mask

        topk_weights, topk_ids = require_topk_weights_and_ids(
            topk_weights,
            topk_ids,
            "[vllm-ascend/310P/unquantized]",
        )
        topk_weights = topk_weights.to(x.dtype)
        result = _EXTRA_CTX.moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=QuantType.NONE,
                dynamic_eplb=False,
                expert_map=expert_map,
                pertoken_scale=pertoken_scale,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ),
        )
        return _wrap_fused_experts_result(result)


class AscendRoutedExperts310(AscendRoutedExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_310p_topology()
        self.moe_config.supports_eplb = False
        self.global_expert_map = None
        self.local_expert_map = None
        self.log2phy = None
        self.global_redundant_expert_num = 0
        self.dynamic_eplb = False
        _install_310p_allgather_comm(self.moe_config)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is None:
            return AscendUnquantizedFusedMoEMethod310(moe_config)

        quant_method = quant_config.get_quant_method(self, prefix)
        assert quant_method is not None
        return quant_method

    def _get_quant_type(self) -> QuantType:
        quant_type = super()._get_quant_type()
        if quant_type not in (QuantType.NONE, QuantType.W8A8):
            raise RuntimeError("Only Unquant and W8A8 is supported on 310P.")
        return quant_type

    def _validate_310p_topology(self) -> None:
        if self.moe_config.ep_size > 1:
            raise RuntimeError("Expert Parallel is not supported on 310P. Please remove --enable-expert-parallel.")
        if self.ascend_topology.dynamic_eplb:
            raise RuntimeError("Dynamic EPLB is not supported on 310P.")
        if self.ascend_topology.upstream_num_redundant_experts != 0:
            raise RuntimeError("Redundant experts are not supported on 310P.")
        if self.ascend_topology.mix_placement:
            raise RuntimeError("mix_placement is not supported on 310P.")
        if self.local_num_experts != self.global_num_experts:
            raise RuntimeError("310P requires local_num_experts to equal global_num_experts.")


class AscendSharedExpertsExecutor310(AscendSharedExpertsExecutor):
    pass


class AscendMoERunner310(AscendMoERunner):
    router_cls = AscendFusedMoERouter310
    shared_experts_executor_cls = AscendSharedExpertsExecutor310

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.moe_config.ep_size > 1:
            raise RuntimeError("Expert Parallel is not supported on 310P. Please remove --enable-expert-parallel.")
        _install_310p_allgather_comm(self.moe_config)
