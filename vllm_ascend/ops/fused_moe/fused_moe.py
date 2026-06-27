#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # type: ignore
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    FusedExpertsResult,
    setup_moe_comm_method,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    enable_sp,
    is_310p,
    maybe_trans_nz,
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    swiglu_limit: float = 0.0


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    swiglu_limit: float = 0.0


@dataclass
class AscendMoETopology:
    original_num_experts: int
    n_shared_experts: int
    placement_num_experts: int
    ascend_num_redundant_experts: int
    upstream_num_experts: int
    upstream_num_redundant_experts: int
    upstream_global_num_experts: int
    local_num_experts: int
    global_expert_map: torch.Tensor | None
    local_expert_map: torch.Tensor | None
    log2phy: torch.Tensor | None
    mix_placement: bool
    dynamic_eplb: bool
    expert_map_path_enabled: bool
    moe_instance_id: int
    tp_size: int
    tp_rank: int
    dp_size: int
    dp_rank: int
    ep_size: int
    ep_rank: int
    sp_size: int
    pcp_size: int
    pcp_rank: int
    use_ep: bool
    enable_eplb: bool
    expert_map_path: str | None


@dataclass
class _AscendEplbPlanningConfig:
    num_experts: int
    ep_size: int
    ep_rank: int


class AscendMoEInstanceRegistry:
    moe_counter = -1
    gate_stream: torch.npu.Stream | None = None

    @classmethod
    def next_moe_instance_id(cls) -> int:
        cls.moe_counter += 1
        return cls.moe_counter

    @classmethod
    def get_gate_stream(cls) -> torch.npu.Stream:
        if cls.gate_stream is None:
            cls.gate_stream = torch.npu.Stream()
        return cls.gate_stream


def _make_ascend_moe_parallel_config(
    *,
    tp_size: int | None,
    dp_size: int | None,
    pcp_size: int | None,
    is_sequence_parallel: bool,
) -> FusedMoEParallelConfig:
    vllm_config = get_current_vllm_config()
    tp_size_ = tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
    dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
    pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size
    sp_size_ = tp_size_ if is_sequence_parallel else 1
    return FusedMoEParallelConfig.make(
        tp_size_=tp_size_,
        pcp_size_=pcp_size_,
        dp_size_=dp_size_,
        sp_size_=sp_size_,
        vllm_parallel_config=vllm_config.parallel_config,
    )


def _plan_ascend_moe_topology_for_main(
    *,
    num_experts: int,
    n_shared_experts: int | None,
    tp_size: int | None,
    dp_size: int | None,
    pcp_size: int | None,
    is_sequence_parallel: bool,
    enable_eplb: bool,
    num_redundant_experts: int,
    prefix: str,
    num_redundant_experts_was_explicit: bool = False,
) -> AscendMoETopology:
    del prefix

    ascend_config = get_ascend_config()
    eplb_config = ascend_config.eplb_config
    shared_experts = n_shared_experts or 0
    mix_placement = getattr(ascend_config, "mix_placement", False)
    placement_num_experts = num_experts + shared_experts if mix_placement else num_experts
    moe_instance_id = AscendMoEInstanceRegistry.next_moe_instance_id()
    moe_parallel_config = _make_ascend_moe_parallel_config(
        tp_size=tp_size,
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=is_sequence_parallel,
    )

    if is_310p():
        expert_map_path = getattr(eplb_config, "expert_map_path", None)
        if moe_parallel_config.ep_size > 1:
            raise RuntimeError("Expert Parallel is not supported on 310P. Please remove --enable-expert-parallel.")
        if enable_eplb or eplb_config.dynamic_eplb or expert_map_path:
            raise RuntimeError("EPLB is not supported on 310P.")
        if num_redundant_experts:
            raise RuntimeError("Redundant experts are not supported on 310P.")
        if mix_placement:
            raise RuntimeError("mix_placement is not supported on 310P.")

        return AscendMoETopology(
            original_num_experts=num_experts,
            n_shared_experts=shared_experts,
            placement_num_experts=num_experts,
            ascend_num_redundant_experts=0,
            upstream_num_experts=num_experts,
            upstream_num_redundant_experts=0,
            upstream_global_num_experts=num_experts,
            local_num_experts=num_experts,
            global_expert_map=None,
            local_expert_map=None,
            log2phy=None,
            mix_placement=False,
            dynamic_eplb=False,
            expert_map_path_enabled=False,
            moe_instance_id=moe_instance_id,
            tp_size=moe_parallel_config.tp_size,
            tp_rank=moe_parallel_config.tp_rank,
            dp_size=moe_parallel_config.dp_size,
            dp_rank=moe_parallel_config.dp_rank,
            ep_size=moe_parallel_config.ep_size,
            ep_rank=moe_parallel_config.ep_rank,
            sp_size=moe_parallel_config.sp_size,
            pcp_size=moe_parallel_config.pcp_size,
            pcp_rank=moe_parallel_config.pcp_rank,
            use_ep=False,
            enable_eplb=False,
            expert_map_path=None,
        )

    planning_config = _AscendEplbPlanningConfig(
        num_experts=placement_num_experts,
        ep_size=moe_parallel_config.ep_size,
        ep_rank=moe_parallel_config.ep_rank,
    )
    vllm_config = get_current_vllm_config()
    eplb_tp_size = getattr(vllm_config.parallel_config, "tensor_parallel_size", tp_size)
    global_expert_map, local_expert_map, log2phy, ascend_num_redundant_experts = init_eplb_config(
        eplb_config,
        moe_instance_id,
        planning_config,
        mix_placement,
        shared_experts,
        tp_size=eplb_tp_size,
    )
    # fused_moe refactor: num_redundant_experts arrives from the upstream model, which
    # forwards vLLM's eplb_config value. On Ascend, redundant experts are configured via
    # the Ascend additional_config eplb_config and computed by init_eplb_config, so a 0
    # from the upstream side just means "not requested via vLLM" and must not be treated
    # as a conflict (Ascend's value is authoritative and is propagated downstream below).
    # Only a non-zero upstream value that disagrees is a real misconfiguration.
    if (
        num_redundant_experts_was_explicit
        and num_redundant_experts
        and num_redundant_experts != ascend_num_redundant_experts
    ):
        raise ValueError(
            "num_redundant_experts does not match Ascend EPLB topology: "
            f"got {num_redundant_experts}, expected {ascend_num_redundant_experts}"
        )

    upstream_num_experts = placement_num_experts
    upstream_num_redundant_experts = ascend_num_redundant_experts
    upstream_global_num_experts = upstream_num_experts + upstream_num_redundant_experts
    local_num_experts = (
        int((local_expert_map != -1).sum().item()) if local_expert_map is not None else upstream_global_num_experts
    )
    expert_map_path = getattr(eplb_config, "expert_map_path", None)
    expert_map_path_enabled = bool(expert_map_path)
    dynamic_eplb = bool(eplb_config.dynamic_eplb and log2phy is not None)

    return AscendMoETopology(
        original_num_experts=num_experts,
        n_shared_experts=shared_experts,
        placement_num_experts=placement_num_experts,
        ascend_num_redundant_experts=ascend_num_redundant_experts,
        upstream_num_experts=upstream_num_experts,
        upstream_num_redundant_experts=upstream_num_redundant_experts,
        upstream_global_num_experts=upstream_global_num_experts,
        local_num_experts=local_num_experts,
        global_expert_map=global_expert_map,
        local_expert_map=local_expert_map,
        log2phy=log2phy,
        mix_placement=mix_placement,
        dynamic_eplb=dynamic_eplb,
        expert_map_path_enabled=expert_map_path_enabled,
        moe_instance_id=moe_instance_id,
        tp_size=moe_parallel_config.tp_size,
        tp_rank=moe_parallel_config.tp_rank,
        dp_size=moe_parallel_config.dp_size,
        dp_rank=moe_parallel_config.dp_rank,
        ep_size=moe_parallel_config.ep_size,
        ep_rank=moe_parallel_config.ep_rank,
        sp_size=moe_parallel_config.sp_size,
        pcp_size=moe_parallel_config.pcp_size,
        pcp_rank=moe_parallel_config.pcp_rank,
        use_ep=moe_parallel_config.use_ep,
        enable_eplb=enable_eplb or dynamic_eplb or expert_map_path_enabled,
        expert_map_path=expert_map_path,
    )


def mock_false():
    return False


def mock_true():
    return True


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.tid2eid = tid2eid

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        # TODO: Current dispatch_ffn_combine fusion operator ONLY supports NZ format.
        # Therefore, we must cast weights to NZ when fusion is enabled.
        # Once the underlying dispatch_ffn_combine operator is updated to support
        # ND format (or other formats), remove this specific 'if' check and the forced
        # npu_format_cast. At that point, the operator should be able to handle weights
        # in their native format without explicit casting here.
        if get_ascend_config().enable_fused_mc2:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        else:
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

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
        if topk_weights is None or topk_ids is None:
            raise ValueError("Ascend MoE apply requires precomputed topk_weights and topk_ids.")
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # NOTE: In the MoECommType.FUSED_MC2 branch, we wrap weights (w1, w2) into lists
        # and provide dummy scales (w1_scale, w2_scale). This is required because:
        # The underlying Ascend fused operator (e.g., dispatch_ffn_combine) expects
        # inputs in a list format.
        # TODO: Passing an empty tensor as scale for float (BF16) cases is semantically
        # incorrect. The ideal solution is to pass None. However, if the underlying
        # dispatch_ffn_combine C++ operator does not support None for the scale argument
        # (due to signature constraints), we are forced to use a placeholder empty tensor.
        # This TODO tracks the requirement to update the C++ operator to accept Optional[Tensor]
        # or None for scales in non-quantized scenarios.
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2 = [layer.w2_weight]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = layer.w13_weight
            w1_scale = None
            w2 = layer.w2_weight
            w2_scale = None
            w1_scale_bias = None
            w2_scale_bias = None

        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=getattr(layer, "dynamic_eplb", self.dynamic_eplb),
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=layer.swiglu_limit,
            )
        )


class AscendFusedMoERouter(FusedMoERouter):
    def __init__(
        self,
        base_router: FusedMoERouter,
        *,
        top_k: int,
        num_logical_experts: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: int | None,
        num_expert_group: int | None,
        custom_routing_function: Callable | None,
        scoring_func: str,
        routed_scaling_factor: float,
        e_score_correction_bias: torch.Tensor | None,
        tid2eid=None,
        hash_enabled=None,
        hash_indices_table: torch.Tensor | None = None,
        zero_expert_type: str | None = None,
        mix_placement: bool = False,
        num_shared_experts: int = 0,
    ):
        self.base_router = base_router
        self._eplb_state = getattr(base_router, "eplb_state", None)
        super().__init__(eplb_state=self._eplb_state)
        self.top_k = top_k
        self.num_logical_experts = num_logical_experts
        self.use_grouped_topk = use_grouped_topk
        self.renormalize = renormalize
        self.topk_group = topk_group
        self.num_expert_group = num_expert_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.tid2eid = tid2eid
        self.hash_enabled = hash_enabled
        self.hash_indices_table = hash_indices_table
        self.zero_expert_type = zero_expert_type
        self.mix_placement = mix_placement
        self.num_shared_experts = num_shared_experts
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self._zero_expert_output: torch.Tensor | None = None

    @property
    def eplb_state(self):
        return getattr(self.base_router, "eplb_state", self._eplb_state)

    @eplb_state.setter
    def eplb_state(self, value) -> None:
        self._eplb_state = value
        if hasattr(self, "base_router"):
            self.base_router.eplb_state = value

    @property
    def routing_method_type(self):
        return self.base_router.routing_method_type

    def set_eplb_state(self, eplb_state) -> None:
        self.eplb_state = eplb_state
        if hasattr(self.base_router, "set_eplb_state"):
            self.base_router.set_eplb_state(eplb_state)

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        self.capture_fn = capture_fn
        if hasattr(self.base_router, "set_capture_fn"):
            self.base_router.set_capture_fn(capture_fn)

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = select_experts(
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
            indices_type=topk_indices_dtype,
            mix_placement=self.mix_placement,
            num_logical_experts=self.num_logical_experts,
            num_shared_experts=self.num_shared_experts,
            num_experts=self.num_logical_experts,
            input_ids=input_ids,
            tid2eid=self.tid2eid,
        )
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        if self.zero_expert_type is not None:
            topk_ids, topk_weights, self._zero_expert_output = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=self.num_logical_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )

        if _EXTRA_CTX.in_profile_run:
            random_matrix = torch.rand(topk_ids.size(0), self.num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        return topk_weights.to(hidden_states.dtype), topk_ids

    @property
    def zero_expert_output(self) -> torch.Tensor | None:
        output = self._zero_expert_output
        self._zero_expert_output = None
        return output


class AscendRoutedExperts(RoutedExperts):
    swiglu_limit: float | None

    def __init__(
        self,
        *args,
        original_num_experts: int,
        original_routed_scaling_factor: float,
        original_activation: str,
        n_shared_experts: int,
        tid2eid=None,
        hash_enabled=None,
        hash_indices_table: torch.Tensor | None = None,
        ascend_topology: AscendMoETopology,
        **kwargs,
    ):
        self.vllm_config = get_current_vllm_config()
        self.ascend_config = get_ascend_config()
        self.original_num_experts = original_num_experts
        self.original_routed_scaling_factor = original_routed_scaling_factor
        self.original_activation = original_activation
        self.n_shared_experts = n_shared_experts
        self.ascend_topology = ascend_topology
        # NOTE(fused_moe refactor): _get_quant_method (below) is invoked by the upstream
        # RoutedExperts.__init__ and reads self.tid2eid, so tid2eid must exist BEFORE super().
        # For deepseek_v4 tid2eid is an nn.Parameter owned by the gate module; set it via
        # object.__setattr__ to bypass nn.Module's "cannot assign parameters before
        # Module.__init__()" guard and to avoid re-registering the gate's Parameter into this
        # module's state_dict.
        object.__setattr__(self, "tid2eid", tid2eid)
        super().__init__(*args, **kwargs)
        self.hash_enabled = hash_enabled
        self.hash_indices_table = hash_indices_table
        # Upstream defaults swiglu_limit to None for models without a swiglu clamp;
        # Ascend kernels expect a float (0.0 == no clamp). Normalize once at the source
        # so every consumer (quant methods, runner, shared experts) sees 0.0.
        if self.swiglu_limit is None:
            self.swiglu_limit = 0.0
        self.global_expert_map = ascend_topology.global_expert_map
        self.log2phy = ascend_topology.log2phy
        self.global_redundant_expert_num = ascend_topology.ascend_num_redundant_experts
        # NOTE(fused_moe refactor): upstream RoutedExperts builds `_expert_map` via its
        # expert_map_manager with length = num_experts + num_redundant_experts (it already
        # bakes in the redundant physical experts). The Ascend MC2 dispatch derives the global
        # physical expert count as `len(expert_map) + global_redundant_expert_num`, so feeding it
        # the upstream map double-counts redundancy (e.g. 130 + 2 -> 66 local experts, exceeding
        # the grouped-matmul kernel's 65-expert limit and breaking the dynamic-EPLB run). Install
        # the Ascend topology's logical expert map (length = num_experts) here — the same map the
        # dynamic-EPLB worker later installs via update_expert_map — so dispatch sizing and weight
        # placement stay consistent.
        if ascend_topology.local_expert_map is not None:
            self.update_expert_map(ascend_topology.local_expert_map.to(self._expert_map.device))
        self.dynamic_eplb = ascend_topology.dynamic_eplb
        self.mix_placement = ascend_topology.mix_placement
        self.moe_instance_id = ascend_topology.moe_instance_id
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        self.quant_type = self._get_quant_type()
        self.multi_stage = False
        self.load_counter = None
        self.num_iter = 0
        self.moe_load = None
        if self.dynamic_eplb:
            eplb_config = self.ascend_config.eplb_config
            self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()
            if eplb_config.eplb_policy_type == 3:
                self.multi_stage = True
                self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
                self.num_iter = eplb_config.expert_heat_collection_interval
                self.moe_load = torch.zeros((self.num_iter, self.local_num_experts), dtype=torch.int32, device="npu")
        setup_moe_comm_method(self.moe_config)
        self._validate_ascend_topology()

    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is None:
            # tid2eid is forwarded so the unquantized hash-routing path (AscendUnquantized
            # FusedMoEMethod) can consume it; it is set before super().__init__ above.
            return AscendUnquantizedFusedMoEMethod(moe_config, tid2eid=self.tid2eid)

        quant_method = quant_config.get_quant_method(self, prefix)
        assert quant_method is not None
        return quant_method

    def _get_quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)

        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)

        return quant_type

    def _validate_ascend_topology(self) -> None:
        if self.moe_config.num_experts != self.ascend_topology.upstream_global_num_experts:
            raise ValueError(
                "Upstream FusedMoEConfig expert count does not match Ascend topology: "
                f"got {self.moe_config.num_experts}, "
                f"expected {self.ascend_topology.upstream_global_num_experts}"
            )
        if self.global_num_experts != self.ascend_topology.upstream_global_num_experts:
            raise ValueError(
                "Upstream RoutedExperts expert count does not match Ascend topology: "
                f"got {self.global_num_experts}, "
                f"expected {self.ascend_topology.upstream_global_num_experts}"
            )

    def collect_eplb_load(self, fused_experts_results: FusedExpertsResult) -> None:
        if not self.dynamic_eplb or not _EXTRA_CTX.eplb_heat_collection_status:
            return
        expert_tokens = fused_experts_results.expert_tokens
        group_list_type = fused_experts_results.group_list_type
        assert expert_tokens is not None and group_list_type is not None, (
            "expert_tokens and group_list_type should not be None when dynamic_eplb is enabled."
        )
        local_load = (
            expert_tokens
            if group_list_type == 1
            else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
        )
        if self.multi_stage:
            assert self.load_counter is not None
            assert self.moe_load is not None
            cur_iter = torch.remainder(self.load_counter, self.num_iter)
            self.moe_load.index_add_(
                dim=0,
                index=cur_iter,
                source=local_load.to(torch.int32, non_blocking=True).view(1, -1),
            )
            self.load_counter.add_(1)
        else:
            assert self.moe_load is not None
            self.moe_load.add_(local_load)

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> FusedExpertsResult:
        del shared_experts, shared_experts_input
        assert not self.quant_method.is_monolithic
        fused_experts_results = self.quant_method.apply(
            layer=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=self.expert_map,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            pertoken_scale=pertoken_scale,
            mc2_mask=mc2_mask,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        self.collect_eplb_load(fused_experts_results)
        return fused_experts_results

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()
        if self.multi_stage:
            assert self.load_counter is not None
            self.load_counter.zero_()

    def update_expert_map(self, new_expert_map=None):
        if new_expert_map is None:
            return super().update_expert_map()
        self._expert_map = new_expert_map
        self.expert_map_manager._expert_map = new_expert_map
        return None

    @property
    def ep_rank(self):
        # NOTE(fused_moe refactor): upstream RoutedExperts exposes ep info only via
        # moe_config.moe_parallel_config, not as `ep_rank`/`ep_size`. Expose them here from
        # the Ascend topology so the runner can delegate eplb state to routed_experts (and
        # the EPLB adaptor can read layer.ep_rank).
        return self.ascend_topology.ep_rank

    @property
    def ep_size(self):
        return self.ascend_topology.ep_size


class AscendSharedExpertsExecutor:
    def __init__(
        self,
        runner: "AscendMoERunner",
        shared_experts: torch.nn.Module | None,
    ):
        self.runner = runner
        self.shared_experts = shared_experts

    def wrap_process_weights_after_loading(self) -> None:
        if self.shared_experts is None or not self.runner.multistream_overlap_shared_expert:
            return

        original_process_weights = self.runner._quant_method.process_weights_after_loading

        @wraps(original_process_weights)
        def wrapped_process_weights(*args, **kwargs):
            result = original_process_weights(*args, **kwargs)
            self._validate_shared_expert_consistency()
            return result

        self.runner._quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def prepare_flash_common3(self) -> None:
        if self.shared_experts is not None and self.runner.shared_multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self.shared_experts)

    def maybe_run_flash_common3_gate_stream(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        if self.shared_experts is None or not self.runner.shared_multistream_overlap_gate:
            return

        gate_stream = AscendMoEInstanceRegistry.get_gate_stream()
        gate_stream.wait_stream(torch.npu.current_stream())
        with npu_stream_switch(gate_stream, enabled=self.runner.multistream_overlap_gate):
            shared_out = self.shared_experts(hidden_states)
            shared_out = self._maybe_reduce_shared_output(shared_out)
            set_flash_common3_context(
                shared_out=shared_out,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )

    def wait_flash_common3_gate_stream(self) -> None:
        if self.runner.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(AscendMoEInstanceRegistry.get_gate_stream())

    def forward_shared_experts(
        self,
        hidden_states: torch.Tensor,
        fused_experts_results: FusedExpertsResult,
        events: FusedMoEEvents,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None

        if self.runner.shared_multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            assert fc3_context.shared_out is not None
            return fc3_context.shared_out

        return self._forward_shared_experts(
            hidden_states,
            FusedMoEEvents(
                before_routed_experts=events.before_routed_experts,
                after_routed_experts=events.after_routed_experts,
                before_dispatch=fused_experts_results.before_dispatch_evt,
                before_gmm2=fused_experts_results.before_gmm2_evt,
                before_combine=fused_experts_results.before_combine_evt,
                swiglu_limit=fused_experts_results.swiglu_limit,
            ),
        )

    def _validate_shared_expert_consistency(self) -> None:
        test_input = (
            torch.rand(
                10,
                self.runner.moe_config.hidden_dim,
                device="npu",
                dtype=self.runner.moe_config.in_dtype,
            )
            * 2
            - 1
        )

        assert self.shared_experts is not None
        integrated_out = self.shared_experts(test_input)
        part1_out = self._shared_experts_part1(test_input)
        split_out = self._shared_experts_part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.runner.moe_config.hidden_dim,
                self.runner.moe_config.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        assert self.shared_experts is not None
        shared_gate_up, _ = self.shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(
        self,
        hidden_states: torch.Tensor,
        shared_gate_up: torch.Tensor,
    ):
        assert self.shared_experts is not None
        shared_act = self.shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self.shared_experts.down_proj(shared_act)  # type: ignore

        if hasattr(self.shared_experts, "expert_gate") and self.shared_experts.expert_gate is not None:
            gate_out, _ = self.shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def _forward_shared_experts(
        self,
        hidden_states: torch.Tensor,
        fused_moe_evts: FusedMoEEvents,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(
            shared_experts_calculation_stream(),
            enabled=self.runner.multistream_overlap_shared_expert,
        ):
            has_quantized_shared = hasattr(self.shared_experts.gate_up_proj, "weight_scale") and hasattr(
                self.shared_experts.down_proj,
                "weight_scale",
            )
            if has_quantized_shared and self.runner.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

                maybe_wait_event(fused_moe_evts.after_routed_experts)
                hidden_states = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.shared_experts.gate_up_proj.weight,
                    self.shared_experts.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )

                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                    x=hidden_states,
                    weight_scale=self.shared_experts.gate_up_proj.weight_scale_fp32,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=None,
                    activate_left=True,
                    quant_mode=1,
                    swiglu_mode=1,
                    clamp_limit=fused_moe_evts.swiglu_limit,
                )

                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.shared_experts.down_proj.weight,
                    self.shared_experts.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared and self.runner.quant_type == QuantType.W4A8MXFP:
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states,
                    dst_type=torch.float8_e4m3fn,
                )

                maybe_wait_event(fused_moe_evts.before_dispatch)
                hidden_states = self.shared_experts.gate_up_proj((quantized_x, pertoken_scale))[0]

                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                    hidden_states,
                    topk_weight=None,
                    group_index=None,
                    dst_type=torch.float8_e4m3fn,
                    quant_mode=2,
                    clamp_value=fused_moe_evts.swiglu_limit,
                )

                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self.shared_experts.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                maybe_wait_event(fused_moe_evts.before_dispatch)
                part1_out = self._shared_experts_part1(hidden_states)
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self._shared_experts_part2(hidden_states, part1_out)

        if self.runner.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        return self._maybe_reduce_shared_output(shared_out)

    def _maybe_reduce_shared_output(self, shared_out: torch.Tensor) -> torch.Tensor:
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if (
            moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
            and not shared_expert_dp_enabled()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out


class AscendMoERunner(MoERunner):
    router_cls: type[AscendFusedMoERouter] = AscendFusedMoERouter
    shared_experts_executor_cls: type[AscendSharedExpertsExecutor] = AscendSharedExpertsExecutor
    router: AscendFusedMoERouter
    routed_experts: AscendRoutedExperts
    shared_experts_executor: AscendSharedExpertsExecutor
    enable_shared_expert_dp: bool
    enable_npugraph_ex_static_kernel: bool
    multistream_overlap_gate: bool
    multistream_overlap_shared_expert: bool
    shared_multistream_overlap_gate: bool
    quant_type: QuantType

    def __init__(
        self,
        *args,
        original_num_experts: int,
        original_routed_scaling_factor: float,
        original_activation: str,
        n_shared_experts: int,
        tid2eid=None,
        hash_enabled=None,
        hash_indices_table: torch.Tensor | None = None,
        zero_expert_type: str | None = None,
        ascend_topology: AscendMoETopology,
        **kwargs,
    ):
        shared_experts = kwargs.get("shared_experts")
        super().__init__(*args, **kwargs)
        base_router = self.router
        self.original_num_experts = original_num_experts
        self.original_routed_scaling_factor = original_routed_scaling_factor
        self.original_activation = original_activation
        self.n_shared_experts = n_shared_experts
        self.tid2eid = tid2eid
        self.hash_enabled = hash_enabled
        self.hash_indices_table = hash_indices_table
        self.zero_expert_type = zero_expert_type
        self.ascend_topology = ascend_topology
        self.router = self.router_cls(
            base_router=base_router,
            top_k=self.moe_config.experts_per_token,
            num_logical_experts=ascend_topology.placement_num_experts,
            use_grouped_topk=self.routed_experts.use_grouped_topk,
            renormalize=self.routed_experts.renormalize,
            topk_group=self.routed_experts.topk_group,
            num_expert_group=self.routed_experts.num_expert_group,
            custom_routing_function=self.routed_experts.custom_routing_function,
            scoring_func=self.routed_experts.scoring_func,
            routed_scaling_factor=original_routed_scaling_factor,
            e_score_correction_bias=self.routed_experts.e_score_correction_bias,
            tid2eid=tid2eid,
            hash_enabled=hash_enabled,
            hash_indices_table=hash_indices_table,
            zero_expert_type=zero_expert_type,
            mix_placement=ascend_topology.mix_placement,
            num_shared_experts=ascend_topology.n_shared_experts,
        )
        ascend_config = get_ascend_config()
        has_shared_experts = shared_experts is not None
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert and has_shared_experts
        self.shared_multistream_overlap_gate = ascend_config.multistream_overlap_gate and has_shared_experts
        self.quant_type = self.routed_experts.quant_type
        self.shared_experts_executor = self.shared_experts_executor_cls(
            runner=self,
            shared_experts=shared_experts,
        )
        self.shared_experts_executor.wrap_process_weights_after_loading()
        if self.multistream_overlap_gate:
            AscendMoEInstanceRegistry.get_gate_stream()
            logger.info_once("[fused_moe/layer] Multistream overlap gate is enabled.")
        if self.multistream_overlap_shared_expert:
            logger.info_once("[fused_moe/layer] Multistream overlap shared expert is enabled.")
        if enable_sp() and has_shared_experts:
            logger.info_once(
                "[fused_moe/layer] Sequence parallelism is enabled, shared experts are replicated for best performance."
            )

        VllmEplbAdaptor.register_layer(self)

    @property
    def use_dp_chunking(self) -> bool:
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    @property
    def _fused_output_is_reduced(self) -> bool:
        # For MC2/ALLTOALL/FUSED_MC2 comm types, finalize() already includes
        # TP all-reduce for the routed output, and _forward_shared_experts
        # handles it for the shared output. Signal this to the upstream
        # MoERunner.forward() so _maybe_reduce_final_output does not apply a
        # second TP all-reduce (which would double-count the contributions).
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and _EXTRA_CTX.flash_comm_v1_enabled)

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # _forward_shared_experts already handles shared expert TP all-reduce
        # for MC2/ALLTOALL/FUSED_MC2. For AllGather the reduction is done
        # via _maybe_reduce_final_output on the combined (shared + routed)
        # output. Skip any additional reduction here.
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int,
    ) -> torch.Tensor:
        states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
        return states[..., :trunc_size]

    def _maybe_add_zero_expert_output(self, result: torch.Tensor) -> torch.Tensor:
        zero_expert_output = getattr(self.router, "zero_expert_output", None)
        if zero_expert_output is not None:
            result = result + zero_expert_output
        return result

    @property
    def local_num_experts(self):
        return self.routed_experts.local_num_experts

    @property
    def ep_rank(self):
        return self.routed_experts.ep_rank

    @property
    def ep_size(self):
        return self.routed_experts.ep_size

    @property
    def global_expert_map(self):
        return self.routed_experts.global_expert_map

    @property
    def moe_load(self):
        return self.routed_experts.moe_load

    def get_log2phy_map(self):
        return self.routed_experts.get_log2phy_map()

    def clear_moe_load(self):
        return self.routed_experts.clear_moe_load()

    def update_expert_map(self, *args, **kwargs):
        return self.routed_experts.update_expert_map(*args, **kwargs)

    def _maybe_apply_internal_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate is None:
            return router_logits

        expected_router_experts = self.ascend_topology.placement_num_experts
        if router_logits.shape[-1] == expected_router_experts:
            return router_logits

        if self._fse_fuse_gate:
            self._maybe_fuse_gate_weights()
            assert self._combined_gate_weight is not None
            return F.linear(hidden_states, self._combined_gate_weight)

        gate_output = self.gate(hidden_states)
        return gate_output[0] if isinstance(gate_output, tuple) else gate_output

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.routed_experts._ensure_moe_quant_config_init()

        self.shared_experts_executor.prepare_flash_common3()
        before_routed_experts = torch.npu.current_stream().record_event()
        routed_router_logits = self._maybe_apply_internal_router_logits(
            hidden_states,
            router_logits,
        )
        after_routed_experts = torch.npu.current_stream().record_event() if self.is_internal_router else None
        shared_hidden_states = hidden_states if shared_experts_input is None else shared_experts_input

        with self._sequence_parallel_context():
            forward_context = get_forward_context()
            if self.enable_npugraph_ex_static_kernel and forward_context.all_moe_layers:
                forward_context.moe_layer_index %= len(forward_context.all_moe_layers)

            select_experts_fn = self.router.select_experts
            topk_weights, topk_ids = select_experts_fn(
                hidden_states=hidden_states,
                router_logits=routed_router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

            if isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    topk_weights,
                    True,
                    True,
                )
                topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    topk_ids,
                    True,
                    True,
                )

            self.shared_experts_executor.maybe_run_flash_common3_gate_stream(
                shared_hidden_states,
                topk_weights,
                topk_ids,
            )

            prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
                hidden_states=hidden_states,
                router_logits=routed_router_logits,
                replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
                enable_shared_expert_dp=self.enable_shared_expert_dp,
                quant_type=self.quant_type,
            )
            self.shared_experts_executor.wait_flash_common3_gate_stream()

            # The dispatch-based comm methods (MC2 / All2All) split hidden_states
            # across TP ranks inside prepare(); apply the identical pad/slice to the
            # topk tensors so x and topk_ids share dim0 at dispatch. AllGather already
            # aligned topk above via maybe_all_gather_and_maybe_unpad.
            if not isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                topk_weights, topk_ids = _EXTRA_CTX.moe_comm_method.pad_and_split_topk(
                    topk_weights,
                    topk_ids,
                )

            fused_experts_results = self.routed_experts.forward_modular(
                x=prepare_output.hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                pertoken_scale=prepare_output.pertoken_scale,
                mc2_mask=prepare_output.mc2_mask,
            )

            routed_out = _EXTRA_CTX.moe_comm_method.finalize(
                hidden_states=fused_experts_results.routed_out,
                reduce_results=isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl),
                padded_hidden_states_shape=prepare_output.padded_hidden_states_shape,
            )

            shared_out = self.shared_experts_executor.forward_shared_experts(
                shared_hidden_states,
                fused_experts_results,
                FusedMoEEvents(
                    before_routed_experts=before_routed_experts,
                    after_routed_experts=after_routed_experts,
                ),
            )
            if shared_out is None:
                return routed_out
            return shared_out, routed_out


_ORIGINAL_FUSED_MOE = None


def _select_ascend_moe_classes_for_current_device():
    if is_310p():
        from vllm_ascend._310p.fused_moe.fused_moe import (
            AscendMoERunner310,
            AscendRoutedExperts310,
        )

        return AscendMoERunner310, AscendRoutedExperts310

    return AscendMoERunner, AscendRoutedExperts


def _create_ascend_fused_moe_runner(*args, **kwargs):
    assert _ORIGINAL_FUSED_MOE is not None

    import inspect

    kwargs = dict(kwargs)
    hash_enabled = kwargs.pop("hash", None)
    tid2eid = kwargs.pop("tid2eid", None)

    signature = inspect.signature(_ORIGINAL_FUSED_MOE)
    parameter_names = list(signature.parameters)
    explicit_parameters = set(kwargs)
    explicit_parameters.update(parameter_names[: len(args)])
    bound_args = signature.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    factory_kwargs = dict(bound_args.arguments)

    runner_args = dict(factory_kwargs.pop("runner_args", {}) or {})
    routed_experts_args = dict(factory_kwargs.pop("routed_experts_args", {}) or {})
    original_num_experts = factory_kwargs["num_experts"]
    n_shared_experts = factory_kwargs.get("n_shared_experts") or 0

    ascend_topology = _plan_ascend_moe_topology_for_main(
        num_experts=original_num_experts,
        n_shared_experts=n_shared_experts,
        tp_size=factory_kwargs.get("tp_size"),
        dp_size=factory_kwargs.get("dp_size"),
        pcp_size=factory_kwargs.get("pcp_size"),
        is_sequence_parallel=factory_kwargs.get("is_sequence_parallel", False),
        enable_eplb=factory_kwargs.get("enable_eplb", False),
        num_redundant_experts=factory_kwargs.get("num_redundant_experts", 0),
        prefix=factory_kwargs.get("prefix", ""),
        num_redundant_experts_was_explicit="num_redundant_experts" in explicit_parameters,
    )
    factory_kwargs["num_experts"] = ascend_topology.upstream_num_experts
    factory_kwargs["num_redundant_experts"] = ascend_topology.upstream_num_redundant_experts
    factory_kwargs["enable_eplb"] = ascend_topology.enable_eplb
    runner_cls, routed_experts_cls = _select_ascend_moe_classes_for_current_device()

    common_ascend_args = {
        "original_num_experts": original_num_experts,
        "original_routed_scaling_factor": factory_kwargs.get("routed_scaling_factor", 1.0),
        "original_activation": factory_kwargs.get("activation", "silu"),
        "n_shared_experts": n_shared_experts,
        "tid2eid": tid2eid,
        "hash_enabled": hash_enabled,
        "hash_indices_table": factory_kwargs.get("hash_indices_table"),
        "ascend_topology": ascend_topology,
    }
    runner_args.update(common_ascend_args)
    routed_experts_args.update(common_ascend_args)
    # zero_expert_type is a routing concern consumed by the runner/router only;
    # upstream RoutedExperts.__init__ does not accept it, so keep it out of
    # routed_experts_args to avoid a TypeError on forward to super().__init__.
    runner_args["zero_expert_type"] = factory_kwargs.get("zero_expert_type")

    factory_kwargs["runner_cls"] = runner_cls
    factory_kwargs["routed_experts_cls"] = routed_experts_cls
    factory_kwargs["runner_args"] = runner_args
    factory_kwargs["routed_experts_args"] = routed_experts_args

    return _ORIGINAL_FUSED_MOE(**factory_kwargs)


def patch_fused_moe_factory(original_fused_moe=None) -> None:
    global _ORIGINAL_FUSED_MOE

    import vllm.model_executor.layers.fused_moe as fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    if original_fused_moe is not None:
        _ORIGINAL_FUSED_MOE = original_fused_moe
    elif _ORIGINAL_FUSED_MOE is None:
        _ORIGINAL_FUSED_MOE = fused_moe_layer.FusedMoE

    if fused_moe_layer.FusedMoE is _create_ascend_fused_moe_runner:
        return

    fused_moe_layer.FusedMoE = _create_ascend_fused_moe_runner
    fused_moe_pkg.FusedMoE = _create_ascend_fused_moe_runner
