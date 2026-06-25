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
from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.utils import vllm_version_is


def is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    if vllm_version_is("0.23.0"):
        return isinstance(layer, RoutedExperts)
    fused_moe_cls = FusedMoE
    return isinstance(fused_moe_cls, type) and isinstance(layer, fused_moe_cls)
