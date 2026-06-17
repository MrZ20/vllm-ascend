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
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.eplb import utils as eplb_utils
from vllm_ascend.eplb.utils import get_moe_weight_owner


class TestGetMoeWeightOwner(unittest.TestCase):
    """get_moe_weight_owner adapts to vLLM PR #41184's MoE weight ownership move.

    Both vLLM versions are exercised here by patching the (functools.cache-d)
    version check, so the dual-version behavior is covered regardless of which
    vLLM is installed in the test environment.
    """

    def test_legacy_returns_experts_itself(self):
        # vLLM PR #41184 has not landed in v0.22.1: mlp.experts is still the
        # legacy FusedMoE object that owns the MoE weights.
        experts = SimpleNamespace(routed_experts=object())
        with patch.object(eplb_utils, "vllm_version_is", return_value=True):
            self.assertIs(get_moe_weight_owner(experts), experts)

    def test_target_main_returns_routed_experts(self):
        # vLLM PR #41184 moves the MoE weights under MoERunner.routed_experts.
        routed_experts = object()
        experts = SimpleNamespace(routed_experts=routed_experts)
        with patch.object(eplb_utils, "vllm_version_is", return_value=False):
            self.assertIs(get_moe_weight_owner(experts), routed_experts)

    def test_target_main_missing_routed_experts_raises(self):
        # On target main the helper fails loudly (not with a deep AttributeError)
        # if upstream changes the layout away from #41184.
        experts = SimpleNamespace()
        with (
            patch.object(eplb_utils, "vllm_version_is", return_value=False),
            self.assertRaises(TypeError),
        ):
            get_moe_weight_owner(experts)


if __name__ == "__main__":
    unittest.main()
