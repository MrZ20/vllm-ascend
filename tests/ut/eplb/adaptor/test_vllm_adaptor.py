import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import DeepseekV2Config

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.utils import vllm_version_is


class TestVllmAdaptor(unittest.TestCase):
    def setUp(self):
        n_routed_experts = 256
        mock_model = MagicMock()
        mock_model.model.named_parameters.return_value = dict()
        config = DeepseekV2Config(n_routed_experts=n_routed_experts)
        mock_model.config = config
        mock_model.get_expert_map.return_value = [i for i in range(n_routed_experts)]
        mock_model.get_log2phy_map.return_value = [i for i in range(n_routed_experts)]
        del mock_model.language_model
        self.model = mock_model
        num_dense_layers = getattr(config, "first_k_dense_replace", 0)
        if vllm_version_is("0.22.1"):
            # vLLM PR #41184 has not landed in v0.22.1, so mlp.experts owns
            # quant_type/local expert state directly.
            moe_weight_owner = self.model.model.layers[num_dense_layers].mlp.experts
            last_moe_weight_owner = self.model.model.layers[-1].mlp.experts
        else:
            # vLLM PR #41184 moved the MoE weight owner under
            # MoERunner.routed_experts on target main. Mirror that shape in the
            # adaptor tests instead of relying on MagicMock's auto attributes.
            moe_weight_owner = self.model.model.layers[num_dense_layers].mlp.experts.routed_experts
            last_moe_weight_owner = self.model.model.layers[-1].mlp.experts.routed_experts
        moe_weight_owner.quant_type = QuantType.W8A8
        last_moe_weight_owner.local_num_experts = 1

        self.mock_rank = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_rank", return_value=0).start()
        self.mock_size = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_world_size", return_value=4).start()

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_fp16(self, mock_func):
        self.model.quant_config = None
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_w8a8(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_language_model_w8a8(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        model = MagicMock()
        model.language_model = self.model
        model.config.text_config = self.model.config
        VllmEplbAdaptor(model)

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()


if __name__ == "__main__":
    unittest.main()
