from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def test_base_moe_gating_top_k_uses_custom_op_when_available():
    x = torch.randn(2, 4)
    expected_weights = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    topk_ids = torch.tensor([[1, 3], [2, 0]], dtype=torch.int64)
    expected_out = torch.randn(2, 4)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch.ops._C_ascend.moe_gating_top_k",
            return_value=(expected_weights, topk_ids, expected_out),
            create=True,
        ) as custom_topk,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_moe_gating_top_k",
            create=True,
        ) as npu_topk,
    ):
        weights, ids, out = BaseDeviceAdaptor.moe_gating_top_k(
            x,
            k=2,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=1,
            norm_type=0,
            out_flag=False,
        )

    assert weights is expected_weights
    assert out is expected_out
    assert ids.dtype == torch.int32
    custom_topk.assert_called_once()
    npu_topk.assert_not_called()


def test_base_moe_gating_top_k_falls_back_to_torch_npu_when_custom_op_missing():
    x = torch.randn(2, 4)
    fallback_weights = torch.tensor([[2.0, 2.0], [1.0, 3.0]])
    topk_ids = torch.tensor([[1, 3], [2, 0]], dtype=torch.int64)
    expected_out = torch.randn(2, 4)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch.ops._C_ascend.moe_gating_top_k",
            side_effect=AttributeError("missing moe_gating_top_k"),
            create=True,
        ) as custom_topk,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_moe_gating_top_k",
            return_value=(fallback_weights, topk_ids, expected_out),
            create=True,
        ) as npu_topk,
    ):
        weights, ids, out = BaseDeviceAdaptor.moe_gating_top_k(
            x,
            k=2,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=1,
            norm_type=0,
            out_flag=False,
            routed_scaling_factor=0.5,
        )

    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5], [0.25, 0.75]]))
    assert ids.dtype == torch.int32
    assert out is expected_out
    custom_topk.assert_called_once()
    npu_topk.assert_called_once()
    assert npu_topk.call_args.kwargs["renorm"] == 0
    assert npu_topk.call_args.kwargs["routed_scaling_factor"] == 0.5


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
