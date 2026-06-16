[2026-06-16 12:49:01.885405][UC][W] Triton is installed, but `triton.backends` could not be imported. Disabling Triton. [5015,5015][importing.py:66,<module>]
[2026-06-16 12:49:01.885468][UC][I] Triton not installed or not compatible; certain GPU-related functions will not be available. [5015,5015][importing.py:81,<module>]
`Qwen2VLImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Qwen2VLImageProcessor` instead.
/usr/local/python3.12.13/lib/python3.12/site-packages/pytest_asyncio/plugin.py:247: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
[1m============================= test session starts ==============================[0m
platform linux -- Python 3.12.13, pytest-8.3.2, pluggy-1.6.0 -- /usr/local/python3.12.13/bin/python
cachedir: .pytest_cache
rootdir: /__w/vllm-ascend/vllm-ascend
configfile: pyproject.toml
plugins: cov-7.1.0, xdist-3.6.1, asyncio-1.3.0, mock-3.15.1, anyio-4.13.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
[1mcollecting ... [0m[2026-06-16 12:49:20.706326][UC][I] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader` [5015,5015][__init__.py:115,_wrapper]
collected 1272 items

tests/ut/_310p/attention/test_attention_mask_310.py::TestAttentionMaskBuilder310::test_get_attention_mask_310 [32mPASSED[0m
tests/ut/_310p/attention/test_attention_mask_310.py::TestAttentionMaskBuilder310::test_get_splitfuse_attn_mask_310 [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackend310::test_get_builder_cls [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackend310::test_get_impl_cls [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackend310::test_get_kv_cache_shape_not [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackendImpl310::test_forward_chunked_prefill_310 [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackendImpl310::test_forward_mtp_310 [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackendImpl310::test_forward_paged_attention_310 [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackendImpl310::test_forward_prefill_310 [32mPASSED[0m
tests/ut/_310p/attention/test_attention_v1_310.py::TestAscendAttentionBackendImpl310::test_forward_prefill_cache_hit_310 [32mPASSED[0m
tests/ut/_310p/fused_moe/test_experts_selector_310.py::TestExpertsSelector310::test_select_experts[256] [32mPASSED[0m
tests/ut/_310p/fused_moe/test_experts_selector_310.py::TestExpertsSelector310::test_select_experts[128] [32mPASSED[0m
tests/ut/_310p/fused_moe/test_moe_mlp_310.py::TestUnifiedApplyMLP310::test_all_gather_apply_mlp_returns_common_tuple_contract [32mPASSED[0m
tests/ut/_310p/fused_moe/test_moe_mlp_310.py::TestUnifiedApplyMLP310::test_unified_apply_mlp_with_quantization_310 [32mPASSED[0m
tests/ut/_310p/fused_moe/test_moe_mlp_310.py::TestUnifiedApplyMLP310::test_unified_apply_mlp_without_quantization_310 [32mPASSED[0m
tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::test_forward_shared_experts_without_gate_310 [31mFAILED[0m
tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::test_forward_shared_experts_with_gate_310 [31mFAILED[0m
tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::test_forward_impl_with_shared_experts_returns_tuple_310 [31mFAILED[0m
tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::test_forward_impl_without_shared_experts_integration_310 [31mFAILED[0m
tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::test_forward_impl_without_shared_experts_returns_routed_only_310 [31mFAILED[0m
tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::test_is_internal_router_is_false_310 [31mFAILED[0m
tests/ut/_310p/ops/test_chunk_gated_delta_rule_310.py::test_chunk_gated_delta_rule_310_output_shape_and_dtype [32mPASSED[0m
tests/ut/_310p/ops/test_chunk_gated_delta_rule_310.py::test_chunk_gated_delta_rule_310_varlen_path [32mPASSED[0m
tests/ut/_310p/ops/test_chunk_gated_delta_rule_310.py::test_chunk_gated_delta_rule_310_varlen_tnd_path [32mPASSED[0m
tests/ut/_310p/ops/test_conv_310.py::test_conv3d_310_forward_oot_uses_forward_native [32mPASSED[0m
tests/ut/_310p/ops/test_layernorm_310.py::test_rmsnorm_gated_310_forward_oot_uses_rmsnorm_activation_mul [32mPASSED[0m
tests/ut/_310p/ops/test_layernorm_310.py::test_rmsnorm_gated_310_forward_oot_uses_rmsnorm_without_gate [32mPASSED[0m
tests/ut/_310p/ops/test_layernorm_310.py::test_rmsnorm_gated_310_forward_oot_keeps_native_for_group_norm [32mPASSED[0m
tests/ut/_310p/ops/test_mm_encoder_attention_310.py::test_register_customop_overrides_mm_encoder_attention_for_310p [32mPASSED[0m
tests/ut/_310p/ops/test_mm_encoder_attention_310.py::test_mm_encoder_attention_310_forward_oot_with_padding [32mPASSED[0m
tests/ut/_310p/ops/test_rotary_embedding_310.py::test_set_mrope_apply_rotary_slices_populates_globals [32mPASSED[0m
tests/ut/_310p/ops/test_rotary_embedding_310.py::test_set_mrope_apply_rotary_slices_reuses_buffer_address [32mPASSED[0m
tests/ut/_310p/quantization/test_modelslim_config_310.py::TestAscendModelSlimConfig310::test_get_quant_method_for_fused_moe_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_modelslim_config_310.py::TestAscendModelSlimConfig310::test_get_quant_method_for_linear_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_modelslim_config_310.py::TestAscendModelSlimConfig310::test_get_quant_method_maps_lm_head_prefix_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_dynamic_310.py::TestAscendW8A8FusedMoEMethod310::test_get_dynamic_quant_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_dynamic_310.py::TestAscendW8A8FusedMoEMethod310::test_get_weight_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_dynamic_310.py::TestAscendW8A8DynamicLinearMethod310::test_apply_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_dynamic_310.py::TestAscendW8A8DynamicLinearMethod310::test_get_perchannel_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_dynamic_310.py::TestAscendW8A8DynamicLinearMethod310::test_get_weight_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_dynamic_310.py::TestAscendW8A8DynamicLinearMethod310::test_process_weights_after_loading_calls_nz_format_cast_310p [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_static_310.py::TestAscendW8A8LinearMethod310::test_apply_with_x_is_int8_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_static_310.py::TestAscendW8A8LinearMethod310::test_apply_with_x_not_int8_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_static_310.py::TestAscendW8A8LinearMethod310::test_get_perchannel_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_static_310.py::TestAscendW8A8LinearMethod310::test_get_pertensor_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_static_310.py::TestAscendW8A8LinearMethod310::test_get_weight_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8_static_310.py::TestAscendW8A8LinearMethod310::test_process_weights_after_loading_calls_nz_format_cast_310p [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8s_310.py::TestAscendW8A8SLinearMethod310::test_apply_with_x_is_int8_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8s_310.py::TestAscendW8A8SLinearMethod310::test_apply_with_x_not_int8_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8s_310.py::TestAscendW8A8SLinearMethod310::test_get_perchannel_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8s_310.py::TestAscendW8A8SLinearMethod310::test_get_pertensor_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8s_310.py::TestAscendW8A8SLinearMethod310::test_get_weight_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8sc_310.py::TestAscendW8A8SCLinearMethod310::test_apply_with_x_is_int8_310 [33mSKIPPED[0m
tests/ut/_310p/quantization/test_w8a8sc_310.py::TestAscendW8A8SCLinearMethod310::test_apply_with_x_not_int8_310 [33mSKIPPED[0m
tests/ut/_310p/quantization/test_w8a8sc_310.py::TestAscendW8A8SCLinearMethod310::test_get_perchannel_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8sc_310.py::TestAscendW8A8SCLinearMethod310::test_get_pertensor_param_310 [32mPASSED[0m
tests/ut/_310p/quantization/test_w8a8sc_310.py::TestAscendW8A8SCLinearMethod310::test_get_weight_310 [32mPASSED[0m
tests/ut/_310p/sample/test_sampler_310.py::TestSampler310pStandalone::test_random_sample_310p_fallback_to_initial_seed_when_set_state_failed [32mPASSED[0m
tests/ut/_310p/sample/test_sampler_310.py::TestSampler310pStandalone::test_random_sample_310p_rebuild_cache_when_generator_identity_changes [32mPASSED[0m
tests/ut/_310p/sample/test_sampler_310.py::TestSampler310pStandalone::test_random_sample_310p_reuse_cpu_generator_cache [32mPASSED[0m
tests/ut/_310p/test_block_table_310p.py::TestBlockTable310::test_compute_slot_mapping_rejects_device_tensor_inputs [32mPASSED[0m
tests/ut/_310p/test_block_table_310p.py::TestBlockTable310::test_compute_slot_mapping_with_query_start_loc_signature [32mPASSED[0m
tests/ut/_310p/test_block_table_310p.py::TestBlockTable310::test_compute_slot_mapping_with_req_indices_signature [32mPASSED[0m
tests/ut/_310p/test_block_table_310p.py::TestBlockTable310::test_multi_group_compute_slot_mapping_accepts_none_compressed_args [32mPASSED[0m
tests/ut/_310p/test_block_table_310p.py::TestBlockTable310::test_multi_group_compute_slot_mapping_uses_compressed_inputs_per_group [32mPASSED[0m
tests/ut/_310p/test_kv_block_zeroer_310p.py::TestAscendKVBlockZeroer310::test_init_meta_deduplicates_kv_pointers [32mPASSED[0m
tests/ut/_310p/test_kv_block_zeroer_310p.py::TestAscendKVBlockZeroer310::test_zero_block_ids_noop_when_empty [32mPASSED[0m
tests/ut/_310p/test_kv_block_zeroer_310p.py::TestAscendKVBlockZeroer310::test_zero_block_ids_zeros_target_slices [32mPASSED[0m
tests/ut/_310p/test_model_runner_310p.py::test_prepare_inputs_keeps_aclgraph_metadata_on_cpu [32mPASSED[0m
tests/ut/_310p/test_model_runner_310p.py::TestNPUModelRunner310::test_may_reinitialize_input_batch_expands_prefix_mamba_block_table [32mPASSED[0m
tests/ut/_310p/test_sharded_state_loader_310p.py::TestShardedStateLoader310::test_generate_quant_description_float_model_310 [32mPASSED[0m
tests/ut/_310p/test_sharded_state_loader_310p.py::TestShardedStateLoader310::test_generate_quant_description_int_model_310 [32mPASSED[0m
tests/ut/_310p/test_sharded_state_loader_310p.py::TestShardedStateLoader310::test_generate_quant_description_no_quant_config_310 [32mPASSED[0m
tests/ut/_310p/test_sharded_state_loader_310p.py::TestShardedStateLoader310::test_save_model_with_nd_format_310 [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_device_list_uses_all_visible_devices_when_env_unset [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_device_list_exits_when_env_unset_and_torch_query_fails Error: ASCEND_RT_VISIBLE_DEVICES is unset and failed to run torch.npu.device_count().
Details: query failed
[32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_device_list_parses_visible_devices [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_device_list_parses_single_id [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_print_config_block [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_load_first_apply_baseline_no_file [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_load_first_apply_baseline_malformed_json [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_load_first_apply_baseline_invalid_original_qos [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_load_first_apply_baseline_success [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_run_unset_exits_without_state_file [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_run_unset_parse_failed_bad_json_deletes_file [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_run_unset_parse_failed_invalid_structure_deletes_file [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_run_unset_restores_and_deletes_file [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_AiqosConfig_set_qos_captures_and_writes_state [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_AiqosConfig_second_apply_reuses_baseline_fewer_capture_qos [32mPASSED[0m
tests/ut/_tools/test_ai_qos_tool.py::test_AiqosConfig_merges_baseline_when_device_list_grows [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_block_scanner_parses_metadata_and_trims_raw_block [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_block_scanner_rejects_duplicate_block_names [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_block_scanner_rejects_unsupported_metadata [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_single_node_converter_uses_case_index_defaults_and_extra_args [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_single_node_converter_defaults_to_first_case_and_preserves_port [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_single_node_converter_reports_invalid_case_index [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_multi_node_converter_uses_host_index_and_token_list [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_multi_node_converter_requires_host_index [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_multi_node_converter_rejects_extra_positional_args [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_generator_service_writes_selected_artifact [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_cli_generates_block_to_stdout [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_cli_rejects_invalid_block_reference [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_substitute_template_positionals [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_external_dp_template_converter_maps_positionals [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_external_dp_template_converter_requires_host_index [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_external_dp_launch_converter_combines_all_nodes [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_external_dp_proxy_converter_expands_groups [32mPASSED[0m
tests/ut/_tools/test_docs_codegen.py::test_external_dp_proxy_converter_rejects_unsupported_routing [32mPASSED[0m
tests/ut/attention/test_attention_fa3.py::test_fa_custom_ops_tnd[data_type0-1-1-1-1024-1024-128-128-False] [33mSKIPPED[0m
tests/ut/attention/test_attention_fa3.py::test_fa_custom_ops_tnd[data_type1-5-4-1-1024-1024-128-128-True] [33mSKIPPED[0m
tests/ut/attention/test_attention_fa3.py::test_fa_custom_ops_tnd[data_type2-7-16-8-512-512-128-128-False] [33mSKIPPED[0m
tests/ut/attention/test_attention_mask.py::TestAttentionMaskBuilder::test_get_attn_mask [32mPASSED[0m
tests/ut/attention/test_attention_mask.py::TestAttentionMaskBuilder::test_get_splitfuse_attn_mask [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_cp_metadata [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_decode_only_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_prefill_compact_block_metadata [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_prefills_only [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_with_mlapo_enabled [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_with_prefills_and_decodes [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_build_with_speculative_and_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_compact_varlen_decode_slot_mapping_basic [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_compact_varlen_decode_slot_mapping_zero_tokens [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_init_default [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_init_no_cp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPMetadataBuilder::test_init_with_mlapo_enabled [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_align_to_graph_bucket_tokens_already_aligned [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_align_to_graph_bucket_tokens_no_forward_context [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_align_to_graph_bucket_tokens_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_align_to_graph_bucket_tokens_none_input [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_align_to_graph_bucket_tokens_pad_smaller [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_align_to_graph_bucket_tokens_truncate [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_exec_kv_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_exec_kv_with_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_indexer_select_ascend_op [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_indexer_select_torch_npu [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_sparse_flash_attention [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_sparse_flash_attention_process_decode_and_prefill_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_sparse_flash_attention_process_decode_and_prefill_with_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_sparse_flash_attention_process_decode_only [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_sparse_flash_attention_process_prefill_only_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_execute_sparse_flash_attention_process_prefill_with_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_gather_block_table [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_gather_kv_cross_cp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_gather_kv_cross_cp_compact [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_gather_kv_cross_cp_compact_no_cp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_gather_kv_cross_cp_no_cp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_get_full_kv_mlapo [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_get_full_kv_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_get_full_kv_with_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_indexer_select_post_process_decode_and_prefill_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_indexer_select_post_process_decode_and_prefill_with_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_indexer_select_post_process_decode_only_no_triton [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_indexer_select_post_process_decode_only_simple [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_indexer_select_post_process_prefill_only_no_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_indexer_select_post_process_prefill_with_pcp [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_init_default [32mPASSED[0m
tests/ut/attention/test_sfa_cp.py::TestAscendSFACPImpl::test_init_no_cp [32mPASSED[0m
tests/ut/compilation/test_npugraph_ex_utils_check.py::test_extra_stream_scope_check_logic [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkConfig::test_default_values [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkConfig::test_disabled_without_pp_ok [2026-06-16 12:49:22.831622][UC][I] Chunked prefill is enabled with max_num_batched_tokens=2048. [5015,5015][scheduler.py:247,__post_init__]
[32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkConfig::test_enabled_with_pp_ok [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkConfig::test_enabled_without_pp_raises [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkConfig::test_invalid_min_chunk_raises [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkConfig::test_invalid_smooth_factor_raises [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestChunkSizePredictor::test_fit_and_predict [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestChunkSizePredictor::test_fit_chunk_and_predict_with_history [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestChunkSizePredictor::test_predict_decreases_with_history [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestChunkSizePredictor::test_predict_not_ready_returns_none [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkManager::test_not_ready_before_profiling [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkManager::test_record_batch_refines_model [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkManager::test_run_profiling_all_fail [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkManager::test_run_profiling_success [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_run_profiling_chunk_init_none_executor [2026-06-16 12:49:22.846630][UC][I] Chunked prefill is enabled with max_num_batched_tokens=8192. [5015,5015][scheduler.py:247,__post_init__]
[32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_run_profiling_chunk_init_skips_second_call [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_run_profiling_chunk_init_success [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_schedule_chunked_prefill_running [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_schedule_new_requests [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_schedule_with_profiling_ready [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_scheduler_init [32mPASSED[0m
tests/ut/core/test_profiling_chunk.py::TestProfilingChunkScheduler::test_update_from_output [32mPASSED[0m
tests/ut/core/test_recompute_scheduler.py::test_pd_consumer_first_step_injects_placeholder_spec_tokens [32mPASSED[0m
tests/ut/device/test_device_op.py::test_npu_flash_attention_uses_fusion_attention_for_fp32 [32mPASSED[0m
tests/ut/device/test_device_op.py::test_npu_flash_attention_uses_unpad_attention_for_low_precision[dtype0] [32mPASSED[0m
tests/ut/device/test_device_op.py::test_npu_flash_attention_uses_unpad_attention_for_low_precision[dtype1] [32mPASSED[0m
tests/ut/device/test_device_op.py::test_a5_npu_flash_attention_uses_python_sequence_lengths [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_create_and_map_calls_python_create_and_map[handle0] [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_create_and_map_calls_python_create_and_map[handle1] [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_create_and_map_calls_python_create_and_map[handle2] [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_unmap_and_release_calls_python_unmap_and_release[handle0] [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_unmap_and_release_calls_python_unmap_and_release[handle1] [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_get_pluggable_allocator [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_singleton_behavior [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_python_malloc_and_free_callback [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_sleep_offload_and_discard [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_wake_up_loads_and_clears_cpu_backup [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_use_memory_pool_context_manager [32mPASSED[0m
tests/ut/device_allocator/test_camem.py::TestCaMem::test_get_current_usage [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_execute_command [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_execute_command_kills_timed_out_process [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_expand_cpu_list [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_get_all_logic_npus [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_get_all_logic_npus_filters_invalid_values [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_get_npu_map_info [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_get_running_npus [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_get_running_npus_filters_invalid_rows_and_visible_devices [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_get_running_npus_skips_non_pipe_rows_inside_process_section [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_parse_allowed_cpus_raises_when_field_missing [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_parse_allowed_cpus_returns_empty_when_status_file_missing [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_parse_topo_affinity [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestDeviceInfo::test_parse_topo_affinity_skips_affinity_header_and_non_npu_rows [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_allocate [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_average_distribute [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_bind_npu_irq_a3_uses_card_chip_mapping [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_bind_threads [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_binding_mode_table [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_cpu_node_map [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_cpu_pools_fallback_to_global_slice [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_cpu_pools_global_slice_mode [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_distributes_remainder_by_npu_id [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_fallback_to_affinity_len [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_fallback_to_running_len [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_raises_invalid_npu_id [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_raises_when_cpu_insufficient [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_returns_when_running_or_allowed_empty [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_splits_same_cpuset_across_processes [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_build_global_slice_cpu_pool_uses_total_logic_npus [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuAlloc::test_extend_numa [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_memory_executes_on_valid_numa_target [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_memory_skips_when_cpu_pool_or_numa_invalid [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_memory_skips_when_migratepages_missing [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_keeps_irqbalance_when_inactive [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_returns_when_current_npu_has_no_cpu_pool [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_returns_when_irq_path_not_writable [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_scans_multiple_interrupt_lines [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_skips_irqbalance_handling_when_service_absent [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_skips_when_cpu_pool_too_small [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_skips_when_pci_address_missing [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_skips_when_sq_irq_not_found [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_npu_irq_stops_irqbalance_and_writes_affinity_masks [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_raises_for_failed_taskset [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_skips_empty_cpu_list [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_threads_binds_main_acl_and_release_threads [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_bind_uses_sub_thread_flag [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_binding_mode_defaults_to_topo_affinity_for_unknown_device [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_build_cpu_node_map_skips_blank_and_header_rows [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_build_cpu_pools_raises_on_affinity_conflict [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_build_cpu_pools_topo_mode_builds_and_splits_duplicate_groups [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_build_cpu_pools_topo_mode_excludes_non_running_npu_from_final_pool [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_build_cpu_pools_topo_mode_skips_non_running_npu_without_cpuset_overlap [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_build_cpu_pools_topo_mode_splits_hidden_same_affinity_npus_across_processes [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_cpu_to_mask_handles_single_and_multi_group_masks [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_extend_numa_returns_original_list_when_multiple_nodes_present [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_get_threads_map_skips_irrelevant_lines [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_print_plan_handles_empty_release_assignment [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_print_plan_handles_non_empty_release_assignment [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestCpuBindingSupplemental::test_run_all_invokes_steps_in_order [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestBindingSwitch::test_bind_cpus_runs_allocator_on_arm [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestBindingSwitch::test_bind_cpus_skip_non_arm [32mPASSED[0m
tests/ut/device_allocator/test_cpu_binding.py::TestBindingSwitch::test_is_arm_cpu [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreKVEvents::test_add_and_get_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreKVEvents::test_aggregate [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreKVEvents::test_clear_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreKVEvents::test_get_number_of_workers [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreKVEvents::test_increment_workers [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreKVEvents::test_repr [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_get_kv_connector_kv_cache_events_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_get_kv_connector_kv_cache_events_with_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_init_scheduler_role [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_init_worker_role [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_save_kv_layer_consumer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_save_kv_layer_not_layerwise [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_scheduler_methods_delegate [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_take_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_update_connector_output_accumulate [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_update_connector_output_no_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_update_connector_output_with_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_wait_for_layer_load_not_layerwise [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_wait_for_save_consumer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_worker_methods [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestBackendABC::test_cannot_instantiate [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_from_file [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_from_file_defaults [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_from_file_ssd_offload [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_load_from_env [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_load_from_env_missing [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_ssd_offload_requires_absolute_path [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_ssd_offload_requires_path_in_json [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_ssd_setup_kwargs_off_when_disabled [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_ssd_setup_kwargs_raises_on_old_mooncake [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeStoreConfig::test_ssd_setup_kwargs_when_supported [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_b [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_empty_string [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_float_input [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_gb [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_int [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_invalid_format [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_kb [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_mb [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_no_unit [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestParseGlobalSegmentSize::test_unsupported_type [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestConvertToBytes::test_invalid_number [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestConvertToBytes::test_valid [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongConfig::test_load_from_env [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongConfig::test_load_from_env_defaults [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongConfig::test_load_from_env_missing [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_make_blob_lists [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_make_blob_lists_inner_length_mismatch [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_make_blob_lists_length_mismatch [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_make_blob_lists_no_device ERROR 06-16 12:49:23 [yuanrong_backend.py:73] Device id is not set. Check device initialization and configuration.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_normalize_keys_long_key [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_normalize_keys_short_valid [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongHelper::test_normalize_keys_with_invalid_chars [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_exists [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_get [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_get_error ERROR 06-16 12:49:23 [mooncake_backend.py:225] Failed to get key. keys=['k1'], result=[-1]. Check key existence and memory state.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_get_exception ERROR 06-16 12:49:23 [mooncake_backend.py:232] Failed to get key. keys=['k1'], type=Exception, error=fail. Check store state and network.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_put [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_put_error ERROR 06-16 12:49:23 [mooncake_backend.py:191] Failed to put key. keys=['k1'], result=[-1]. Check memory and store capacity.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_put_exception ERROR 06-16 12:49:23 [mooncake_backend.py:195] Failed to put key. keys=['k1'], type=Exception, error=fail. Check store state and memory.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendMethods::test_register_buffer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_ensure_device_ready [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_ensure_device_ready_already_set [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_exists [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_exists_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_exists_exception ERROR 06-16 12:49:23 [yuanrong_backend.py:147] Failed to check keys. keys_count=1, type=Exception, error=fail. Check network and yuanrong service.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_get [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_get_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_get_exception ERROR 06-16 12:49:23 [yuanrong_backend.py:181] Failed to get keys. keys_count=1, type=Exception, error=fail. Check network and yuanrong service.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_get_failed_keys ERROR 06-16 12:49:23 [yuanrong_backend.py:175] Failed to get keys. failed_count=1, sample_keys=['k1']. Check key existence and memory state.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_put [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_put_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_put_exception ERROR 06-16 12:49:23 [yuanrong_backend.py:205] Failed to put keys. keys_count=1, type=Exception, error=fail. Check network and yuanrong service.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestYuanrongBackendMethods::test_register_buffer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_exists [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_get [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_get_error ERROR 06-16 12:49:23 [memcache_backend.py:115] Failed to get key. keys=['k1'], result=[1]. Check key existence and memory state.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_get_exception ERROR 06-16 12:49:23 [memcache_backend.py:120] Failed to get key. keys=['k1'], type=Exception, error=fail. Check store state and network.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_put [32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_put_error ERROR 06-16 12:49:23 [memcache_backend.py:135] Failed to put key. keys=['k1'], result=[1]. Check memory and store capacity.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_put_exception ERROR 06-16 12:49:23 [memcache_backend.py:139] Failed to put key. keys=['k1'], type=Exception, error=fail. Check store state and memory.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_backend.py::TestMemcacheBackendMethods::test_register_buffer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestKeyMetadata::test_fields [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestPoolKey::test_hash_diff [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestPoolKey::test_hash_equal [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestPoolKey::test_split_layers [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestPoolKey::test_to_string [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestLayerPoolKey::test_hash [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestLayerPoolKey::test_to_string_contains_layer_id [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_decode_adaptor_prefill_pp_multi_partition [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_decode_adaptor_prefill_pp_no_partitions [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_decode_adaptor_prefill_pp_single_partition [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_get_block_hashes_rehashes_grouped_bytes_hashes [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_get_block_hashes_rehashes_grouped_str_hashes [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_make_key_by_hash [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_prepare_value [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_prepare_value_layer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_prepare_value_partial_block [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_process_tokens_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_process_tokens_rehashes_grouped_hashes [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_process_tokens_token_len_shorter_than_all_blocks [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_process_tokens_with_bytes_hashes [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_process_tokens_with_mask [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestChunkedTokenDatabase::test_process_tokens_with_str_hashes [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestLoadSpec::test_fields [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestLoadSpec::test_token_len_default [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestRequestTracker::test_from_new_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestRequestTracker::test_from_new_request_nested_block_ids [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestRequestTracker::test_update_invalid_type [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestRequestTracker::test_update_with_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestRequestTracker::test_update_with_list [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestRequestTracker::test_update_with_tuple [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_already_saved [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_basic_save [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_load_spec_cannot_load [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_no_discard [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_partial_tokens_discarded [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_skip_save [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_with_load_spec [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestReqMeta::test_from_request_tracker_with_original_block_size [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestAscendConnectorMetadata::test_add_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_config_data.py::TestLayerMultiBlockReqMeta::test_fields [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_add_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_get_and_clear_finished_requests [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_handle_request_base_noop [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_lookup_all_exist [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_lookup_exception ERROR 06-16 12:49:23 [kv_transfer.py:115] Remote connection failed in lookup. type=Exception, error=conn fail. Check network and remote store.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_lookup_partial [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferThread::test_update_and_get_kv_events [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_add_dec_delete_stored_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_dec_nonexistent_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_delete_nonexistent_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_all_exist_no_put [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_consumer_role [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_dcp_size_gt_1 [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_not_in_stored [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_puts_missing_keys [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_with_current_event [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreSendingThread::test_handle_request_with_kv_event [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreRecvingThread::test_handle_request ERROR 06-16 12:49:23 [kv_transfer.py:753] Failed to load blocks. failed_count=2, failed_blocks={0, 1}. Check block availability and memory state.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_handle_request_all_exist_last_chunk_final_layer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_handle_request_all_exist_not_last [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_handle_request_empty_keys [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_handle_request_last_chunk_final_layer_with_missing [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_handle_request_puts_missing [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_handle_request_with_current_event [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_layerwise_kv_event_not_published_before_final_layer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_layerwise_kv_event_published_on_final_layer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerSendingThread::test_layerwise_kv_event_uses_missing_blocks_from_previous_layers [32mPASSED[0m
tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreLayerRecvingThread::test_handle_request [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestGetZmqRpcPathLookup::test_default_port [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestGetZmqRpcPathLookup::test_lookup_rpc_port [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestGetZmqRpcPathLookup::test_mooncake_rpc_port_fallback [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_get_num_new_matched_tokens_all_hit [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_get_num_new_matched_tokens_async [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_get_num_new_matched_tokens_consumer_no_load [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_get_num_new_matched_tokens_hit [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_get_num_new_matched_tokens_less_than_computed [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_get_num_new_matched_tokens_too_short [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_request_finished_consumer_no_put [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_request_finished_empty_blocks [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_request_finished_no_tracker [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_request_finished_with_saved_tokens [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_update_state_after_alloc_no_load_spec [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_update_state_after_alloc_with_load [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolScheduler::test_update_state_after_alloc_zero_external [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerBuildMeta::test_build_connector_meta_consumer_skip_save [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerBuildMeta::test_build_connector_meta_finished_req [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerBuildMeta::test_build_connector_meta_new_req [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerBuildMeta::test_build_connector_meta_preempted [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClient::test_close [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClient::test_lookup [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_check_all_layers_exists_all_present [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_check_all_layers_exists_none [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_check_all_layers_exists_partial [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_find_max_hit_index_all_one [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_find_max_hit_index_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_find_max_hit_index_first_pos [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_find_max_hit_index_found [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_consumer_partition_config [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_get_and_clear_finished_requests [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_get_kv_events_empty [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_get_kv_events_with_send_thread [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_init_basic [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_init_kv_head_less_than_tp [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_init_mla [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_lookup_all_cached [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_lookup_exception ERROR 06-16 12:49:23 [pool_worker.py:915] Remote connection failed in get_common_prefix_length. type=Exception, error=conn error. Check network and remote store.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerInit::test_lookup_partial [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_get_and_clear_finished_req_still_running [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_get_and_clear_finished_requests_with_preempted [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_get_and_clear_finished_stored_req [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_get_finished_consumer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_get_finished_producer [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_lookup_layerwise [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_lookup_scheduler_all_cached [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_lookup_scheduler_exception ERROR 06-16 12:49:24 [pool_worker.py:1075] Remote connection failed in lookup. type=Exception, error=fail. Check network and remote store.
[32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_lookup_scheduler_layerwise [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_lookup_scheduler_multi_tp [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_lookup_scheduler_partial [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_register_kv_caches_non_mla [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_start_load_kv_no_load [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_start_load_kv_sync [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_wait_for_save [32mPASSED[0m
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerRegisterAndTransfer::test_wait_for_save_skip_non_save [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl.py::TestPyHcclCommunicator::test_load_hccl_fail ERROR 06-16 12:49:24 [pyhccl_wrapper.py:186] Failed to load HCCL library. so_file=/not/exist/path/libhccl.so, error=/not/exist/path/libhccl.so: cannot open shared object file: No such file or directory. The hccl library might not exist, be corrupted or it does not support the current platform Linux-5.10.0-182.0.0.95.r2673_211.hce2.x86_64-x86_64-with-glibc2.35. If you already have the library, please set the environment variable HCCL_SO_PATH to point to the correct hccl library path.
[32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl.py::TestPyHcclCommunicator::test_multi_gpu_pg_torch [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl.py::TestPyHcclCommunicator::test_stateless_group [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl.py::TestPyHcclCommunicator::test_world_size_1_return_early [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHcclUniqueId::test_construct [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHcclDataTypeEnum::test_torch_dtype_mapping [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHcclDataTypeEnum::test_unsupported_dtype_raises [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHcclRedOpTypeEnum::test_torch_reduce_op_mapping [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHcclRedOpTypeEnum::test_unsupported_op_raises [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestFunction::test_construct_with_valid_args [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hcclCommDestroy_success [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hccl_all_reduce [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hccl_broad_cast [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hccl_check [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hccl_comm_initRank [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hccl_get_error_string [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_hccl_get_uniqueId [32mPASSED[0m
tests/ut/distributed/device_communicators/test_pyhccl_wrapper.py::TestHCLLLibrary::test_init_with_nonexistent_so ERROR 06-16 12:49:24 [pyhccl_wrapper.py:186] Failed to load HCCL library. so_file=/definitely/not/exist/libhccl.so, error=/definitely/not/exist/libhccl.so: cannot open shared object file: No such file or directory. The hccl library might not exist, be corrupted or it does not support the current platform Linux-5.10.0-182.0.0.95.r2673_211.hce2.x86_64-x86_64-with-glibc2.35. If you already have the library, please set the environment variable HCCL_SO_PATH to point to the correct hccl library path.
[32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_all_blocks_fail [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_all_blocks_succeed [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_empty_lists [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_large_block_ids [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_logs_failed_blocks [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_mixed_error_codes [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_negative_return_codes [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_no_log_when_all_succeed [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_non_hybrid_single_block_semantics [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_partial_blocks_fail [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_single_block_fail [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocks::test_single_block_succeed [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocksEdgeCases::test_consecutive_failures [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocksEdgeCases::test_duplicate_block_ids_all_fail [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestRecordFailedBlocksEdgeCases::test_zero_block_id_with_failure [32mPASSED[0m
tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py::TestAscendStoreConnector::test_get_block_ids_with_load_errors_forwards_to_worker [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_b_unit [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_gb_unit [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_gb_unit_edge_cases [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_int_input [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_kb_unit [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_mb_unit [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_no_unit [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestParseGlobalSegmentSize::test_non_string_non_int_input [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestConvertToBytes::test_invalid_numbers [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_config_data.py::TestConvertToBytes::test_valid_conversion [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py::TestKVTransferMissingKeyPut::test_layer_sending_thread_only_puts_missing_keys [32mPASSED[0m
tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py::TestKVTransferMissingKeyPut::test_sending_thread_only_puts_missing_keys [32mPASSED[0m
tests/ut/distributed/test_communicator.py::TestNPUCommunicator::test_all_to_all_with_sizes [32mPASSED[0m
tests/ut/distributed/test_communicator.py::TestNPUCommunicator::test_all_to_all_without_sizes [32mPASSED[0m
tests/ut/distributed/test_parallel_state.py::test_init_ascend_model_parallel [32mPASSED[0m
tests/ut/eplb/adaptor/test_vllm_adaptor.py::TestVllmAdaptor::test_init_fp16 [32mPASSED[0m
tests/ut/eplb/adaptor/test_vllm_adaptor.py::TestVllmAdaptor::test_init_w8a8 [32mPASSED[0m
tests/ut/eplb/adaptor/test_vllm_adaptor.py::TestVllmAdaptor::test_language_model_w8a8 [32mPASSED[0m
tests/ut/eplb/core/policy/test_policy_factory.py::TestEplbRebalancePolicies::test_flashlb_rebalance_experts [32mPASSED[0m
tests/ut/eplb/core/policy/test_policy_factory.py::TestEplbRebalancePolicies::test_swift_balance_rebalance_experts [32mPASSED[0m
tests/ut/eplb/core/test_eplb_device_transfer_loader.py::test_generate_task_and_state_flow [32mPASSED[0m
tests/ut/eplb/core/test_eplb_device_transfer_loader.py::test_asyn_transfer_and_update [32mPASSED[0m
tests/ut/eplb/core/test_eplb_device_transfer_loader.py::test_set_log2phy_map [32mPASSED[0m
tests/ut/eplb/core/test_eplb_device_transfer_loader.py::test_invalid_state_asyn_update [32mPASSED[0m
tests/ut/eplb/test_eplb_updator.py::TestEplbUpdatorComputeAndSetMoeLoad::test_compute_and_set_moe_load_multi_stage [32mPASSED[0m
tests/ut/eplb/test_eplb_updator.py::TestEplbUpdatorComputeAndSetMoeLoad::test_compute_and_set_moe_load_normal [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheTaskTrackerInit::test_init_basic_properties [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestGetAndClearFinishedSingleRequests::test_concurrent_access [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestGetAndClearFinishedSingleRequests::test_empty_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestGetAndClearFinishedSingleRequests::test_multiple_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestGetAndClearFinishedSingleRequests::test_single_request [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheSendingThreadInit::test_ready_event_reference [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheSendingThreadInit::test_thread_daemon_property [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheSendingThreadInit::test_thread_name_format [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestGetAndClearFinishedRequests::test_get_and_clear_finished_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheSendingThread::test_reformat_kv_cache_hybrid_linear_uses_cache_block_size [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheSendingThread::test_run_handles_get_meta_and_done_recv_msgs [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheRecvingThreadBasic::test_add_request [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheRecvingThreadBasic::test_clear_failed_recv_request [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheRecvingThreadBasic::test_get_and_clear_invalid_block_ids [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheRecvingThreadBasic::test_get_finished_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheRecvingThreadBasic::test_mark_and_is_failed [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestSocketManagement::test_get_remote_socket [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestSocketManagement::test_return_socket_to_pool [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestCoreFunctionality::test_handle_request [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestCoreFunctionality::test_transfer_kv_cache [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestCoreFunctionality::test_transfer_kv_cache_failure ERROR 06-16 12:49:32 [mooncake_connector.py:780] Mooncake transfer failed for request. remote_request_id=req1, ret=-1. 
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestCoreFunctionality::test_transfer_prefix_cache_trims_remote_kernel_blocks [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestCoreFunctionality::test_transfer_prefix_cache_uses_computed_token_offset [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMetadataHandling::test_get_remote_metadata_failure [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMetadataHandling::test_get_remote_metadata_success [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMainThreadLoop::test_run_loop_normal [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheTaskTracker::test_duplicate_task_update [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheTaskTracker::test_retrieve_expired_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheTaskTracker::test_update_done_task_count [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestKVCacheTaskTracker::test_updtate_add_delayed_request [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorMetadata::test_add_new_req [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorSchedulerMatchedTokens::test_build_connector_meta [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorSchedulerMatchedTokens::test_get_num_new_matched_tokens [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestHelperFunctions::test_group_concurrent_contiguous [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestHelperFunctions::test_group_concurrent_contiguous_empty [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestHelperFunctions::test_string_to_int64_hash [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorForScheduler::test_scheduler_methods [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorForScheduler::test_scheduler_role [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnector::test_build_connector_meta [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnector::test_get_num_new_matched_tokens [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnector::test_request_finished [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnector::test_scheduler_initialization [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnector::test_update_state_after_alloc [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorScheduler::test_get_num_new_matched_tokens_no_remote_prefill [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorScheduler::test_get_num_new_matched_tokens_with_remote_prefill [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorScheduler::test_request_finished_no_remote_decode [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorScheduler::test_update_state_after_alloc_no_remote_prefill [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorScheduler::test_update_state_after_alloc_with_remote_prefill [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_ensure_zmq_recv_success [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_ensure_zmq_recv_timeout_and_fail [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_ensure_zmq_send_retry_and_fail [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_ensure_zmq_send_success [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_group_concurrent_contiguous [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_group_empty [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_string_to_int64_hash [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_zmq_ctx_invalid_type [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestUtils::test_zmq_ctx_ok [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_device_id_selection_with_physical_devices [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_get_kv_split_metadata [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_get_remote_tp_rank [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_get_tp_num_need_pulls [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_pd_disaggregated_hybrid_prefix_tp_and_pp_unequal [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_pd_disaggregated_hybrid_remote_pcp_splits_attention_and_final_mamba_state [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_pd_disaggregated_split_cross_covers_prefix_tp_cp_pp [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_register_kv_caches_consumer [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_register_kv_caches_mla_case [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_connector.py::TestMooncakeConnectorWorker::test_register_kv_caches_producer [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_hybrid_connector.py::TestMooncakeHybridConnectorScheduler::test_compute_transfer_block_ids_trims_swa_groups [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_hybrid_connector.py::TestMooncakeHybridConnectorScheduler::test_request_finished_trims_before_swa_clip [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_hybrid_connector.py::TestMooncakeHybridConnectorScheduler::test_request_finished_uses_num_prompt_tokens [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheSendingLayerThread::test_callback_invoked_on_final_layer [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheSendingLayerThread::test_transfer_pd_gt1_uses_buffers_and_calls_engine [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheSendingLayerThread::test_transfer_skips_when_no_local_blocks [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheRecvingLayerThread::test_get_and_clear_done_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheRecvingLayerThread::test_get_and_clear_failed_requests [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheRecvingLayerThread::test_run_loop_handles_meta_done_invalid_unexpected_and_ack [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheRecvingLayerThread::test_run_loop_pd_head_ratio_gt1_requires_multiple_done [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheRecvingLayerThread::test_update_done_task_aggregates_by_pd_head_ratio [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestKVCacheRecvingLayerThread::test_update_failed_task_aggregates_by_pd_head_ratio [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorMetadata::test_add_new_req [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorSchedulerMatchedTokens::test_build_connector_meta [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorSchedulerMatchedTokens::test_get_num_new_matched_tokens [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_build_connector_meta_accumulates_cached_blocks [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_build_connector_meta_consumes_reqs_need_recv_and_clears [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_build_connector_meta_emits_when_tokens_reach_total [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_get_num_new_matched_tokens_with_prefill_block_aligned [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_request_finished_returns_false_none [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_update_state_after_alloc_decode_records_send_layerwise [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorScheduler_More::test_update_state_after_alloc_prefill_records_and_resets_flag ERROR 06-16 12:49:34 [mooncake_layerwise_connector.py:1029] Failed to connect to metaserver. url=None, retry=1. 
[32mPASSED[0mERROR 06-16 12:49:34 [mooncake_layerwise_connector.py:1029] Failed to connect to metaserver. url=None, retry=2. 
ERROR 06-16 12:49:34 [mooncake_layerwise_connector.py:1029] Failed to connect to metaserver. url=None, retry=3. 

tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_ensure_zmq_recv_success ERROR 06-16 12:49:34 [mooncake_layerwise_connector.py:911] Access metaserver fail. error=Invalid type for url.  Expected str or httpx.URL, got <class 'NoneType'>: None. 
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_ensure_zmq_recv_timeout_and_fail [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_ensure_zmq_send_retry_and_fail [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_ensure_zmq_send_success [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_group_concurrent_contiguous [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_group_concurrent_contiguous_empty [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_string_to_int64_hash [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_zmq_ctx_invalid_type [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestHelperFunctions::test_zmq_ctx_ok [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorForScheduler::test_scheduler_methods [2026-06-16 12:49:34.550982][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorForScheduler::test_scheduler_role [2026-06-16 12:49:34.588443][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnector::test_build_connector_meta [2026-06-16 12:49:34.625794][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnector::test_get_num_new_matched_tokens [2026-06-16 12:49:34.663206][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnector::test_request_finished [2026-06-16 12:49:34.700574][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnector::test_scheduler_initialization [2026-06-16 12:49:34.737921][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnector::test_update_state_after_alloc [2026-06-16 12:49:34.775177][UC][W] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design. [5015,5015][base.py:190,__init__]
[32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorWorker::test_register_kv_caches_consumer [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorWorker::test_register_kv_caches_mla_case [32mPASSED[0m
tests/ut/kv_offload/test_mooncake_layerwise_connector.py::TestMooncakeLayerwiseConnectorWorker::test_register_kv_caches_producer [32mPASSED[0m
tests/ut/lora/test_lora.py::test_lora_placeholder [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_init_with_extra_config_file [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_init_with_extra_config [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_init_with_invalid_config [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_load_model_elastic_success [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_load_draft_model_elastic_success [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_load_draft_model_port_offset_and_group_name [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader.py::test_load_draft_model_skips_invalid_source_addresses [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_elastic_client_init [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_elastic_client_register [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_elastic_client_register_error_response ERROR 06-16 12:49:34 [elastic.py:83] Connect to 127.0.0.1:12345 fails, detail: Send data {'label': 'JOIN', 'content': {'device_id': 0, 'model_path': 'mocked_model_path', 'tp': 1, 'pp': 1, 'port': <MagicMock name='socket().__enter__().getsockname().__getitem__()' id='140121104564640'>, 'group_name': 'netloader'}} to server fails, detail: Object of type MagicMock is not JSON serializable
ERROR 06-16 12:49:34 [elastic.py:96] All sources exhausted, no connection established for device_id=0, model_path=mocked_model_path, sources=[127.0.0.1:12345]
[32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_elastic_client_register_exception ERROR 06-16 12:49:34 [elastic.py:83] Connect to 127.0.0.1:12345 fails, detail: Send data {'label': 'JOIN', 'content': {'device_id': 0, 'model_path': 'mocked_model_path', 'tp': 1, 'pp': 1, 'port': <MagicMock name='socket().getsockname().__getitem__()' id='140121112602112'>, 'group_name': 'netloader'}} to server fails, detail: Object of type MagicMock is not JSON serializable
ERROR 06-16 12:49:34 [elastic.py:96] All sources exhausted, no connection established for device_id=0, model_path=mocked_model_path, sources=[127.0.0.1:12345]
[32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_server_initialization [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_int8_cache_handling[dram-cpu] [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_int8_cache_handling[no-None] [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_int8_cache_handling[invalid-None] [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_client_handler_valid_join [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_client_handler_mismatch [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_client_handler_invalid_requests[invalid_data0-True] [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_client_handler_invalid_requests[invalid_data1-True] [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_client_handler_invalid_requests[plain text-False] ERROR 06-16 12:49:34 [elastic.py:342] Failed to load plain text as JSON string from ('192.168.1.1', 12345)
[32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_client_handler_invalid_requests[invalid_bytes-False] ERROR 06-16 12:49:34 [elastic.py:342] Failed to load invalid_bytes as JSON string from ('192.168.1.1', 12345)
[32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_server_start [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_server_cleanup [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_draft_group_name_in_client_register [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_elastic.py::test_draft_group_name_in_server_p2p_send [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_load.py::test_sources_this_device_empty [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_load.py::test_client_s_none ERROR 06-16 12:49:34 [elastic.py:96] All sources exhausted, no connection established for device_id=0, model_path=model_path, sources=[a, b]
[32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_load.py::test_client_ack_none ERROR 06-16 12:49:34 [elastic.py:96] All sources exhausted, no connection established for device_id=0, model_path=model_path, sources=[a, b]
[32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_load.py::test_model_load_fail [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_load.py::test_model_load_success [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_load.py::test_elastic_load_passes_draft_group_name [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_find_free_port [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_is_valid_path_prefix_empty [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_is_valid_path_prefixIllegal_characters [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_is_valid_path_prefixRelative_path [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_is_valid_path_prefixAbsolute_path [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_is_valid_path_prefix_no_directory [32mPASSED[0m
tests/ut/model_loader/netloader/test_netloader_utils.py::test_is_valid_path_prefix_directory_exists [32mPASSED[0m
tests/ut/ops/test_comm_utils.py::TestDistributedCommunication::test_async_all_to_all[input_tensor0-output_split_sizes0-input_split_sizes0] [32mPASSED[0m
tests/ut/ops/test_comm_utils.py::TestDistributedCommunication::test_async_all_to_all[input_tensor1-None-None] [32mPASSED[0m
tests/ut/ops/test_comm_utils.py::TestDistributedCommunication::test_gather_along_first_dim[1-test_tensor0-expected0] [32mPASSED[0m
tests/ut/ops/test_comm_utils.py::TestDistributedCommunication::test_gather_along_first_dim[4-test_tensor1-expected1] [32mPASSED[0m
tests/ut/ops/test_comm_utils.py::TestDistributedCommunication::test_gather_from_sequence_parallel_region[input_tensor0-None] [32mPASSED[0m
tests/ut/ops/test_comm_utils.py::TestDistributedCommunication::test_gather_from_sequence_parallel_region[input_tensor1-output_split_sizes1] [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_init [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_flashcomm2_oshard_enable_both_enabled [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_flashcomm2_oshard_enable_flashcomm2_disabled [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_flashcomm2_oshard_enable_o_shard_disabled [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_flashcomm2_oshard_enable_both_disabled [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_register_layer_hidden_layer [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_register_layer_non_hidden_layer [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_register_layer_default_prefetch_step [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_register_layer_overwrites_existing_layer_with_same_index [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_register_layer_missing_prefix_raises [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_register_layer_extract_layer_index_failure_propagates [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_get_layer_existing [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_get_layer_non_existing [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_get_layer_empty_dict [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_trigger_broadcast_for_layer_success [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_trigger_broadcast_for_layer_not_hidden [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_trigger_broadcast_for_layer_not_registered [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_trigger_broadcast_for_layer_not_registered_short_circuits_hidden_check [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_trigger_broadcast_for_layer_empty_manager [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_trigger_broadcast_for_layer_extract_layer_index_failure_propagates [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_post_process_after_loading_with_layers [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_post_process_after_loading_uses_first_registered_layer [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_post_process_after_loading_empty [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestFlashcomm2OShardManager::test_post_process_after_loading_single_layer [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestGlobalInstance::test_global_instance_exists [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestGlobalInstance::test_global_instance_has_shard_layers [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestIntegration::test_full_workflow [32mPASSED[0m
tests/ut/ops/test_flashcomm2_oshard_manager.py::TestIntegration::test_register_and_post_process_workflow [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::test_ascend_unquantized_skips_upstream_modular_kernel_init [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendUnquantizedFusedMoEMethod-UnquantizedFusedMoEMethod-__init__] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendUnquantizedFusedMoEMethod-UnquantizedFusedMoEMethod-process_weights_after_loading] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendUnquantizedFusedMoEMethod-UnquantizedFusedMoEMethod-apply] [33mSKIPPED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendMoERunner-MoERunner-__init__] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendMoERunner-MoERunner-forward_impl] [33mSKIPPED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendMoERunner-MoERunner-_forward_impl] [33mSKIPPED[0m
tests/ut/ops/test_fused_moe.py::TestVllmParentInterfaceCompatibility::test_overridden_method_signature_accepts_parent_interface[AscendRoutedExperts-RoutedExperts-__init__] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendUnquantizedFusedMoEMethod::test_process_weights_after_loading_transposes_and_formats[True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendUnquantizedFusedMoEMethod::test_process_weights_after_loading_transposes_and_formats[False] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendUnquantizedFusedMoEMethod::test_apply_builds_fused_experts_input[MoECommType.MC2] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendUnquantizedFusedMoEMethod::test_apply_builds_fused_experts_input[MoECommType.FUSED_MC2] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendUnquantizedFusedMoEMethod::test_apply_adds_zero_expert_result_and_force_balances [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_runner_reduction_properties[MoECommType.ALLTOALL-False-True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_runner_reduction_properties[MoECommType.MC2-False-True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_runner_reduction_properties[MoECommType.FUSED_MC2-False-True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_runner_reduction_properties[MoECommType.ALLGATHER-False-False] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_runner_reduction_properties[MoECommType.ALLGATHER-True-True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_forward_impl_delegates_to_routed_experts[False] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendMoERunner::test_forward_impl_delegates_to_routed_experts[True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoE::test_simple_helpers [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoE::test_forward_delegates_to_runner [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoE::test_forward_impl_prepare_apply_finalize[True] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoE::test_forward_impl_prepare_apply_finalize[False] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoE::test_forward_impl_dynamic_eplb_multi_stage [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoESharedExperts::test_properties_and_forward_delegate [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoESharedExperts::test_shared_experts_split_with_expert_gate [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoESharedExperts::test_shared_forward_impl_routes_shared_output[False] [32mPASSED[0m
tests/ut/ops/test_fused_moe.py::TestAscendFusedMoESharedExperts::test_shared_forward_impl_routes_shared_output[True] [32mPASSED[0m
tests/ut/ops/test_gate_linear.py::TestAscendGateLinear::test_forward_keeps_router_logits_fp32 [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_ascend_gdn_attention_uses_ascend_backend [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_sequence_index_buffers_cover_spec_decode_when_cudagraph_disabled [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_non_spec_prefill_fallback_meta_matches_original_inputs_and_runtime_helpers[pure_non_spec_prefill-0-None] [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_non_spec_prefill_fallback_meta_matches_original_inputs_and_runtime_helpers[mixed_spec_non_spec_with_padding-3-num_decode_draft_tokens_cpu1] [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_non_spec_prefill_fallback_meta_matches_original_inputs_and_runtime_helpers[mixed_prefill_decode_without_spec-0-None] [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_build_non_spec_causal_conv1d_host_meta_avoids_seq_lens_cpu_fallback [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_build_non_spec_causal_conv1d_host_meta_requires_has_initial_state [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_get_non_spec_causal_conv1d_host_args_falls_back_to_runtime_metadata [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_get_non_spec_causal_conv1d_host_args_requires_runtime_metadata [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_get_non_spec_chunked_prefill_meta_allows_missing_prefill_fallback_meta [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_builder_uses_device_chunk_builder_with_non_spec_query_start_loc [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_builder_skips_prebuilt_meta_without_non_spec_prefill[batch_spec0] [32mPASSED[0m
tests/ut/ops/test_gdn_attn_builder.py::test_builder_skips_prebuilt_meta_without_non_spec_prefill[batch_spec1] [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestDisposeTensor::test_dispose_tensor_replaces_with_empty [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestDisposeTensor::test_dispose_tensor_preserves_device_and_dtype [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestLayerMetadata::test_layer_metadata_creation [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestShardWindowMetadata::test_shard_window_metadata_creation [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_is_source_rank_zero [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_is_source_rank_one [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_post_process_after_loading_basic [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_post_process_after_loading_with_prefetch [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_post_process_after_loading_already_initialized [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_post_process_after_loading_empty_layers [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_reach_layer [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_wait_weight [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestSeriesMetadata::test_wait_weight_no_work [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestLayerExternalMetadata::test_layer_external_metadata_creation [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestCreateForwardWrapper::test_create_forward_wrapper_calls_wait_weight [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestCreateForwardWrapper::test_create_forward_wrapper_preserves_return_value [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestRegisterLayerToShardWeightSeries::test_register_layer_creates_new_series [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestRegisterLayerToShardWeightSeries::test_register_layer_adds_to_existing_series [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestRegisterLayerToShardWeightSeries::test_register_layer_disposes_weight_for_non_source [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestIsHiddenLayer::test_is_hidden_layer_true [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestIsHiddenLayer::test_is_hidden_layer_false [32mPASSED[0m
tests/ut/ops/test_layer_shard_linear.py::TestIsHiddenLayer::test_is_hidden_layer_boundary [32mPASSED[0m
tests/ut/ops/test_layernorm.py::test_RMSNorm_forward[None] [33mSKIPPED[0m (...)
tests/ut/ops/test_layernorm.py::test_RMSNorm_forward[residual1] [33mSKIPPED[0m
tests/ut/ops/test_layernorm.py::test_RMSNorm_forward_310p[None] [33mSKIPPED[0m
tests/ut/ops/test_layernorm.py::test_RMSNorm_forward_310p[residual1] [33mSKIPPED[0m
tests/ut/ops/test_linear.py::TestAscendUnquantizedLinearMethod::test_process_weights_after_loading_with_nz0 [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendUnquantizedLinearMethod::test_process_weights_after_loading_with_nz1 [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendUnquantizedLinearMethod::test_process_weights_after_loading_with_nz2 [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendRowParallelLinear::test_mlp_optimize [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendRowParallelLinear::test_oproj_tp [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendMergedColumnParallelLinear::test_merged_mlp_tp_init [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendReplicatedLinear::test_init_disable_tp [32mPASSED[0m
tests/ut/ops/test_linear.py::TestAscendReplicatedLinear::test_init_without_disable_tp [32mPASSED[0m
tests/ut/ops/test_mla.py::TestIndexerWrapper::test_forward [32mPASSED[0m
tests/ut/ops/test_mla.py::TestIndexerWrapper::test_initialization [32mPASSED[0m
tests/ut/ops/test_mla.py::TestAscendMultiHeadLatentAttention::test_forward [32mPASSED[0m
tests/ut/ops/test_mla.py::TestAscendMultiHeadLatentAttention::test_initialization [32mPASSED[0m
tests/ut/ops/test_moe_comm_method.py::TestMoECommMethod::test_all_gather_comm_impl [32mPASSED[0m
tests/ut/ops/test_moe_comm_method.py::TestMoECommMethod::test_alltoall_comm_impl [32mPASSED[0m
tests/ut/ops/test_moe_comm_method.py::TestMoECommMethod::test_fused_experts_method [32mPASSED[0m
tests/ut/ops/test_moe_comm_method.py::TestMoECommMethod::test_mc2_comm_impl [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestCumsumGroupList::test_cumsum_group_list_invalid_type_valueerror [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestCumsumGroupList::test_cumsum_group_list_supported_conversion [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestCumsumGroupList::test_cumsum_group_list_unsupported_conversion_notimplementederror [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestW4A8RuntimeFlags::test_w4a8_per_channel_gmm_swiglu_flag [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestUnifiedApplyMlpRequest::test_request_quant_path [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestUnifiedApplyMlpRequest::test_request_quant_path_passes_w4a8_per_channel_flag [32mPASSED[0m
tests/ut/ops/test_moe_mlp.py::TestUnifiedApplyMlpRequest::test_request_unquant_path [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_fused_experts_input_constructs_internal_mxfp_leaf_from_primitives [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_fused_experts_input_merges_dense_and_quant_weights [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_fused_experts_input_preserves_runtime_semantics [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_fused_experts_input_requires_primitive_mxfp_params_for_mxfp_quant [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_mlp_compute_input_derives_fusion_and_preserves_mxfp_params [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_token_dispatch_input_supports_remapped_topk_ids [32mPASSED[0m
tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_runtime_args_facade_exports_public_contracts_and_builders [32mPASSED[0m
tests/ut/ops/test_prepare_finalize.py::TestPrepareAndFinalize::test_all2all_prepare_finalize [32mPASSED[0m
tests/ut/ops/test_prepare_finalize.py::TestPrepareAndFinalize::test_all2all_tp_split_allgather [32mPASSED[0m
tests/ut/ops/test_prepare_finalize.py::TestPrepareAndFinalize::test_allgather_prepare_finalize [32mPASSED[0m
tests/ut/ops/test_prepare_finalize.py::TestPrepareAndFinalize::test_mc2_prepare_finalize [32mPASSED[0m
tests/ut/ops/test_prepare_finalize.py::TestPrepareAndFinalize::test_mc2_tp_split_allgather [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_basic_call_delegates_to_npu_op [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_neox_style_override_true [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_neox_style_override_false [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_neox_style_override_none_uses_self [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_gather_unpad_called_when_all_conditions_met [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_gather_unpad_skipped_unless_all_conditions_met[False-True-True] [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_gather_unpad_skipped_unless_all_conditions_met[True-False-True] [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_gather_unpad_skipped_unless_all_conditions_met[True-True-False] [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendEmbeddingForwardOOT::test_parent_init_signature_has_not_changed [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendYaRNRotaryEmbeddingForwardOOT::test_delegates_to_ascend_rotary_forward_oot [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendYaRNRotaryEmbeddingForwardOOT::test_return_value_passed_through [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendYaRNRotaryEmbeddingForwardOOT::test_is_neox_style_override_forwarded[True] [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendYaRNRotaryEmbeddingForwardOOT::test_is_neox_style_override_forwarded[False] [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendYaRNRotaryEmbeddingForwardOOT::test_all_args_forwarded_together [32mPASSED[0m
tests/ut/ops/test_rotary_embedding.py::TestAscendYaRNRotaryEmbeddingForwardOOT::test_parent_init_signature_has_not_changed [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestCustomVocabParallelEmbedding::test_forward_with_invalid_vocab [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestCustomVocabParallelEmbedding::test_forward_with_tp [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestCustomVocabParallelEmbedding::test_forward_with_tp_size_1 [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestCustomVocabParallelEmbedding::test_get_masked_input_and_mask [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestCustomVocabParallelEmbedding::test_output_shape [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestAscendLogitsProcessor::test_create_processor [32mPASSED[0m
tests/ut/ops/test_vocab_parallel_embedding.py::TestAscendLogitsProcessor::test_get_logits [32mPASSED[0m
tests/ut/patch/platform/test_patch_anthropic_system_message.py::test_inline_system_role_is_accepted_by_anthropic_requests [32mPASSED[0m
tests/ut/patch/platform/test_patch_anthropic_system_message.py::test_inline_system_string_is_merged_and_not_kept_as_chat_message [32mPASSED[0m
tests/ut/patch/platform/test_patch_anthropic_system_message.py::test_inline_system_list_content_is_merged_with_billing_header_stripped [32mPASSED[0m
tests/ut/patch/platform/test_patch_anthropic_system_message.py::test_multiple_inline_system_messages_are_all_merged [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_thinking.py::test_deepseek_v4_reasoning_effort_accepts_latest_values [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_thinking.py::test_reasoning_effort_enables_thinking_unless_user_overrides [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_thinking.py::test_deepseek_v4_tokenizer_maps_latest_reasoning_effort_values [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_streaming_deepseek_v4_tool_calls_emit_chunked_arguments [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_streaming_tool_call_metadata_only_first_chunk [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_streaming_wrapper_param_arguments_fragment [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_streaming_full_tool_call_single_chunk_drains_all_deltas [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_streaming_matches_non_streaming_conversion_fallbacks [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_composed_schema_conversion_in_streaming [32mPASSED[0m
tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py::test_registered_parser_is_patch_loaded [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm47_tool_call_parser.py::test_glm47_streaming_inline_zero_arg_tool_call_waits_until_complete [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm47_tool_call_parser.py::test_glm45_reasoning_glm47_streaming_inline_zero_arg_tool_call [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm_tool_call_streaming.py::test_remaining_args_delta_preserves_metadata_by_default [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm_tool_call_streaming.py::test_empty_remaining_args_delta_keeps_original_delta [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm_tool_call_streaming.py::test_remaining_args_delta_uses_explicit_fallback_metadata [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm_tool_call_streaming.py::test_terminal_argument_chunk_is_split_before_finish_chunk [32mPASSED[0m
tests/ut/patch/platform/test_patch_glm_tool_call_streaming.py::test_non_terminal_and_done_chunks_are_not_split [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_registered_parser_is_patch_loaded [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_plain_content_before_tool_call_is_preserved [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_streaming_emits_tool_name_before_argument_fragments [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_streaming_partial_arguments_before_invoke_closes [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_complete_single_chunk_still_reconstructs_tool_call [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_start_token_can_arrive_as_special_token_id [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_start_token_id_survives_empty_chunks_before_invoke_text [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_chat_tool_schema_drives_type_conversion [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_patch_does_not_require_private_v0202_schema_helpers [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py::test_responses_function_tool_schema_drives_type_conversion [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_count_reasoning_tokens[minimax-reasoning-before-end-token] [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_count_reasoning_tokens[append-think-reasoning-before-end-token] [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_count_reasoning_tokens[minimax-no-end-token-means-all-output-is-reasoning] [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_count_reasoning_tokens[append-think-no-end-token-means-all-output-is-reasoning] [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_count_reasoning_tokens[minimax-end-token-first-means-no-reasoning-tokens] [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_count_reasoning_tokens[append-think-end-token-first-means-no-reasoning-tokens] [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_update_usage_tracking_state_tracks_prompt_and_completion_tokens [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_make_usage_info_injects_reasoning_token_details [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_make_usage_info_injects_zero_cached_tokens [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_make_full_response_usage_sums_reasoning_tokens [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_stream_usage_details_are_injected_without_replacing_source [32mPASSED[0m
tests/ut/patch/platform/test_patch_minimax_usage_accounting.py::test_stream_usage_details_inject_prompt_details_without_reasoning [32mPASSED[0m
tests/ut/patch/platform/test_patch_tool_choice_none_content.py::test_parse_tool_calls_from_content_allows_named_tool_choice_with_none_content [33mSKIPPED[0m
tests/ut/patch/platform/test_patch_tool_choice_none_content.py::test_responses_parser_allows_named_tool_choice_with_none_content [32mPASSED[0m
tests/ut/patch/platform/test_patch_tool_choice_none_content.py::test_chat_completion_response_omits_empty_tool_calls_payload [32mPASSED[0m
tests/ut/patch/platform/test_patch_tool_choice_none_content.py::test_chat_completion_response_keeps_non_empty_tool_calls_payload [32mPASSED[0m
tests/ut/patch/platform/test_patch_tool_choice_none_content.py::test_chat_completion_stream_response_omits_empty_tool_calls_payload [32mPASSED[0m
tests/ut/patch/platform/test_patch_tool_choice_none_content.py::test_chat_completion_stream_response_keeps_non_empty_tool_calls_payload [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_resolve_kv_cache_block_sizes_with_cp_hybrid_groups[cp-without-prefix-caching] [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_resolve_kv_cache_block_sizes_with_cp_hybrid_groups[cp-with-prefix-caching] [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_get_effective_block_size[full-attention-scales-with-cp] [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_get_effective_block_size[mamba-keeps-physical-block-size-with-prefix-caching] [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_get_effective_block_size[full-attention-no-cp] [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_get_kv_cache_coordinator_delegates_single_group [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_get_kv_cache_coordinator_delegates_hybrid_without_caching [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_get_kv_cache_coordinator_uses_ascend_for_deepseek_v4 [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_deepseek_v4_detection_handles_non_mapping_nested_specs [32mPASSED[0m
tests/ut/patch/platform/test_prefix_cache_cp_patches.py::test_ascend_mamba_manager_uses_logical_block_size_with_prefix_caching [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_respects_rank_order_and_reuse_domain [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_mapping_hccl_config_affects_distinct_keys [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_accepts_realistic_options_object_defaults [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_accepts_matching_global_ranks_in_group [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_fails_closed_on_mismatched_global_ranks_in_group [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_ignores_runtime_populated_group_identity_fields [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_make_hccl_pg_key_fails_closed_for_unknown_mapping_fields [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_registry_release_only_destroys_real_pg_at_zero_refcount [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_release_of_non_group_member_only_drops_registry_entry [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_acquire_reuses_cached_handle_and_refcount [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_acquire_duplicate_non_group_member_handle_is_not_destroyed [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_clear_removes_entries_without_destroying_handles [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_release_non_group_member_uses_actual_sentinel [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_acquire_fails_closed_when_unknown_non_default_option_is_present [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_hccl_pg_registry.py::test_acquire_fails_closed_for_unknown_mapping_fields [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_group_coordinator_is_patched [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_same_hccl_group_reuses_device_pg_once [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_same_hccl_group_reuses_with_realistic_options_object [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_eplb_stays_isolated_from_ep_even_when_pg_options_match [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_mc2_stays_isolated_from_ep_even_when_pg_options_match [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_dynamic_eplb_stays_separate_from_ep_when_pg_options_differ [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_unknown_groups_share_by_default_when_ranks_and_options_match [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_hccl_pg_options_are_recreated_for_each_group_ranks_entry [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_destroy_releases_all_acquired_keys_in_reverse_order [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_failed_cpu_group_init_rolls_back_acquired_hccl_keys [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_failed_device_communicator_init_releases_all_keys_in_reverse_order [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_shared_hccl_group_is_destroyed_only_after_last_coordinator [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_destroy_distributed_environment_clears_registry_before_reinit [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_destroy_cleans_up_fail_closed_hccl_device_group [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_non_hccl_destroy_path_destroys_device_group_directly [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_all_to_all_returns_input_when_world_size_is_one [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_all_to_all_raises_assertion_on_invalid_scatter_dim [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_all_to_all_raises_assertion_on_invalid_gather_dim [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_all_to_all_calls_device_communicator_with_correct_args [32mPASSED[0m
tests/ut/patch/worker/patch_common/test_patch_distributed.py::test_all_to_all_calls_device_communicator_without_sizes [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_single_dp [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_multi_dp_naive_dispatch [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_multi_dp_modular_kernel [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_multi_dp_padded_all_gather [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_sp_modular_kernel_all2all [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_sp_modular_kernel_mc2 [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_unexpected_batch_dim [32mPASSED[0m
tests/ut/patch/worker/test_patch_routed_experts_capture.py::TestRoutedExpertsCapturerCapture::test_layer_id_out_of_bounds [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_create_profiler_config_enables_msmonitor_over_env [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_create_profiler_config_overrides_msmonitor_env [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_create_profiler_disabled [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_create_profiler_empty_dir [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_create_profiler_enabled [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_create_profiler_raises_when_msmonitor_env_enabled [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_init_creates_underlying_profiler [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_profiler_step_returns_true [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_start_stop_delegate_to_underlying_profiler [32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_step_calls_underlying_start_after_delay_iterations [2026-06-16 12:49:35.713422][UC][W] Using 'torch' profiler with delay_iterations or max_iterations while ignore_frontend is False may result in high overhead. [5015,5015][profiler.py:129,_validate_profiler_config]
[2026-06-16 12:49:35.713670][UC][I] GPU profiling will start 3 steps after start_profile. [5015,5015][wrapper.py:23,__init__]
[2026-06-16 12:49:35.714869][UC][I] Starting profiler after delay... [5015,5015][wrapper.py:96,step]
[32mPASSED[0m
tests/ut/profiler/test_torch_npu_profiler.py::TestTorchNPUProfilerWrapper::test_step_stops_underlying_profiler_after_max_iterations [2026-06-16 12:49:35.715933][UC][I] GPU profiling will stop after 1 worker steps, or when stop_profile is received. [5015,5015][wrapper.py:30,__init__]
[2026-06-16 12:49:35.716294][UC][I] Max profiling iterations reached. Stopping profiler... [5015,5015][wrapper.py:112,step]
[2026-06-16 12:49:35.716457][UC][I] Profiler stopped successfully. [5015,5015][wrapper.py:66,_call_stop]
[32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_shape_mismatch [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_single_element [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_single_element_int [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_tp_sharding_first_rank [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_tp_sharding_last_rank [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_tp_sharding_middle_rank [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestWeightLoader::test_weight_loader_with_different_dtypes [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendFAQuantAttentionMethodInit::test_init_with_full_config [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendFAQuantAttentionMethodInit::test_init_without_both_attributes [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendFAQuantAttentionMethodCreateWeights::test_create_weights_adds_submodules [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendFAQuantAttentionMethodCreateWeights::test_create_weights_creates_correct_tensors [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendFAQuantAttentionMethodCreateWeights::test_create_weights_registers_parameters [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendFAQuantAttentionMethodProcessWeights::test_process_weights_with_single_value_scale [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestIntegration::test_complete_workflow [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestC8KVScaleWeightLoader::test_dtype_preserved_as_param_dtype [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestC8KVScaleWeightLoader::test_shape_match_copies_value [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestC8KVScaleWeightLoader::test_shape_mismatch_resizes_param [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestC8KVScaleWeightLoader::test_squeeze_before_compare [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_apply_raises_runtime_error [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_create_weights_assigns_weight_loader [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_create_weights_does_not_set_int8_when_kv_producer [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_create_weights_initial_values [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_create_weights_registers_scale_offset_params [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_create_weights_sets_kv_cache_torch_dtype [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8KVCacheAttentionMethod::test_process_weights_after_loading_flattens [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8AttentionBackendImplScales::test_dequant_paged_kv_to_dense_round_trip [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8AttentionBackendImplScales::test_prepare_c8_scales_creates_bnsd_shape [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8AttentionBackendImplScales::test_prepare_c8_scales_idempotent [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8AttentionBackendImplScales::test_prepare_c8_scales_runs_once [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8AttentionBackendImplScales::test_quantize_kv_to_int8_formula [32mPASSED[0m
tests/ut/quantization/methods/test_kv_c8.py::TestAscendC8AttentionBackendImplScales::test_quantize_kv_to_int8_output_dtype [32mPASSED[0m
tests/ut/quantization/methods/test_moe_logical_experts.py::test_get_moe_num_logical_experts_uses_vllm_config_field [32mPASSED[0m
tests/ut/quantization/methods/test_moe_logical_experts.py::test_get_moe_num_logical_experts_falls_back_for_older_configs [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestRegisterScheme::test_register_scheme [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestRegisterScheme::test_register_scheme_duplicate_raises [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestGetSchemeClass::test_all_linear_schemes_subclass_ascend_linear_scheme [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestGetSchemeClass::test_all_moe_schemes_subclass_ascend_moe_scheme [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestGetSchemeClass::test_get_existing_scheme_class_existing_linear [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestGetSchemeClass::test_get_nonexistent_scheme_class [32mPASSED[0m
tests/ut/quantization/methods/test_registry.py::TestGetSchemeClass::test_registry_not_empty [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4LinearMethod::test_apply_3d_input [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4LinearMethod::test_get_pergroup_param_based_on_group_size [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4LinearMethod::test_get_weight_various_input_sizes [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4LinearMethod::test_process_weights_after_loading_transposes [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4MoEMethod::test_apply_full_params [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4MoEMethod::test_get_dynamic_quant_param_based_on_group_size [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4MoEMethod::test_get_weight_static_method [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4.py::TestAscendW4A4MXFP4MoEMethod::test_process_weights_transposes_weights [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestGetDecomposeDim::test_decomposition_product_equals_n [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestGetDecomposeDim::test_fallback_when_left_times_m_exceeds_max [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestGetDecomposeDim::test_non_square_decomposition [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestGetDecomposeDim::test_perfect_square_decomposition [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestGetDecomposeDim::test_raises_when_dim_sum_exceeds_max [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_apply [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_apply_dimension_mismatch_raises [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_apply_preserves_input_shape [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_get_pergroup_param [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_get_pertensor_param_non_row [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_get_pertensor_param_row [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_get_weight [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_get_weight_raises_on_odd_input [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_init_default [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_init_raises_on_oversized_tp [32mPASSED[0m
tests/ut/quantization/methods/test_w4a4_mxfp4_flatquant_dynamic.py::TestAscendW4A4MXFP4FlatQuantDynamicLinearMethod::test_process_weights_after_loading [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8LinearMethod::test_apply [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8LinearMethod::test_get_pergroup_param_group_size_variations [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8LinearMethod::test_get_weight_various_input_sizes [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8LinearMethod::test_process_weights_stores_original_shapes [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8LinearMethod::test_restore_after_process_returns_original_shape [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8MoEMethod::test_apply_full_params [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8MoEMethod::test_get_dynamic_quant_param_dtype_uint8 [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8MoEMethod::test_get_weight_various_expert_counts [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8MoEMethod::test_process_weights_stores_original_shapes [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_mxfp8.py::TestAscendW8A8MXFP8MoEMethod::test_restore_weights_for_rl_loading [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixLinearScheme::test_apply_uses_dynamic_for_non_kv_consumer [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixLinearScheme::test_apply_uses_static_for_kv_consumer [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixLinearScheme::test_get_perchannel_param_delegates_to_static [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixLinearScheme::test_get_pertensor_param_delegates_to_static [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixLinearScheme::test_get_weight_delegates_to_static [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixLinearScheme::test_process_weights_after_loading_sets_is_kv_consumer [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8_pdmix.py::TestAscendW8A8PDMixMoEScheme::test_get_dynamic_quant_param [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8DynamicLinearMethod::test_act_quant_type [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8DynamicLinearMethod::test_get_perchannel_param_dtype_variations [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8DynamicLinearMethod::test_get_weight_various_sizes [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8FusedMoEMethod::test_apply_uses_explicit_dispatch_and_mlp_args [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8FusedMoEMethod::test_get_weight_dtype_is_float8_e4m3fn [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8FusedMoEMethod::test_get_weight_various_expert_counts [32mPASSED[0m
tests/ut/quantization/methods/test_w8a8fp8_dynamic.py::TestAscendW8A8FP8FusedMoEMethod::test_quant_type_is_w8a8fp8 [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsQuanType::test_detect_unsupported_raises [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsQuanType::test_detect_w4a16 [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsQuanType::test_detect_w4a8_dynamic [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsQuanType::test_detect_w8a8_dynamic [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsQuanType::test_detect_w8a8_static [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsConfigGetQuantMethod::test_get_linear_quant_method [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsConfigGetQuantMethod::test_get_linear_unquantized_method [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsConfigGetQuantMethod::test_get_moe_quant_method [31mFAILED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsConfigGetQuantMethod::test_get_moe_unquantized_method [32mPASSED[0m
tests/ut/quantization/test_compressed_tensors_config.py::TestAscendCompressedTensorsConfigGetQuantMethod::test_no_quant_method [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendLinearMethod::test_apply_delegates_to_scheme [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendLinearMethod::test_create_weights [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendLinearMethod::test_process_weights_after_loading_delegates [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendKVCacheMethod::test_apply_delegates [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendKVCacheMethod::test_create_weights_delegates [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendKVCacheMethod::test_process_weights_after_loading_delegates [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendFusedMoEMethod::test_apply_method [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendFusedMoEMethod::test_create_weights_registers_parameters [32mPASSED[0m
tests/ut/quantization/test_method_adapters.py::TestAscendFusedMoEMethod::test_process_weights_after_loading_delegates [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_apply_extra_quant_adaptations_shared_head [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_apply_extra_quant_adaptations_weight_packed [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_from_config [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_config_filenames [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_min_capability ERROR 06-16 12:49:36 [modelslim_config.py:454] Ascend hardware does not support 'get_min_capability' feature.
[32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_name [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_quant_method_for_attention [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_quant_method_for_c8_kv_cache_attention [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_quant_method_for_fused_moe [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_quant_method_for_linear [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_get_supported_act_dtypes [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_init [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_init_with_default_config [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_is_layer_skipped_ascend [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_maybe_update_config_already_populated [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_maybe_update_config_loads_from_file [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_maybe_update_config_non_directory_raises ERROR 06-16 12:49:36 [modelslim_config.py:733] ModelSlim quantization config not found for model 'not_a_real_directory_path'. Searched path: not_a_real_directory_path. Found JSON files: N/A.
[32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_maybe_update_config_raises_when_file_missing ERROR 06-16 12:49:36 [modelslim_config.py:733] ModelSlim quantization config not found for model '/tmp/tmptpf8pjhj'. Searched path: /tmp/tmptpf8pjhj. Found JSON files: N/A.
[32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_maybe_update_config_raises_with_json_files_listed ERROR 06-16 12:49:36 [modelslim_config.py:733] ModelSlim quantization config not found for model '/tmp/tmpthhh1zej'. Searched path: /tmp/tmpthhh1zej. Found JSON files: ['config.json'].
[32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_override_quantization_method [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAscendModelSlimConfig::test_repr [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestApplyVllmMapper::test_apply_mapper_with_populated_quant_description [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestQuantPrefixMapper::test_lm_head_keeps_original_prefix_when_quant_key_exists [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestQuantPrefixMapper::test_lm_head_maps_to_language_model_lm_head_when_quant_key_exists [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestGetCacheScale::test_c8_kv_cache_type_k_proj_scale [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestGetCacheScale::test_no_match [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestGetKvQuantDtype::test_enable_fa_quant [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestGetKvQuantDtype::test_enable_fa_quant_false [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestGetKvQuantSplitFactor::test_enable_fa_quant_false [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestGetKvQuantSplitFactor::test_enable_fa_quant_true [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAddKvcacheQuantMetadata::test_with_fa_quant_type [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAddKvcacheQuantMetadata::test_with_indexer_quant_type [32mPASSED[0m
tests/ut/quantization/test_modelslim_config.py::TestAddKvcacheQuantMetadata::test_with_neither_quant_type [32mPASSED[0m
tests/ut/quantization/test_quant_parser.py::TestGetRollbackQuantType::test_returns_default_when_no_down_proj [32mPASSED[0m
tests/ut/quantization/test_quant_parser.py::TestGetRollbackQuantType::test_returns_down_proj_quant_type [32mPASSED[0m
tests/ut/quantization/test_quant_parser.py::TestParseMxfpQuantParams::test_default_values [32mPASSED[0m
tests/ut/quantization/test_quant_parser.py::TestParseQuantMoeDownProjParams::test_w4a4_mxfp4_respects_parsed_round_mode [32mPASSED[0m
tests/ut/quantization/test_quant_parser.py::TestParseQuantMoeDownProjParams::test_w8a8_mxfp8_uses_rint_round_mode [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_detects_compressed_tensors [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_detects_modelslim [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_modelslim_takes_priority_over_compressed_tensors [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_returns_none_for_config_without_quant_config [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_returns_none_for_malformed_config_json [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_returns_none_for_no_quant [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_returns_none_for_non_compressed_tensors_quant_method [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestDetectQuantizationMethod::test_returns_none_for_non_existent_path [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestMaybeAutoDetectQuantization::test_auto_detect_sets_quantization_and_logs_info [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestMaybeAutoDetectQuantization::test_no_detection_does_nothing [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestMaybeAutoDetectQuantization::test_no_detection_emits_info_log [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestMaybeAutoDetectQuantization::test_passes_revision_to_detect [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestMaybeAutoDetectQuantization::test_user_mismatch_logs_warning [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestMaybeAutoDetectQuantization::test_user_specified_same_method_no_change [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestEnableFaQuant::test_fa3_quantization_scenario [32mPASSED[0m
tests/ut/quantization/test_utils.py::TestEnableFaQuant::test_non_quantization_scenarios [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_expand_batch_to_tokens [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_expand_pytorch [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_reduce_sample_recovered_tokens_blockwise_pytorch [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_reduce_sample_recovered_tokens_blockwise_pytorch_ngram [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_reduce_sample_recovered_tokens_pytorch_ngram [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_rejection_greedy_sample_pytorch [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_rejection_random_reduce_sample_block_verify_pytorch [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_rejection_random_sample_block_verify_pytorch_standard [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_rejection_random_sample_pytorch [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_rejection_random_sample_pytorch_rejects_all_placeholder_mtp3 [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_rejection_random_sample_pytorch_rejects_placeholder [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_sample_recovered_tokens_pytorch_keeps_placeholder_distribution [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestAscendRejectionSampler::test_sample_recovered_tokens_pytorch_ngram [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestEntropyVerify::test_entropy_verify_block_verify [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestEntropyVerify::test_entropy_verify_block_verify_ngram [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestEntropyVerify::test_entropy_verify_ngram [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestEntropyVerify::test_entropy_verify_no_ori_probs_fallback [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestEntropyVerify::test_entropy_verify_standard_high_entropy_accepts_more [32mPASSED[0m
tests/ut/sample/test_rejection_sampler.py::TestEntropyVerify::test_entropy_verify_standard_low_entropy_stricter [32mPASSED[0m
tests/ut/sample/test_sampler.py::TestAscendSampler::test_init_with_raw_logprobs [32mPASSED[0m
tests/ut/spec_decode/test_extract_hidden_states_proposer.py::test_proposer_initialization [32mPASSED[0m
tests/ut/spec_decode/test_extract_hidden_states_proposer.py::test_dummy_run_basic [32mPASSED[0m
tests/ut/spec_decode/test_extract_hidden_states_proposer.py::test_prepare_next_token_ids_padded [32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_clear_ascend_config [2026-06-16 12:49:37.967760][UC][I] Asynchronous scheduling is enabled. [5015,5015][vllm.py:984,__post_init__]
[2026-06-16 12:49:37.968260][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[2026-06-16 12:49:37.968598][UC][I] Enabled custom fusions: norm_quant, act_quant [5015,5015][compilation.py:310,log_enabled_passes]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_enable_flashcomm1_config_overrides_disabled_env [2026-06-16 12:49:37.970620][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_enable_sp_falls_back_to_env_without_current_config [32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_flashcomm2_warning_uses_enable_flashcomm1_config [2026-06-16 12:49:37.974678][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_get_ascend_config [2026-06-16 12:49:37.976917][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_get_ascend_config_without_init [32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_dump_config_and_path_conflict [2026-06-16 12:49:37.979144][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_dump_config_type_validation [2026-06-16 12:49:37.980695][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_enable_npugraph_ex [2026-06-16 12:49:37.982317][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_recreates_for_new_vllm_config [2026-06-16 12:49:37.983864][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[2026-06-16 12:49:37.984508][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_with_additional_config [2026-06-16 12:49:37.986065][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_with_dump_config_materializes_fixed_file [2026-06-16 12:49:37.987631][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_init_ascend_config_without_additional_config [2026-06-16 12:49:37.990392][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_migrated_config_falls_back_to_envs [2026-06-16 12:49:37.992051][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_migrated_config_overrides_envs [2026-06-16 12:49:37.994768][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_config.py::TestAscendConfig::test_migrated_config_skips_default_env_fallback_logs [2026-06-16 12:49:37.997514][UC][I] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']) [5015,5015][kernel.py:272,set_platform_defaults]
[32mPASSED[0m
tests/ut/test_ascend_forward_context.py::test_ascend_forward_context_placeholder [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_override_envs_for_invariance [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_enable_batch_invariant_mode_ascendc_path [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_enable_batch_invariant_mode_triton_path [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_enable_batch_invariant_mode_no_backend [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_init_batch_invariance[True-True-info] [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_init_batch_invariance[True-False-warning] [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_init_batch_invariance[False-True-None] [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_init_batch_invariance[False-False-None] [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_add_rms_norm [32mPASSED[0m
tests/ut/test_batch_invariant.py::TestBatchInvariant::test_add_rms_norm_consistency [32mPASSED[0m
tests/ut/test_compressed_prefix_cache.py::test_compressed_prefix_cache_uses_logical_block_hash [32mPASSED[0m
tests/ut/test_compressed_prefix_cache.py::test_compressed_prefix_cache_hits_identical_logical_block [32mPASSED[0m
tests/ut/test_compressed_prefix_cache.py::test_hybrid_coordinator_rejects_partial_compressed_prefix_hit [32mPASSED[0m
tests/ut/test_cpu_binding.py::test_cpu_binding_placeholder [32mPASSED[0m
tests/ut/test_envs.py::TestEnvVariables::test_dir_and_getattr [32mPASSED[0m
tests/ut/test_envs.py::TestEnvVariables::test_env_vars_behavior [32mPASSED[0m
tests/ut/test_flash_common3_context.py::test_flash_common3_context_placeholder [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_ascend_colored_formatter_adds_prefix [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_ascend_formatter_adds_prefix_nested_file [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_ascend_formatter_adds_prefix_root_file [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_ascend_formatter_pass_through_vllm_logs [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_infer_module_name_edge_cases [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_infer_module_name_nested_file [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_infer_module_name_root_file [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_is_ascend_module_with_ascend_path [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_is_ascend_module_with_vllm_path [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_log_dir_constant [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_rotating_handler_rotates_on_size [32mPASSED[0m
tests/ut/test_logger.py::TestLoggerModule::test_setup_file_logging_creates_handler [32mPASSED[0m
tests/ut/test_meta_registration.py::test_meta_registration_placeholder [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_apply_config_platform_defaults_respects_explicit_max [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_apply_config_platform_defaults_respects_explicit_sizes [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_apply_config_platform_defaults_sets_ascend_default_max [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_apply_config_platform_defaults_skips_when_scheduler_max_num_seqs_is_missing [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_310p_no_custom_ops [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_balance_scheduler_rejects_pd_disaggregated_kv_consumer [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_balance_scheduler_rejects_pd_disaggregated_kv_producer [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_basic_config_update [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_cache_config_block_size [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_enforce_eager_mode [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_no_model_config_warning [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_preserves_platform_default_max_input [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_recompute_scheduler_rejects_pd_mixed_kv_both [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_recompute_scheduler_rejects_pd_mixed_no_kv_transfer [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_unsupported_compilation_level [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_unsupported_cudagraph_mode [33mSKIPPED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_check_and_update_config_v1_worker_class_selection [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_class_variables [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_attn_backend_cls_use_v1_and_mla [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_attn_backend_cls_use_v1_only [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_current_memory_usage_when_query_fails [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_current_memory_usage_when_reset_stats_fails [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_current_memory_usage_with_default_device [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_current_memory_usage_with_specific_device [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_device_capability [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_device_communicator_cls_returns_correct_value [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_device_name [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_device_total_memory_falls_back_on_permission_error [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_device_total_memory_prefers_npu_smi [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_device_uuid [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_punica_wrapper [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_get_static_graph_wrapper_cls_returns_correct_value [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_inference_mode [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_is_pin_memory_available_returns_true [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_is_sleep_mode_available [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_pre_register_and_update_with_existing_ascend_quant [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_pre_register_and_update_with_parser [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_pre_register_and_update_with_parser_no_quant_action [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_pre_register_and_update_without_parser [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_set_additional_forward_context_reads_v2_profile_override [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_set_additional_forward_context_v2_includes_required_moe_fields [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_update_block_size_for_backend_preserves_hybrid_block_size [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_update_block_size_for_backend_preserves_user_block_size [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_kv_load_failure_policy_rejects_hybrid_recompute [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_layer_sharding_config_accepts_kv_producer [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_layer_sharding_config_rejects_kv_both [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_layer_sharding_config_rejects_missing_kv_transfer_config [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_layer_sharding_config_rejects_non_kv_producer [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_parallel_config_accepts_dp_only [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_parallel_config_accepts_neither [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_parallel_config_accepts_pcp_only [32mPASSED[0m
tests/ut/test_platform.py::TestNPUPlatform::test_validate_parallel_config_rejects_pcp_plus_dp [32mPASSED[0m
tests/ut/test_profiling_config.py::test_profiling_config_placeholder [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_aligned_16 [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_current_stream [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_custom_op [33mSKIPPED[0m (Sk...)
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_layer_shard_accepts_kv_producer [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_layer_shard_rejects_kv_both [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_layer_shard_rejects_missing_kv_transfer [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_layer_shard_rejects_when_dsa_cp_disabled [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_o_proj_tp_accepts_kv_both [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_o_proj_tp_accepts_missing_kv_transfer [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_o_proj_tp_rejects_single_role_pd [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_enable_dsa_cp_with_o_proj_tp_rejects_when_dsa_cp_disabled [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_find_hccl_library [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_get_max_hidden_layers [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_maybe_trans_nz [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_nd_to_nz_2d [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_register_ascend_customop [32mPASSED[0m
tests/ut/test_utils.py::TestUtils::test_vllm_version_is [32mPASSED[0m
tests/ut/worker/test_dsv4_compressed_positions.py::test_compressed_positions_depend_on_corrected_num_computed_tokens [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_basic[1-1-5-query_lens0-2-False-100-False] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_basic[1-2-3-query_lens1-1-False-50-True] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_basic[2-1-4-query_lens2-2-False-100-True] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_basic[2-1-4-query_lens3-2-True-100-True] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_basic[2-1-3-query_lens4-3-False-50-True] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_basic[2-1-3-query_lens5-0-False-150-True] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_mla_tail_projection_indices[2-0-query_lens0] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_mla_tail_projection_indices[2-1-query_lens1] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_mla_tail_projection_indices[4-0-query_lens2] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_metadata_mla_tail_projection_indices[4-3-query_lens3] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_update_tokens_for_pcp_basic[tokens0-3-num_computed_tokens0-num_prompt_tokens0-4-0-expected_pcp_tokens0] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_update_tokens_for_pcp_basic[tokens1-3-num_computed_tokens1-num_prompt_tokens1-4-0-expected_pcp_tokens1] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_update_tokens_for_pcp_basic[tokens2-3-num_computed_tokens2-num_prompt_tokens2-4-0-expected_pcp_tokens2] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_update_tokens_for_pcp_basic[tokens3-1-num_computed_tokens3-num_prompt_tokens3-4-0-expected_pcp_tokens3] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens0-1-1-1-target0] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens1-2-1-1-target1] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens2-1-2-1-target2] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens3-2-2-1-target3] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens4-2-1-2-target4] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens5-2-1-128-target5] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_get_cp_local_seq_lens[seq_lens6-2-2-128-target6] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_mtp_input[req_ids0-num_computed_tokens0-token_ids_tensor_list0-1-6-num_scheduled_tokens0-target_input_ids_pcp_full0-target_query_start_loc_pcp_full0] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_mtp_input[req_ids1-num_computed_tokens1-token_ids_tensor_list1-1-2-num_scheduled_tokens1-target_input_ids_pcp_full1-target_query_start_loc_pcp_full1] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_mtp_input[req_ids2-num_computed_tokens2-token_ids_tensor_list2-2-12-num_scheduled_tokens2-target_input_ids_pcp_full2-target_query_start_loc_pcp_full2] [32mPASSED[0m
tests/ut/worker/test_pcp_manager.py::test_generate_pcp_mtp_input[req_ids3-num_computed_tokens3-token_ids_tensor_list3-4-19-num_scheduled_tokens3-target_input_ids_pcp_full3-target_query_start_loc_pcp_full3] [32mPASSED[0m
tests/ut/xlite/test_xlite.py::test_xlite_placeholder [32mPASSED[0m

=================================== FAILURES ===================================
[31m[1m_________________ test_forward_shared_experts_without_gate_310 _________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_forward_shared_experts_without_gate_310[39;49;00m():[90m[39;49;00m
>       layer = _build_layer(_DummySharedExperts(with_gate=[94mFalse[39;49;00m))[90m[39;49;00m

[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:60: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:51: in _build_layer
    [0mlayer = AscendFusedMoE310.[92m__new__[39;49;00m(AscendFusedMoE310)[90m[39;49;00m
[1m[31mvllm_ascend/_310p/fused_moe/fused_moe.py[0m:154: in __new__
    [0m[94mreturn[39;49;00m _create_ascend_fused_moe_runner(*args, **kwargs)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = ()
kwargs = {'routed_experts_args': {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ......fused_moe.fused_moe.AscendRoutedExperts'>, 'runner_cls': <class 'vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner'>}
hash_enabled = None, tid2eid = None
routed_experts_args = {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ...}

    [0m[94mdef[39;49;00m[90m [39;49;00m[92m_create_ascend_fused_moe_runner[39;49;00m(*args, **kwargs):[90m[39;49;00m
        [90m# Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the[39;49;00m[90m[39;49;00m
        [90m# supported extension point. Pass Ascend implementations through that[39;49;00m[90m[39;49;00m
        [90m# interface instead of relying on the old FusedMoE subclass replacement.[39;49;00m[90m[39;49;00m
        kwargs = [96mdict[39;49;00m(kwargs)[90m[39;49;00m
        hash_enabled = kwargs.pop([33m"[39;49;00m[33mhash[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        tid2eid = kwargs.pop([33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        routed_experts_args = [96mdict[39;49;00m(kwargs.pop([33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m, {}) [95mor[39;49;00m {})[90m[39;49;00m
        routed_experts_args.update([90m[39;49;00m
            {[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_num_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mnum_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_routed_scaling_factor[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_scaling_factor[39;49;00m[33m"[39;49;00m, [94m1.0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_activation[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mactivation[39;49;00m[33m"[39;49;00m, [33m"[39;49;00m[33msilu[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m, [94m0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m: tid2eid,[90m[39;49;00m
                [33m"[39;49;00m[33mhash_enabled[39;49;00m[33m"[39;49;00m: hash_enabled,[90m[39;49;00m
            }[90m[39;49;00m
        )[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrunner_cls[39;49;00m[33m"[39;49;00m, AscendMoERunner)[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrouted_experts_cls[39;49;00m[33m"[39;49;00m, AscendRoutedExperts)[90m[39;49;00m
        kwargs[[33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m] = routed_experts_args[90m[39;49;00m
>       [94mreturn[39;49;00m FusedMoE(*args, **kwargs)[90m[39;49;00m
[1m[31mE       TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'[0m

[1m[31mvllm_ascend/ops/fused_moe/fused_moe.py[0m:1290: TypeError
[31m[1m__________________ test_forward_shared_experts_with_gate_310 ___________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_forward_shared_experts_with_gate_310[39;49;00m():[90m[39;49;00m
>       layer = _build_layer(_DummySharedExperts(with_gate=[94mTrue[39;49;00m))[90m[39;49;00m

[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:68: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:51: in _build_layer
    [0mlayer = AscendFusedMoE310.[92m__new__[39;49;00m(AscendFusedMoE310)[90m[39;49;00m
[1m[31mvllm_ascend/_310p/fused_moe/fused_moe.py[0m:154: in __new__
    [0m[94mreturn[39;49;00m _create_ascend_fused_moe_runner(*args, **kwargs)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = ()
kwargs = {'routed_experts_args': {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ......fused_moe.fused_moe.AscendRoutedExperts'>, 'runner_cls': <class 'vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner'>}
hash_enabled = None, tid2eid = None
routed_experts_args = {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ...}

    [0m[94mdef[39;49;00m[90m [39;49;00m[92m_create_ascend_fused_moe_runner[39;49;00m(*args, **kwargs):[90m[39;49;00m
        [90m# Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the[39;49;00m[90m[39;49;00m
        [90m# supported extension point. Pass Ascend implementations through that[39;49;00m[90m[39;49;00m
        [90m# interface instead of relying on the old FusedMoE subclass replacement.[39;49;00m[90m[39;49;00m
        kwargs = [96mdict[39;49;00m(kwargs)[90m[39;49;00m
        hash_enabled = kwargs.pop([33m"[39;49;00m[33mhash[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        tid2eid = kwargs.pop([33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        routed_experts_args = [96mdict[39;49;00m(kwargs.pop([33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m, {}) [95mor[39;49;00m {})[90m[39;49;00m
        routed_experts_args.update([90m[39;49;00m
            {[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_num_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mnum_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_routed_scaling_factor[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_scaling_factor[39;49;00m[33m"[39;49;00m, [94m1.0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_activation[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mactivation[39;49;00m[33m"[39;49;00m, [33m"[39;49;00m[33msilu[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m, [94m0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m: tid2eid,[90m[39;49;00m
                [33m"[39;49;00m[33mhash_enabled[39;49;00m[33m"[39;49;00m: hash_enabled,[90m[39;49;00m
            }[90m[39;49;00m
        )[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrunner_cls[39;49;00m[33m"[39;49;00m, AscendMoERunner)[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrouted_experts_cls[39;49;00m[33m"[39;49;00m, AscendRoutedExperts)[90m[39;49;00m
        kwargs[[33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m] = routed_experts_args[90m[39;49;00m
>       [94mreturn[39;49;00m FusedMoE(*args, **kwargs)[90m[39;49;00m
[1m[31mE       TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'[0m

[1m[31mvllm_ascend/ops/fused_moe/fused_moe.py[0m:1290: TypeError
[31m[1m___________ test_forward_impl_with_shared_experts_returns_tuple_310 ____________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_forward_impl_with_shared_experts_returns_tuple_310[39;49;00m():[90m[39;49;00m
>       layer = _build_layer(_DummySharedExperts(with_gate=[94mTrue[39;49;00m))[90m[39;49;00m

[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:76: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:51: in _build_layer
    [0mlayer = AscendFusedMoE310.[92m__new__[39;49;00m(AscendFusedMoE310)[90m[39;49;00m
[1m[31mvllm_ascend/_310p/fused_moe/fused_moe.py[0m:154: in __new__
    [0m[94mreturn[39;49;00m _create_ascend_fused_moe_runner(*args, **kwargs)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = ()
kwargs = {'routed_experts_args': {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ......fused_moe.fused_moe.AscendRoutedExperts'>, 'runner_cls': <class 'vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner'>}
hash_enabled = None, tid2eid = None
routed_experts_args = {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ...}

    [0m[94mdef[39;49;00m[90m [39;49;00m[92m_create_ascend_fused_moe_runner[39;49;00m(*args, **kwargs):[90m[39;49;00m
        [90m# Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the[39;49;00m[90m[39;49;00m
        [90m# supported extension point. Pass Ascend implementations through that[39;49;00m[90m[39;49;00m
        [90m# interface instead of relying on the old FusedMoE subclass replacement.[39;49;00m[90m[39;49;00m
        kwargs = [96mdict[39;49;00m(kwargs)[90m[39;49;00m
        hash_enabled = kwargs.pop([33m"[39;49;00m[33mhash[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        tid2eid = kwargs.pop([33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        routed_experts_args = [96mdict[39;49;00m(kwargs.pop([33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m, {}) [95mor[39;49;00m {})[90m[39;49;00m
        routed_experts_args.update([90m[39;49;00m
            {[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_num_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mnum_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_routed_scaling_factor[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_scaling_factor[39;49;00m[33m"[39;49;00m, [94m1.0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_activation[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mactivation[39;49;00m[33m"[39;49;00m, [33m"[39;49;00m[33msilu[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m, [94m0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m: tid2eid,[90m[39;49;00m
                [33m"[39;49;00m[33mhash_enabled[39;49;00m[33m"[39;49;00m: hash_enabled,[90m[39;49;00m
            }[90m[39;49;00m
        )[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrunner_cls[39;49;00m[33m"[39;49;00m, AscendMoERunner)[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrouted_experts_cls[39;49;00m[33m"[39;49;00m, AscendRoutedExperts)[90m[39;49;00m
        kwargs[[33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m] = routed_experts_args[90m[39;49;00m
>       [94mreturn[39;49;00m FusedMoE(*args, **kwargs)[90m[39;49;00m
[1m[31mE       TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'[0m

[1m[31mvllm_ascend/ops/fused_moe/fused_moe.py[0m:1290: TypeError
[31m[1m___________ test_forward_impl_without_shared_experts_integration_310 ___________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_forward_impl_without_shared_experts_integration_310[39;49;00m():[90m[39;49;00m
>       layer = _build_layer([94mNone[39;49;00m)[90m[39;49;00m

[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:90: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:51: in _build_layer
    [0mlayer = AscendFusedMoE310.[92m__new__[39;49;00m(AscendFusedMoE310)[90m[39;49;00m
[1m[31mvllm_ascend/_310p/fused_moe/fused_moe.py[0m:154: in __new__
    [0m[94mreturn[39;49;00m _create_ascend_fused_moe_runner(*args, **kwargs)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = ()
kwargs = {'routed_experts_args': {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ......fused_moe.fused_moe.AscendRoutedExperts'>, 'runner_cls': <class 'vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner'>}
hash_enabled = None, tid2eid = None
routed_experts_args = {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ...}

    [0m[94mdef[39;49;00m[90m [39;49;00m[92m_create_ascend_fused_moe_runner[39;49;00m(*args, **kwargs):[90m[39;49;00m
        [90m# Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the[39;49;00m[90m[39;49;00m
        [90m# supported extension point. Pass Ascend implementations through that[39;49;00m[90m[39;49;00m
        [90m# interface instead of relying on the old FusedMoE subclass replacement.[39;49;00m[90m[39;49;00m
        kwargs = [96mdict[39;49;00m(kwargs)[90m[39;49;00m
        hash_enabled = kwargs.pop([33m"[39;49;00m[33mhash[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        tid2eid = kwargs.pop([33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        routed_experts_args = [96mdict[39;49;00m(kwargs.pop([33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m, {}) [95mor[39;49;00m {})[90m[39;49;00m
        routed_experts_args.update([90m[39;49;00m
            {[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_num_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mnum_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_routed_scaling_factor[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_scaling_factor[39;49;00m[33m"[39;49;00m, [94m1.0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_activation[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mactivation[39;49;00m[33m"[39;49;00m, [33m"[39;49;00m[33msilu[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m, [94m0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m: tid2eid,[90m[39;49;00m
                [33m"[39;49;00m[33mhash_enabled[39;49;00m[33m"[39;49;00m: hash_enabled,[90m[39;49;00m
            }[90m[39;49;00m
        )[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrunner_cls[39;49;00m[33m"[39;49;00m, AscendMoERunner)[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrouted_experts_cls[39;49;00m[33m"[39;49;00m, AscendRoutedExperts)[90m[39;49;00m
        kwargs[[33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m] = routed_experts_args[90m[39;49;00m
>       [94mreturn[39;49;00m FusedMoE(*args, **kwargs)[90m[39;49;00m
[1m[31mE       TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'[0m

[1m[31mvllm_ascend/ops/fused_moe/fused_moe.py[0m:1290: TypeError
[31m[1m_______ test_forward_impl_without_shared_experts_returns_routed_only_310 _______[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_forward_impl_without_shared_experts_returns_routed_only_310[39;49;00m():[90m[39;49;00m
>       layer = _build_layer([94mNone[39;49;00m)[90m[39;49;00m

[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:96: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:51: in _build_layer
    [0mlayer = AscendFusedMoE310.[92m__new__[39;49;00m(AscendFusedMoE310)[90m[39;49;00m
[1m[31mvllm_ascend/_310p/fused_moe/fused_moe.py[0m:154: in __new__
    [0m[94mreturn[39;49;00m _create_ascend_fused_moe_runner(*args, **kwargs)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = ()
kwargs = {'routed_experts_args': {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ......fused_moe.fused_moe.AscendRoutedExperts'>, 'runner_cls': <class 'vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner'>}
hash_enabled = None, tid2eid = None
routed_experts_args = {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ...}

    [0m[94mdef[39;49;00m[90m [39;49;00m[92m_create_ascend_fused_moe_runner[39;49;00m(*args, **kwargs):[90m[39;49;00m
        [90m# Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the[39;49;00m[90m[39;49;00m
        [90m# supported extension point. Pass Ascend implementations through that[39;49;00m[90m[39;49;00m
        [90m# interface instead of relying on the old FusedMoE subclass replacement.[39;49;00m[90m[39;49;00m
        kwargs = [96mdict[39;49;00m(kwargs)[90m[39;49;00m
        hash_enabled = kwargs.pop([33m"[39;49;00m[33mhash[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        tid2eid = kwargs.pop([33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        routed_experts_args = [96mdict[39;49;00m(kwargs.pop([33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m, {}) [95mor[39;49;00m {})[90m[39;49;00m
        routed_experts_args.update([90m[39;49;00m
            {[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_num_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mnum_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_routed_scaling_factor[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_scaling_factor[39;49;00m[33m"[39;49;00m, [94m1.0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_activation[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mactivation[39;49;00m[33m"[39;49;00m, [33m"[39;49;00m[33msilu[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m, [94m0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m: tid2eid,[90m[39;49;00m
                [33m"[39;49;00m[33mhash_enabled[39;49;00m[33m"[39;49;00m: hash_enabled,[90m[39;49;00m
            }[90m[39;49;00m
        )[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrunner_cls[39;49;00m[33m"[39;49;00m, AscendMoERunner)[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrouted_experts_cls[39;49;00m[33m"[39;49;00m, AscendRoutedExperts)[90m[39;49;00m
        kwargs[[33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m] = routed_experts_args[90m[39;49;00m
>       [94mreturn[39;49;00m FusedMoE(*args, **kwargs)[90m[39;49;00m
[1m[31mE       TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'[0m

[1m[31mvllm_ascend/ops/fused_moe/fused_moe.py[0m:1290: TypeError
[31m[1m_____________________ test_is_internal_router_is_false_310 _____________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_is_internal_router_is_false_310[39;49;00m():[90m[39;49;00m
>       layer = _build_layer(_DummySharedExperts(with_gate=[94mTrue[39;49;00m))[90m[39;49;00m

[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:108: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mtests/ut/_310p/fused_moe/test_shared_fused_moe_310.py[0m:51: in _build_layer
    [0mlayer = AscendFusedMoE310.[92m__new__[39;49;00m(AscendFusedMoE310)[90m[39;49;00m
[1m[31mvllm_ascend/_310p/fused_moe/fused_moe.py[0m:154: in __new__
    [0m[94mreturn[39;49;00m _create_ascend_fused_moe_runner(*args, **kwargs)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = ()
kwargs = {'routed_experts_args': {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ......fused_moe.fused_moe.AscendRoutedExperts'>, 'runner_cls': <class 'vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner'>}
hash_enabled = None, tid2eid = None
routed_experts_args = {'gate': None, 'hash_enabled': None, 'n_shared_experts': 0, 'original_activation': 'silu', ...}

    [0m[94mdef[39;49;00m[90m [39;49;00m[92m_create_ascend_fused_moe_runner[39;49;00m(*args, **kwargs):[90m[39;49;00m
        [90m# Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the[39;49;00m[90m[39;49;00m
        [90m# supported extension point. Pass Ascend implementations through that[39;49;00m[90m[39;49;00m
        [90m# interface instead of relying on the old FusedMoE subclass replacement.[39;49;00m[90m[39;49;00m
        kwargs = [96mdict[39;49;00m(kwargs)[90m[39;49;00m
        hash_enabled = kwargs.pop([33m"[39;49;00m[33mhash[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        tid2eid = kwargs.pop([33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m, [94mNone[39;49;00m)[90m[39;49;00m
        routed_experts_args = [96mdict[39;49;00m(kwargs.pop([33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m, {}) [95mor[39;49;00m {})[90m[39;49;00m
        routed_experts_args.update([90m[39;49;00m
            {[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_num_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mnum_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_routed_scaling_factor[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_scaling_factor[39;49;00m[33m"[39;49;00m, [94m1.0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33moriginal_activation[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mactivation[39;49;00m[33m"[39;49;00m, [33m"[39;49;00m[33msilu[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mn_shared_experts[39;49;00m[33m"[39;49;00m, [94m0[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mgate[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mshared_experts[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m: kwargs.get([33m"[39;49;00m[33mrouted_input_transform[39;49;00m[33m"[39;49;00m),[90m[39;49;00m
                [33m"[39;49;00m[33mtid2eid[39;49;00m[33m"[39;49;00m: tid2eid,[90m[39;49;00m
                [33m"[39;49;00m[33mhash_enabled[39;49;00m[33m"[39;49;00m: hash_enabled,[90m[39;49;00m
            }[90m[39;49;00m
        )[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrunner_cls[39;49;00m[33m"[39;49;00m, AscendMoERunner)[90m[39;49;00m
        kwargs.setdefault([33m"[39;49;00m[33mrouted_experts_cls[39;49;00m[33m"[39;49;00m, AscendRoutedExperts)[90m[39;49;00m
        kwargs[[33m"[39;49;00m[33mrouted_experts_args[39;49;00m[33m"[39;49;00m] = routed_experts_args[90m[39;49;00m
>       [94mreturn[39;49;00m FusedMoE(*args, **kwargs)[90m[39;49;00m
[1m[31mE       TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'[0m

[1m[31mvllm_ascend/ops/fused_moe/fused_moe.py[0m:1290: TypeError
[31m[1m__ TestAscendCompressedTensorsConfigGetQuantMethod.test_get_moe_quant_method ___[0m

self = <tests.ut.quantization.test_compressed_tensors_config.TestAscendCompressedTensorsConfigGetQuantMethod testMethod=test_get_moe_quant_method>
mock_method = <MagicMock name='__init__' id='140121093986096'>

    [0m[37m@patch[39;49;00m([33m"[39;49;00m[33mvllm_ascend.quantization.methods.AscendW8A8DynamicFusedMoEMethod.__init__[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
    [94mdef[39;49;00m[90m [39;49;00m[92mtest_get_moe_quant_method[39;49;00m([96mself[39;49;00m, mock_method):[90m[39;49;00m
        mock_method.return_value = [94mNone[39;49;00m[90m[39;49;00m
        layer = MagicMock(spec=_fused_moe_spec())[90m[39;49;00m
        layer.moe_config = {}[90m[39;49;00m
>       result = [96mself[39;49;00m.config.get_quant_method(layer, [33m"[39;49;00m[33mmodel.layers.0.mlp.experts[39;49;00m[33m"[39;49;00m)[90m[39;49;00m

[1m[31mtests/ut/quantization/test_compressed_tensors_config.py[0m:119: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mvllm_ascend/quantization/compressed_tensors_config.py[0m:196: in get_quant_method
    [0mmoe_scheme = [96mself[39;49;00m._get_moe_scheme(layer=moe_layer, layer_name=layer_name)[90m[39;49;00m
[1m[31mvllm_ascend/quantization/compressed_tensors_config.py[0m:238: in _get_moe_scheme
    [0mweight_quant, input_quant, [96mformat[39;49;00m = [96mself[39;49;00m._get_quant_args(layer, layer_name)[90m[39;49;00m
[1m[31mvllm_ascend/quantization/compressed_tensors_config.py[0m:269: in _get_quant_args
    [0mscheme_dict = [96mself[39;49;00m.get_scheme_dict(layer, layer_name)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <vllm_ascend.quantization.compressed_tensors_config.AscendCompressedTensorsConfig object at 0x7f7095c7c6b0>
layer = <MagicMock spec='RoutedExperts' id='140121093986192'>
layer_name = 'model.layers.0.mlp.experts.0.gate_proj'

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mget_scheme_dict[39;49;00m([90m[39;49;00m
        [96mself[39;49;00m, layer: torch.nn.Module, layer_name: [96mstr[39;49;00m | [94mNone[39;49;00m = [94mNone[39;49;00m[90m[39;49;00m
    ) -> [96mdict[39;49;00m[[96mstr[39;49;00m, QuantizationArgs | [96mstr[39;49;00m | [94mNone[39;49;00m] | [94mNone[39;49;00m:[90m[39;49;00m
    [90m    [39;49;00m[33m"""[39;49;00m
    [33m    Extract the QuantizationArgs for a given layer.[39;49;00m
    [33m[39;49;00m
    [33m    Returns:[39;49;00m
    [33m        dict with {[39;49;00m
    [33m            "weights": QuantizationArgs,[39;49;00m
    [33m            "input_activations": QuantizationArgs | None,[39;49;00m
    [33m            "format": str | None[39;49;00m
    [33m        } | None[39;49;00m
    [33m    """[39;49;00m[90m[39;49;00m
        [94mif[39;49;00m should_ignore_layer(layer_name, ignore=[96mself[39;49;00m.ignore, fused_mapping=[96mself[39;49;00m.packed_modules_mapping):[90m[39;49;00m
            [94mreturn[39;49;00m [94mNone[39;49;00m[90m[39;49;00m
    [90m[39;49;00m
        [94mif[39;49;00m [96mself[39;49;00m.target_scheme_map:[90m[39;49;00m
            matched_target = find_matched_target([90m[39;49;00m
                layer_name=layer_name,[90m[39;49;00m
                module=layer,[90m[39;49;00m
                targets=[96mself[39;49;00m.target_scheme_map.keys(),[90m[39;49;00m
                fused_mapping=[96mself[39;49;00m.packed_modules_mapping,[90m[39;49;00m
            )[90m[39;49;00m
>           scheme_dict = [96mself[39;49;00m.target_scheme_map[matched_target][90m[39;49;00m
[1m[31mE           KeyError: None[0m

[1m[31mvllm_ascend/quantization/compressed_tensors_config.py[0m:310: KeyError
[33m=============================== warnings summary ===============================[0m
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

../../../usr/local/python3.12.13/lib/python3.12/site-packages/google/protobuf/internal/well_known_types.py:91
  /usr/local/python3.12.13/lib/python3.12/site-packages/google/protobuf/internal/well_known_types.py:91: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
    _EPOCH_DATETIME_NAIVE = datetime.datetime.utcfromtimestamp(0)

../../../usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: 14 warnings
  /usr/local/python3.12.13/lib/python3.12/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/ut/test_compressed_prefix_cache.py:27
  /__w/vllm-ascend/vllm-ascend/tests/ut/test_compressed_prefix_cache.py:27: PytestUnknownMarkWarning: Unknown pytest.mark.cpu_test - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    pytestmark = pytest.mark.cpu_test

tests/ut/test_platform.py:710
  /__w/vllm-ascend/vllm-ascend/tests/ut/test_platform.py:710: SyntaxWarning: invalid escape sequence '\('
    with pytest.raises(ValueError, match="PCP \(Prefill Context Parallelism\) and DP \(Data Parallelism\)"):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[36m[1m=========================== short test summary info ============================[0m
[31mFAILED[0m tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::[1mtest_forward_shared_experts_without_gate_310[0m - TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'
[31mFAILED[0m tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::[1mtest_forward_shared_experts_with_gate_310[0m - TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'
[31mFAILED[0m tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::[1mtest_forward_impl_with_shared_experts_returns_tuple_310[0m - TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'
[31mFAILED[0m tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::[1mtest_forward_impl_without_shared_experts_integration_310[0m - TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'
[31mFAILED[0m tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::[1mtest_forward_impl_without_shared_experts_returns_routed_only_310[0m - TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'
[31mFAILED[0m tests/ut/_310p/fused_moe/test_shared_fused_moe_310.py::[1mtest_is_internal_router_is_false_310[0m - TypeError: FusedMoE() missing 4 required positional arguments: 'num_experts', 'top_k', 'hidden_size', and 'intermediate_size'
[31mFAILED[0m tests/ut/quantization/test_compressed_tensors_config.py::[1mTestAscendCompressedTensorsConfigGetQuantMethod::test_get_moe_quant_method[0m - KeyError: None
[31m=========== [31m[1m7 failed[0m, [32m1250 passed[0m, [33m15 skipped[0m, [33m19 warnings[0m[31m in 19.86s[0m[31m ===========[0m
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
