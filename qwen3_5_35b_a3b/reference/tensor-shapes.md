# 关键 Tensor 形状速查

这份表按推理阶段列出本测试最重要的 tensor 和形状。读源码时看到变量名，可以先来这里对照。

## 模型常量

| 项 | 值 |
|---|---:|
| `vocab_size` | `248320` |
| `hidden_size` | `2048` |
| `num_hidden_layers` | `40` |
| `num_attention_heads` | `16` |
| `num_key_value_heads` | `2` |
| `head_dim` | `256` |
| `num_experts` | `256` |
| `num_experts_per_tok` | `8` |
| `moe_intermediate_size` | `512` |
| `shared_expert_intermediate_size` | `512` |

## 输入和输出

| 阶段 | 数据 | 形状或值 | 说明 |
|---|---|---|---|
| prompt | `prompt` | `"Hello, my name is"` | 原始输入字符串 |
| tokenization | `prompt_token_ids` | `[9419, 11, 821, 803, 369]` | tokenizer 输出 |
| generation | `generated_token_ids` | `[498, 7525, 3855, 1089, 321]` | greedy 生成的 5 个 token |
| final | `final_token_ids` | 长度 `10` | prompt token + output token |
| final | `final_text` | `"Hello, my name is [Your Name], and"` | 最终文本 |

## Scheduler 阶段

| 阶段 | 字段 | 形状或值 | 说明 |
|---|---|---|---|
| prefill | `total_num_scheduled_tokens` | `5` | prompt 真实 token 数 |
| prefill | `num_scheduled_tokens_np` | `[5]` | 单 request 的本步 token 数 |
| decode | `total_num_scheduled_tokens` | `1` | 每步一个新 token |
| decode | `num_scheduled_tokens_np` | `[1]` | 单 request 的 decode token 数 |

## Runner 输入准备

| 阶段 | tensor | 典型形状 | 说明 |
|---|---|---|---|
| prefill | `input_ids` | `[8]` 或 `None` | prompt 5 个 token padding 到 8；多模态/embedding 路径可能用 `inputs_embeds` |
| prefill | `inputs_embeds` | `[8, 2048]` | 动态追踪中看到的 embedding 输入 |
| prefill | `positions` | `[3, 8]` 或 `[8]` | M-RoPE 时可为三行位置 |
| prefill | `query_start_loc` | 逻辑上 `[0, 5]` | graph/full 模式下可能 padding |
| decode | `inputs_embeds` | `[1, 2048]` | 每步只处理新 token |
| decode | `positions` | `[3, 1]` 或 `[1]` | 当前新 token 的序列位置 |

## Attention

| 数据 | 形状 | 说明 |
|---|---|---|
| `hidden_states` | `[N_pad, 2048]` | attention 输入 |
| `Q` | `[N, 16, 256]` | query heads |
| `K` | `[N, 2, 256]` | key heads，GQA 中更少 |
| `V` | `[N, 2, 256]` | value heads，GQA 中更少 |
| `attn_output` | `[N_pad, 2048]` | attention 输出回 hidden size |
| `slot_mapping` | `[N_pad]` 或相关后端格式 | 当前 token 到 KV cache slot |
| `block_table` | `[num_reqs, num_blocks]` | request 到 KV cache block |

## MoE

| 数据 | 形状 | 说明 |
|---|---|---|
| `hidden_states` | `[N, 2048]` | 每个 token 的向量 |
| `router_logits` | `[N, 256]` | 每个 expert 一个分数 |
| `topk_ids` | `[N, 8]` | 每个 token 选 8 个 expert |
| `topk_weights` | `[N, 8]` | 8 个 expert 的合并权重 |
| `w13_weight` | `[num_experts, 2 * I_moe, H]` | W8A8 MoE 中 gate/up 打包权重 |
| `w2_weight` | `[num_experts, H, I_moe]` | down projection 权重 |
| `routed_out` | `[N, 2048]` | routed expert 输出 |

## Logits 和 Sampling

| 数据 | 形状 | 说明 |
|---|---|---|
| `hidden_states` | `[N_pad, 2048]` | 模型最后输出 |
| `logits_indices` | `[num_samples]` | 选择要采样的位置 |
| `sample_hidden_states` | `[1, 2048]` | 本次只采样一个位置 |
| `logits` | `[1, 248320]` | 词表分数 |
| `next_token` | 标量 token id | greedy argmax 输出 |

## 形状易混点

- `N` 是真实 token 数，`N_pad` 是为了图捕获或并行约束 padding 后的执行数。
- prefill 真实 token 是 `5`，但动态追踪看到 `inputs_embeds=(8, 2048)`，原因是 padding 到 capture size `8`。
- logits 的第一维是要采样的位置数，不是 prefill token 数；所以 prefill 后 logits 仍然可以是 `[1, 248320]`。
- `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 是可见设备池；本测试 `tensor_parallel_size=2`，所以只需要两个 TP worker。
