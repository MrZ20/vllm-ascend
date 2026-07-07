# 术语表

这份术语表按本测试会遇到的概念组织。每个术语都尽量回答三件事：它是什么、在本测试中在哪里出现、容易和什么混淆。

## A3B

模型名里的 `A3B` 表示每个 token 的激活参数量约为 3B 级别。它通常来自 MoE 稀疏激活：总参数很多，但每个 token 只走部分 expert。

易混点：`35B` 是总参数规模，`A3B` 是每个 token 激活的近似规模，不是模型只有 3B 参数。

## ACL graph / graph capture

运行时把常见形状的执行图捕获下来，后续复用以减少调度开销。日志里可能沿用 `CUDA graph` 字样；在 Ascend 场景中可理解为对应的 NPU 图执行机制。

本测试配置：

```text
cudagraph_capture_sizes = [1, 2, 4, 8]
```

易混点：图捕获的 dummy forward 不是用户 prompt。

## Ascend

华为 NPU 硬件平台。`vllm-ascend` 是 vLLM 的 Ascend 硬件插件。

本测试里的真实模型计算发生在 NPU 上。

## attention metadata

attention kernel 需要的运行时索引和状态信息，包括 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping`、`causal`、`is_prefilling` 等。

易混点：metadata 不是模型权重，也不是新的一层网络；它告诉 kernel 如何正确解释扁平 token batch 和 KV cache。

## block table

request 到 KV cache block 的映射表。decode 时 attention 通过它找到历史 token 的 K/V。

易混点：`block_table` 管的是 cache 存储位置，不是 token ID。

## causal mask

自回归模型的遮罩规则：当前位置只能看见自己和之前的 token，不能看未来 token。

公式：

```text
mask[i, j] = 0     if j <= i
mask[i, j] = -inf  if j > i
```

## decode

prefill 之后的生成阶段。通常每次只处理上一步新生成的 1 个 token，并通过 KV cache 读取历史上下文。

本测试中 decode 每步：

```text
total_num_scheduled_tokens = 1
```

## EngineCore

上游 vLLM v1 的核心执行层。它调用 scheduler 生成 `SchedulerOutput`，再调用 model executor 执行模型和采样。

源码入口：

```text
/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py
```

## EP / expert parallelism

expert parallelism，把 MoE expert 分布到多个设备上。token 根据 expert 路由结果跨设备通信。

本测试：

```text
enable_expert_parallel = False
```

易混点：EP 关闭不等于 MoE 关闭。MoE 仍然运行，只是不按 expert parallel 拆分。

## FusedMoE

融合 MoE 实现，把 router、token dispatch、expert MLP、token combine 等步骤尽量合并或批量化，减少 Python 循环、小算子和数据搬运。

核心数据：

```text
router_logits
topk_ids
topk_weights
```

## GDN / Gated DeltaNet

Qwen3.5 中 `linear_attention` 层使用的状态式注意力组件。它与 full attention 不同，不是每层都显式做完整 `QK^T`。

易混点：本模型同时有 `linear_attention` 和 `full_attention`，不能把 40 层都理解成标准 full attention。

## GQA

grouped-query attention。Query head 多，Key/Value head 少，多个 Q head 共享较少的 KV head。

本模型：

```text
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
```

## greedy sampling

确定性采样方式，直接选择 logits 最大的 token。

公式：

```text
next_token = argmax(logits)
```

本测试通过 `temperature=0.0` 进入 greedy。

## HCCL

Ascend 上的集合通信库，负责 TP/DP/EP 等并行场景中的通信。

本测试设置：

```text
HCCL_BUFFSIZE = 1024
```

## hidden state

模型内部每个 token 的向量表示。形状通常是：

```text
[N, hidden_size]
```

本模型 hidden size 是 `2048`。

## KV cache

缓存历史 token 的 Key 和 Value，让 decode 不必每次重算完整 prompt。

易混点：KV cache 主要服务 attention；它不是 tokenizer cache，也不是模型权重。

## logits

模型对下一个 token 的原始分数，softmax 前的值。

本次实测：

```text
logits shape = [1, 248320]
```

## logits_indices

runner 中用来选择哪些 hidden state 需要计算 logits 的索引。prefill 虽然处理多个 token，但通常只需要最后一个有效 token 的 hidden state 来采样下一个 token。

## M-RoPE

multi-dimensional RoPE。positions 可能是 `[3, N]`，而不是普通的一维 `[N]`。

本模型配置包含：

```text
mrope_section = [11, 11, 10]
partial_rotary_factor = 0.25
```

## ModelSlim

Ascend 量化模型描述格式来源之一。本测试通过 `quant_model_description.json` 判断每个权重是 `W8A8_DYNAMIC` 还是 `FLOAT`。

## MoE

mixture of experts。router 为每个 token 从多个 expert 中选择少量 expert 执行。

本模型：

```text
num_experts = 256
num_experts_per_tok = 8
```

## MTP

multi-token prediction。模型包名含 `mtp`，仓库也有 Qwen3.5 MTP patch。本次普通 greedy generation 的动态追踪中没有使用 speculative decoding metadata。

## NPU

neural processing unit。Ascend 设备在 PyTorch 中表现为 `npu` device。

易混点：`ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 是物理可见设备集合；进程内通常会映射成 `npu:0`、`npu:1` 等逻辑编号。

## prefill

推理第一阶段，一次性处理 prompt 的所有 token，并写入 KV cache。

本测试：

```text
prompt token 数 = 5
prefill padding 后可执行为 8
```

## prompt

给模型的输入文本。本测试：

```text
Hello, my name is
```

## query_start_loc

扁平 token batch 中每个 request 的起始位置数组。例如本步 token 数是 `[5, 1, 3]`：

```text
query_start_loc = [0, 5, 6, 9]
```

## request

vLLM engine 中的一条生成任务，包含 prompt、token IDs、sampling 参数、生成状态和输出状态。

## RoPE

rotary positional embedding，通过旋转 Q/K 的部分维度注入位置信息。

## scheduler

vLLM 中决定当前 step 跑哪些 request、每个 request 跑几个 token、如何分配 KV cache 的组件。

输出数据结构是 `SchedulerOutput`。

## SchedulerOutput

scheduler 给 worker 的任务单。关键字段：

```text
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_new_reqs
scheduled_cached_reqs
finished_req_ids
```

## slot_mapping

当前 token 写入 KV cache 的具体 slot 映射。

易混点：`slot_mapping` 是“当前 token -> cache slot”，`block_table` 是“request -> cache blocks”。

## SwiGLU

gated MLP 激活形式：

```text
silu(gate) * up
```

MoE expert 内部使用类似结构。

## tensor parallelism / TP

把大矩阵或 tensor 计算拆到多个 rank/device 上。

本测试：

```text
tensor_parallel_size = 2
Worker_TP0 / Worker_TP1
```

## token

tokenizer 输出的基本整数单位。本次 prompt token IDs：

```text
[9419, 11, 821, 803, 369]
```

## tokenizer

负责文本和 token ID 互相转换的组件。

## topk_ids / topk_weights

MoE router 选择结果：

```text
topk_ids shape = [N, 8]
topk_weights shape = [N, 8]
```

`topk_ids` 表示选中哪些 expert；`topk_weights` 表示这些 expert 输出如何加权合并。

## VllmRunner

测试工具类，位于 `tests/e2e/conftest.py`，负责把 e2e 测试参数转成 `vllm.LLM` 调用，并整理输出。

## W8A8

权重和激活都使用 8-bit 表示的量化计算方式。

本测试更准确的说法是：

```text
模型量化描述是 W8A8_DYNAMIC；
大量 MoE expert 权重走 W8A8_DYNAMIC；
部分 attention、linear attention、lm_head 代表性权重仍是 FLOAT。
```

## W8A8_DYNAMIC

激活 scale 在运行时根据当前输入动态计算，权重 scale 来自离线量化权重。

核心数据：

```text
quantized_x
pertoken_scale
weight
weight_scale
weight_offset
```
