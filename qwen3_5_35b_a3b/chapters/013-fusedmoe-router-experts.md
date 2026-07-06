# 013. FusedMoE、Router 和 Experts

本章回答：`35B-A3B` 里的 `A3B`、MoE、router、FusedMoE 到底是什么？代码里又是怎么把 token 分给专家再合并回来的？

先说大白话：MoE 就是很多个“专家 MLP”并排放着。每个 token 进来时，router 先判断这个 token 应该交给哪几个专家处理。本模型有 256 个专家，但每个 token 只选 8 个。FusedMoE 的工作，是把“选专家、按专家重排 token、专家计算、按权重合并”尽量变成少数高效 NPU 算子，而不是 Python 循环一个专家一个专家算。

## 1. 本模型 MoE 配置

远端配置：

```text
num_experts = 256
num_experts_per_tok = 8
moe_intermediate_size = 512
shared_expert_intermediate_size = 512
```

量化描述代表性条目：

```text
model.language_model.layers.0.mlp.experts.0.gate_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.up_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.down_proj.weight = W8A8_DYNAMIC
```

这说明主要专家权重走 W8A8 dynamic 量化路径。

## 2. SparseMoeBlock 是如何接入模型层的

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py:147`
- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:101`

Qwen3.5 decoder layer 初始化时：

```text
if config.model_type == "qwen3_5_moe_text":
    self.mlp = Qwen3NextSparseMoeBlock(...)
```

`Qwen3NextSparseMoeBlock` 里有三个关键对象：

```text
self.gate = ReplicatedLinear(hidden_size, num_experts)
self.shared_expert = Qwen3NextMLP(...)
self.experts = FusedMoE(...)
```

`gate` 是 router。`experts` 是 routed experts 的统一执行入口。`shared_expert` 是共享专家，某些路径下会和 routed experts 并行或融合。

## 3. Router logits 是什么

源码入口：

- `/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py:210`

forward 中：

```text
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)
```

数学上：

```text
router_logits = X W_router
```

形状：

```text
X shape = [N, 2048]
W_router shape = [2048, 256]
router_logits shape = [N, 256]
```

`router_logits[n, e]` 表示第 `n` 个 token 分给第 `e` 个专家之前的原始分数。它还不是概率。

## 4. Top-k 选专家

源码入口：

- `vllm_ascend/ops/fused_moe/experts_selector.py:30`
- `vllm_ascend/ops/fused_moe/experts_selector.py:235`
- `vllm_ascend/ops/fused_moe/experts_selector.py:312`

入口函数：

```text
topk_weights, topk_ids = select_experts(
    hidden_states=x,
    router_logits=router_logits,
    top_k=top_k,
    renormalize=renormalize,
    ...
)
```

如果走 native fallback，核心逻辑是：

```text
if scoring_func == "softmax":
    topk_weights = router_logits.softmax(dim=-1)
elif scoring_func == "sigmoid":
    topk_weights = router_logits.sigmoid()
...
topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

本模型 `top_k=8`，所以：

```text
topk_ids shape = [N, 8]
topk_weights shape = [N, 8]
```

举一个小例子。假设只有 4 个专家，每个 token 选 2 个：

```text
router_prob[token0] = [0.10, 0.55, 0.05, 0.30]
topk_ids[token0] = [1, 3]
topk_weights[token0] = [0.55, 0.30]
renormalize 后 = [0.647, 0.353]
```

真实模型是 256 选 8。

## 5. 为什么要 renormalize

如果从 256 个专家概率里只拿 top-8，它们的概率和可能小于 1：

```text
sum(top8_prob) < 1
```

renormalize 后：

```text
topk_weights_i = topk_weights_i / sum(topk_weights)
```

这样选中专家的加权和仍然保持尺度稳定：

```text
MoE(x) = sum_i topk_weights_i * Expert_i(x)
```

如果不 renormalize，输出幅度会受“被丢弃专家概率总量”影响。

## 6. 每个专家内部是 SwiGLU MLP

源码入口：

- `vllm_ascend/ops/fused_moe/moe_mlp.py:364`
- `vllm_ascend/ops/fused_moe/moe_mlp.py:424`

一个 expert 的结构：

```text
gate = x W_gate
up = x W_up
act = silu(gate) * up
out = act W_down
```

在 fused MoE 权重里，`gate_proj` 和 `up_proj` 常被打包成 `w13_weight`：

```text
w13 = [gate_proj, up_proj]
w2 = down_proj
```

未量化路径里可以看到：

```text
gate_up_out = torch_npu.npu_grouped_matmul(... w1 ...)
gate_up_out = torch_npu.npu_swiglu(gate_up_out)
hidden_states = torch_npu.npu_grouped_matmul(... w2 ...)
```

量化路径里类似，但会用 int8 grouped matmul 和融合的 swiglu quant 算子。

## 7. FusedMoE 的四阶段

源码入口：

- `vllm_ascend/ops/fused_moe/moe_comm_method.py:122`

核心流程：

```text
before_dispatch_evt = record_event()
routed_topk_ids = topk_ids
token_dispatch_output = token_dispatcher.token_dispatch(...)
mlp_compute_input = build_mlp_compute_input(...)
mlp_output, before_gmm2_evt = self._apply_mlp(...)
before_combine_evt = record_event()
routed_out = token_dispatcher.token_combine(...)
```

可以拆成四个阶段：

```text
1. route:    选出 topk_ids/topk_weights
2. dispatch:按专家把 token 重排、复制、分组
3. compute: 每组专家做 grouped matmul + SwiGLU + grouped matmul
4. combine: 按 topk_weights 合并回原 token 顺序
```

## 8. Dispatch：为什么 token 要重排

假设有 3 个 token，每个 token 选 2 个专家：

```text
token0 -> expert7, expert3
token1 -> expert3, expert9
token2 -> expert7, expert7
```

如果按 token 顺序逐个算，会变成很多小矩阵乘：

```text
token0 expert7
token0 expert3
token1 expert3
token1 expert9
token2 expert7
token2 expert7
```

这对硬件很不友好。更好的办法是按专家分组：

```text
expert3: token0, token1
expert7: token0, token2, token2
expert9: token1
```

然后每个专家组做 grouped matmul。

AllGather dispatcher 里关键调用：

```text
sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale =
    DeviceOperator.npu_moe_init_routing(...)
```

返回值含义：

- `sorted_hidden_states`：按专家重排后的 token。
- `expanded_row_idx`：如何恢复原顺序。
- `expert_tokens`：每个专家拿到了多少 token。
- `dynamic_scale`：如果 dispatch 阶段顺便做量化，这里会携带 scale。

## 9. Compute：grouped matmul

源码入口：

- `vllm_ascend/ops/fused_moe/moe_mlp.py:88`
- `vllm_ascend/ops/fused_moe/moe_mlp.py:424`

为什么叫 grouped matmul？

因为不同专家有不同权重：

```text
expert0 用 W0
expert1 用 W1
...
expert255 用 W255
```

但 dispatch 已经把 token 按专家分组，所以可以把“很多个专家自己的矩阵乘”交给一个 grouped matmul 算子处理。

量化路径里大致是：

```text
hidden_states, pertoken_scale = npu_dynamic_quant(hidden_states)
gate_up = grouped_matmul(hidden_states_int8, w13_int8, scales)
act = swiglu(gate_up)
act_int8, act_scale = dynamic_quant(act)
out = grouped_matmul(act_int8, w2_int8, scales)
```

实际代码里有多种融合路径：

- `npu_grouped_matmul_swiglu_quant`
- `grouped_matmul_swiglu_quant_weight_nz_tensor_list`
- `npu_dequant_swiglu_quant`
- `npu_grouped_matmul_gmm2`

选择哪条路径取决于量化类型、通信类型、设备能力、是否启用 fused MC2、是否 dynamic EPLB 等。

## 10. Combine：按路由权重合并

源码入口：

- `vllm_ascend/ops/fused_moe/token_dispatcher.py:427`

AllGather combine 中：

```text
final_hidden_states = DeviceOperator.npu_moe_token_unpermute(
    permuted_tokens=hidden_states,
    sorted_indices=expanded_row_idx,
    probs=topk_weights,
)
```

这一步做两件事：

1. 把按专家排序的输出恢复到原 token 顺序。
2. 对同一个 token 的多个专家输出按 `topk_weights` 加权求和。

数学上：

```text
final[token] = sum_i topk_weights[token, i] * expert_out[token, i]
```

## 11. 本测试为什么是 EP=false

测试参数：

```python
enable_expert_parallel=False
tensor_parallel_size=2
```

EP 是 expert parallel，意思是不同专家可以分布在不同 rank 上，需要 all-to-all 或 MC2 这类通信把 token 发到专家所在 rank。

本测试关闭 EP，所以重点是：

- TP=2：模型张量并行。
- EP=false：不把专家作为 expert parallel 跨 rank 切分。
- `setup_moe_comm_method` 在 `ep_size <= 1` 时注册 AllGather 兼容路径。

源码入口：

- `vllm_ascend/ops/fused_moe/moe_comm_method.py:55`

```text
if moe_config.ep_size > 1:
    register ALLTOALL / ALLGATHER / MC2 / FUSED_MC2
else:
    register ALLGATHER
```

所以本测试不是在验证最复杂的专家并行通信，而是在验证 TP=2 + Ascend quantized MoE + Qwen3.5 patch 能正确生成。

## 12. Fused 的真实意义

“Fused” 不是一个玄学词，它通常意味着：

- 减少 Python 循环。
- 减少小 kernel 数量。
- 减少中间 tensor 落回显存的次数。
- 把 dispatch、matmul、activation、quant、combine 中能合并的部分交给 NPU 优化算子。

如果不用 fused 思路，最直观的写法可能是：

```python
for token in tokens:
    out = 0
    for expert_id, weight in selected_experts[token]:
        out += weight * experts[expert_id](token_hidden)
```

这在概念上清楚，但在 NPU/GPU 上性能会很差，因为会产生大量小矩阵乘和小调度。FusedMoE 把它变成适合硬件的大批量 grouped operation。

## 13. 本章完整链路

把源码链路串起来：

```text
Qwen3NextSparseMoeBlock.forward
  -> router_logits = self.gate(hidden_states)
  -> FusedMoE / AscendMoERunner
    -> select_experts(router_logits)
       -> topk_weights, topk_ids
    -> moe_comm_method.prepare(...)
    -> quant_method.apply(...)
       -> build_fused_experts_input(...)
       -> moe_comm_method.fused_experts(...)
          -> token_dispatch(...)
          -> unified_apply_mlp(...)
          -> token_combine(...)
    -> moe_comm_method.finalize(...)
```

## 14. 初学者检查点

1. `router_logits shape = [N, 256]` 中 256 是什么？
2. `topk_ids shape = [N, 8]` 中 8 是什么？
3. 为什么一个 token 会被复制到多个专家任务里？
4. dispatch 和 combine 分别解决什么问题？
5. 为什么 fused MoE 比 Python 循环专家更适合 NPU？

答案要点：

- 256 是专家总数。
- 8 是每个 token 选择的专家数。
- 因为一个 token 的输出是多个专家输出的加权和。
- dispatch 按专家重排，combine 恢复顺序并加权合并。
- 它能形成更大的 grouped matmul，减少小算子和数据搬运。
