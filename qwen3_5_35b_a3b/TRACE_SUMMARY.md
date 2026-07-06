# 动态追踪摘要

这个文件把追踪日志压缩成可读的调用链和数据变化。它是原始运行日志和分章讲解之间的桥。

## 总体调用链

```text
VllmRunner.generate_greedy
  -> VllmRunner.generate
    -> VllmRunner.get_inputs
    -> LLM.generate
      -> OfflineInferenceMixin._run_completion
        -> OfflineInferenceMixin._render_and_add_requests
          -> OfflineInferenceMixin._add_request
            -> LLMEngine.add_request
        -> OfflineInferenceMixin._run_engine
          -> LLMEngine.step
            -> NPUModelRunner.execute_model
              -> NPUModelRunner._prepare_inputs
              -> NPUModelRunner._preprocess
              -> NPUModelRunner._model_forward
              -> model.compute_logits
            -> NPUModelRunner.sample_tokens
              -> NPUModelRunner._sample
                -> AscendSampler.forward
                  -> Sampler.sample
          -> RequestOutput
    -> VllmRunner._finalize_generate_outputs
```

## 主进程证据

prompt 一开始只是普通 Python 字符串：

```text
PROMPTS=['Hello, my name is']
```

`generate_greedy` 创建 greedy 参数：

```text
SamplingParams({'temperature': 0.0, 'max_tokens': 5, 'top_p': 1.0, 'top_k': 0})
```

`get_inputs` 把字符串包装成 vLLM 的文本输入：

```text
{'multi_modal_data': None, 'prompt': 'Hello, my name is'}
```

渲染/tokenization 阶段生成 token 输入：

```text
{'type': 'token',
 'prompt_token_ids': [9419, 11, 821, 803, 369],
 'prompt': 'Hello, my name is',
 'arrival_time': ...}
```

engine 分配 request ID：

```text
'0-b59b0dc7'
```

最终 `RequestOutput` 包含：

```text
prompt='Hello, my name is'
prompt_ids_head=[9419, 11, 821, 803, 369]
generated text=' [Your Name], and'
generated token_ids=[498, 7525, 3855, 1089, 321]
```

最终 `VllmRunner.generate_greedy` 返回：

```text
[([9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321],
  'Hello, my name is [Your Name], and')]
```

## Worker 侧证据

第一个真实模型 step 是 prefill：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=5))
NPUModelRunner._prepare_inputs(... ndarray(shape=(1,), dtype=int32, head=[5]))
NPUModelRunner._preprocess(... num_tokens_padded=8)
```

解释：

- 当前只有 1 个 request。
- prompt 有 5 个 token。
- 运行时把工作 padding 到 8 个 token，对齐图捕获大小。

prefill 阶段产生下一 token 的 logits：

```text
NPUModelRunner._sample(logits=Tensor(shape=(1, 248320), dtype=torch.bfloat16, device=npu:0))
Sampler.sample(logits=Tensor(shape=(1, 248320), dtype=torch.float32, device=npu:0))
```

解释：

- `1` 表示 batch 中当前只有一条序列需要生成。
- `248320` 是词表大小。
- 模型为每个可能的下一个 token 给出一个分数。

每个 decode step 处理一个新 token：

```text
NPUModelRunner.execute_model(SchedulerOutput(total_num_scheduled_tokens=1))
NPUModelRunner._prepare_inputs(... head=[1])
NPUModelRunner._preprocess(... Tensor(shape=(1, 2048), dtype=torch.bfloat16, device=npu:0))
NPUModelRunner._sample(logits=Tensor(shape=(1, 248320), ...))
```

解释：

- prefill 之后，prompt 的历史信息已经进入 cache。
- 后续每步只喂最新 token。
- 这个过程重复，直到生成满 5 个新 token。

## 准备阶段和真实 prompt 的区别

真实 prompt 之前，runner 还做了预热和图捕获：

```text
_model_forward(num_tokens_padded=8192)
_model_forward(num_tokens_padded=8)
_model_forward(num_tokens_padded=4)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 4/4
Capturing CUDA graphs (decode, FULL): 4/4
```

这些调用不是用户 prompt。它们用于准备内存、通信和可复用执行图，让后面的真实请求更稳定、更快。

## 模型结构证据

远端真实量化模型目录：

```text
/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp
```

`config.json` 中的语言模型配置：

```text
model_type=qwen3_5_moe_text
vocab_size=248320
hidden_size=2048
num_hidden_layers=40
num_attention_heads=16
num_key_value_heads=2
head_dim=256
num_experts=256
num_experts_per_tok=8
moe_intermediate_size=512
shared_expert_intermediate_size=512
layer_types={"linear_attention": 30, "full_attention": 10}
```

这些数字解释了 trace 中两个核心形状：

```text
inputs_embeds shape = (8, 2048)
logits shape = (1, 248320)
```

- `2048` 来自 hidden size。
- `248320` 来自 vocab size。
- `8` 是 prompt 5 个 token padding 到图捕获大小 8。

## 量化证据

`quant_model_description.json` 解析结果：

```text
model_quant_type=W8A8_DYNAMIC
language_or_mtp_value_count={"FLOAT": 1367, "W8A8_DYNAMIC": 92250}
```

代表性条目：

```text
model.language_model.layers.0.mlp.experts.0.gate_proj.weight=W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale=W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.down_proj.weight=W8A8_DYNAMIC
model.language_model.layers.0.linear_attn.out_proj.weight=FLOAT
model.language_model.layers.3.self_attn.o_proj.weight=FLOAT
lm_head.weight=FLOAT
```

解释：

- 主要 MoE expert 权重走 W8A8 dynamic 路径。
- 不是所有权重都量化；部分 attention、linear attention、lm_head 保留浮点。
- 因此讲解 W8A8 时要落到具体模块，不能笼统说“整个模型所有计算都是 int8”。

## MoE 内部链路补充

`_model_forward` 内部的 MoE 路径不会在当前追踪脚本中逐层打印，因为那会产生巨大日志；但源码链路已经定位：

```text
Qwen3_5DecoderLayer.forward
  -> self.mlp(hidden_states)
  -> Qwen3NextSparseMoeBlock.forward
     -> router_logits = self.gate(hidden_states)
     -> self.experts(hidden_states, router_logits)
        -> AscendMoERunner / quant_method.apply
           -> select_experts
           -> build_fused_experts_input
           -> token_dispatch
           -> unified_apply_mlp
           -> token_combine
```

这里的关键数据形状：

```text
router_logits shape = [N, 256]
topk_ids shape = [N, 8]
topk_weights shape = [N, 8]
```

`N` 在 prefill 用户真实 token 上是 5，padding 后 kernel 形状可为 8；decode 时真实 `N=1`。
