# 000. 分支和源码阅读地图

本章先说明我们现在站在哪里读代码。

## 1. 两个仓库各自负责什么

```text
/Users/lonng/Mrz20/vllm
  上游 vLLM
  负责通用 LLM API、offline engine、scheduler、模型结构、sampler 基础逻辑。

/Users/lonng/Mrz20/vllm-ascend
  Ascend 硬件插件
  负责 NPU worker、Ascend model runner、Ascend attention、FusedMoE、W8A8、patch。
```

本测试不是只跑 `vllm-ascend` 代码。它是：

```text
上游 vLLM 负责通用推理框架
vLLM Ascend 接管硬件相关执行
```

所以源码讲解必须跨两个仓库。

## 2. 当前教学分支

两个仓库都已切到：

```text
teach-qwen3-5-35b-a3b-source-walkthrough
```

这个分支不是为了改模型行为，而是为了固定当前源码视图，方便你以后跟着文档打开同一套代码。

## 3. 三条阅读线

### 主线 A：请求调度线

回答“prompt 怎么变成一次模型执行”：

```text
tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py
tests/e2e/conftest.py
/Users/lonng/Mrz20/vllm/vllm/entrypoints/offline_utils.py
/Users/lonng/Mrz20/vllm/vllm/v1/engine/llm_engine.py
/Users/lonng/Mrz20/vllm/vllm/v1/engine/core.py
vllm_ascend/worker/model_runner_v1.py
```

### 主线 B：模型计算线

回答“hidden state 怎么经过 Qwen3.5 一层层变化”：

```text
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_5.py
/Users/lonng/Mrz20/vllm/vllm/model_executor/models/qwen3_next.py
vllm_ascend/patch/worker/patch_qwen3_5.py
```

### 主线 C：MoE 和量化线

回答“router、expert、W8A8 怎么真正执行”：

```text
vllm_ascend/ops/fused_moe/fused_moe.py
vllm_ascend/ops/fused_moe/experts_selector.py
vllm_ascend/ops/fused_moe/moe_comm_method.py
vllm_ascend/ops/fused_moe/moe_mlp.py
vllm_ascend/quantization/methods/w8a8_dynamic.py
vllm_ascend/quantization/modelslim_config.py
```

## 4. 读源码时不要混淆两种“流”

第一种是控制流：

```text
哪个函数调用哪个函数
```

第二种是数据流：

```text
prompt -> token ids -> embeddings -> hidden states -> logits -> token ids -> text
```

源码讲解必须把两条线合在一起看。比如：

```text
NPUModelRunner.execute_model()
```

从控制流看，它是 worker 的执行入口。

从数据流看，它接收 `SchedulerOutput`，准备 `input_ids/positions/metadata`，调用模型，拿到 `hidden_states`，再算 `logits`。

## 5. 本次真实数据锚点

后续每章都会回到这些真实数据：

```text
prompt = "Hello, my name is"
prompt_token_ids = [9419, 11, 821, 803, 369]
generated_token_ids = [498, 7525, 3855, 1089, 321]
final_text = "Hello, my name is [Your Name], and"

prefill total_num_scheduled_tokens = 5
prefill num_tokens_padded = 8
decode total_num_scheduled_tokens = 1
hidden_size = 2048
vocab_size = 248320
num_experts = 256
num_experts_per_tok = 8
```

有了这些锚点，你读源码时就不会只看到一堆抽象变量名。
