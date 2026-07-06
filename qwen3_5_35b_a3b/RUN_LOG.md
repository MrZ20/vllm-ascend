# 运行记录

## 第 1 次：目标 pytest 基线验证

状态：通过。

### 命令

```bash
ssh a3-node0 'ssh node1 '"'"'docker exec zsl_m2m_0612_1 bash -lc '"'"'"'"'"'"'"'"'
cd /vllm-workspace/vllm-ascend
export VLLM_LOGGING_LEVEL=ERROR VLLM_USE_MODELSCOPE=True HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
pytest -sv tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py
'"'"'"'"'"'"'"'"''"'"'
```

### 结果

目标 pytest 在远端 A3 Docker 环境中通过。

关键输出：

```text
================== 1 passed, 17 warnings in 228.68s (0:03:48) ==================
```

重要观察：

- `npu-smi info` 显示物理芯片 4、5、6、7 在运行前基本空闲。
- pytest 只收集到 1 个测试用例。
- 模型加载了 10 个 `safetensors` 权重分片。
- 出现了两个 TP worker：`Worker_TP0` 和 `Worker_TP1`。
- 运行中捕获了 4 个 mixed prefill/decode graph 和 4 个 decode graph，对应 `cudagraph_capture_sizes=[1, 2, 4, 8]`。
- pytest 本身不打印生成文本，只执行 `assert outputs[0][1]`，也就是检查文本非空。

运行中看到的警告：

- `use_fast` 参数弃用警告。
- `torch.jit.script_method` 弃用警告。
- `swap_space` 参数弃用警告。
- 退出时有一个 shared memory 清理警告。

## 第 2 次：标准输入脚本尝试

状态：失败，但这个失败有教学价值。

目的：用一个小脚本直接打印生成文本。

结果：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/vllm-workspace/vllm-ascend/<stdin>'
```

解释：`VLLM_WORKER_MULTIPROC_METHOD=spawn` 会让子进程重新加载主 Python 文件。`python -` 从标准输入执行脚本时没有真实文件路径，所以子进程无法重新加载它。因此后续追踪必须使用真实 `.py` 文件。

## 第 3 次：动态追踪脚本

状态：通过。

使用脚本：

```text
qwen3_5_35b_a3b/trace_qwen_run.py
```

远端执行方式：

1. 把本地脚本复制到远端 `/tmp/qwen_trace_run.py`。
2. 再复制进容器 `/tmp/qwen_trace_run.py`。
3. 使用与目标测试相同的模型参数和环境变量运行该脚本。

真实生成结果：

```text
GENERATE_RESULT=[([9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321], 'Hello, my name is [Your Name], and')]
```

关键数据事实：

- prompt 文本：`Hello, my name is`
- prompt token ID：`[9419, 11, 821, 803, 369]`
- 生成 token ID：`[498, 7525, 3855, 1089, 321]`
- 最终 token ID：`[9419, 11, 821, 803, 369, 498, 7525, 3855, 1089, 321]`
- 最终文本：`Hello, my name is [Your Name], and`
- 第一个真实 scheduler step 处理 5 个 prompt token。
- 后续 decode step 每次处理 1 个 token。
- 采样时 logits 形状是 `(1, 248320)`，意思是：1 条活跃序列，对 248,320 个可能 token 分别打分。

## 第 4 次：远端模型结构配置检查

状态：通过。

目的：确认讲义里的模型维度不是猜测，而是来自远端容器真实缓存的 `config.json`。

实际量化模型目录：

```text
/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp
```

关键文件：

```text
config.json
quant_model_description.json
quant_model_weights-00001-of-00010.safetensors
...
quant_model_weights-00010-of-00010.safetensors
```

配置摘要：

```text
top model_type = qwen3_5_moe
architecture = Qwen3_5MoeForConditionalGeneration
text model_type = qwen3_5_moe_text
vocab_size = 248320
hidden_size = 2048
num_hidden_layers = 40
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
moe_intermediate_size = 512
num_experts = 256
num_experts_per_tok = 8
shared_expert_intermediate_size = 512
rms_norm_eps = 1e-6
hidden_act = silu
max_position_embeddings = 262144
attn_output_gate = true
rope_parameters = {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}
layer_types_count = {"linear_attention": 30, "full_attention": 10}
```

使用脚本：

```text
qwen3_5_35b_a3b/inspect_cached_config.py
```

## 第 5 次：远端量化描述检查

状态：通过。

目的：确认 `W8A8` 具体落在哪些权重上。

使用脚本：

```text
qwen3_5_35b_a3b/inspect_quant_description.py
```

解析文件：

```text
/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3___5-35B-A3B-w8a8-mtp/quant_model_description.json
```

关键结果：

```text
top_key_count = 93955
model_quant_type = W8A8_DYNAMIC
version = 1.0.0
group_size = 0
language_or_mtp_value_count = {"FLOAT": 1367, "W8A8_DYNAMIC": 92250}
language_or_mtp_suffix_count = {
  "dt_bias": 30,
  "A_log": 30,
  "weight": 32057,
  "weight_scale": 30750,
  "weight_offset": 30750
}
```

代表性条目：

```text
model.language_model.layers.0.linear_attn.out_proj.weight = FLOAT
model.language_model.layers.0.mlp.experts.0.gate_proj.weight = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.gate_proj.weight_offset = W8A8_DYNAMIC
model.language_model.layers.0.mlp.experts.0.down_proj.weight = W8A8_DYNAMIC
model.language_model.layers.3.self_attn.o_proj.weight = FLOAT
lm_head.weight = FLOAT
```

结论：本测试模型的主要 MoE expert 权重走 `W8A8_DYNAMIC`，但并非所有权重都是 int8；部分 attention、linear attention、lm_head 等代表性条目仍是 `FLOAT`。
