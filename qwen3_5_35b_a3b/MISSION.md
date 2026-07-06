# 使命：从零理解 Qwen3.5-35B-A3B W8A8 推理

## 为什么学

通过真实运行 `Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp` 的双卡 pytest 用例，理解一次 vLLM Ascend 推理从 shell 工作流、pytest 包装、vLLM 离线推理接口、Ascend worker 执行，到模型生成文本的完整过程。

## 学会以后应该能做到

- 按相同的 SSH、Docker、环境变量、NPU 选卡和 pytest 流程运行目标用例。
- 解释 `"Hello, my name is"` 如何变成 token ID、logits、采样 token ID 和最终文本。
- 解释 TP=2、EP 关闭、MoE/A3B、W8A8 量化、MTP、KV cache、prefill、decode、greedy sampling 各是什么意思。
- 能顺着源码说明这个测试在 `vllm-ascend` 和隔壁 `vllm` 仓库里经过了哪些关键函数。

## 约束

- 按小白视角，从 0 到 1 解释，不默认读者懂推理或模型结构。
- 以当前仓库和 `/Users/lonng/Mrz20/vllm` 的源码为依据。
- 保留真实运行工作流和动态追踪证据。
- 概念解释和源码细节分开写，方便分章节阅读。

## 暂不展开

- Qwen 模型训练流程。
- Transformer 的完整数学推导。
- 性能调优、精度评估或 benchmark。
