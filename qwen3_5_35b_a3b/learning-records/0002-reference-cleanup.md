# 0002 - 清理过时信息并补齐参考材料

## 背景

用户指出 `qwen3_5_35b_a3b` 中需要清理过时信息，并补充容易遗漏的复习材料，例如公式速查和术语表。

## 本次更新

- 将 README、OUTLINE、DEEP_OUTLINE 中的“第一轮”“第二版”“临时脚本”等阶段性措辞改成稳定描述。
- 将 `trace_qwen_run.py` 描述为教学插桩脚本，而不是临时脚本。
- 扩充 `reference/formulas.md`，覆盖 embedding、position、RMSNorm、RoPE、attention、KV cache、linear attention、SwiGLU、MoE、FusedMoE、W8A8、TP、logits 和 greedy。
- 扩充 `reference/glossary.md`，加入 A3B、ACL graph、attention metadata、block table、SchedulerOutput、slot mapping、M-RoPE、ModelSlim、GDN、GQA、FusedMoE、W8A8_DYNAMIC 等术语。
- 新增 `reference/tensor-shapes.md`，按推理阶段列出关键 tensor 形状。
- 新增 `reference/runtime-checklist.md`，按环境、加载、调度、runner、attention、MoE、sampling 层排查日志和源码现象。

## 学习意义

这次更新把教学目录从“过程记录”进一步整理成“可复习资料库”。以后读源码时，遇到变量或日志可以先查：

```text
公式不清楚 -> reference/formulas.md
术语不清楚 -> reference/glossary.md
形状不清楚 -> reference/tensor-shapes.md
日志属于哪一层 -> reference/runtime-checklist.md
代码在哪 -> reference/source-map.md
```
