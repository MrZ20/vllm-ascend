# Qwen3.5-35B-A3B W8A8 教学记录

这个目录记录目标用例的运行流程、验证证据、动态函数调用链和小白向讲解：

```bash
pytest -sv tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py
```

## 当前状态

- 源码定位：已完成目标链路的第一轮梳理。
- 教学大纲：已创建。
- 远端 pytest：已通过。
- 动态追踪：已通过临时脚本采集。
- 分章讲解：已升级为连续深度版，包含运行环境、模型加载、request/scheduler、attention metadata、并行内存、数据推导、prefill/decode、Transformer、FusedMoE、W8A8 和 sampling。
- 源码逐跳讲解：已创建教学分支和 `source-walkthrough/`。
- 模型配置：已从远端真实缓存目录读取。
- 量化描述：已从 `quant_model_description.json` 解析。

## 推荐阅读顺序

1. `DEEP_OUTLINE.md`：先看深度路线和已确认事实。
2. `TRACE_SUMMARY.md`：看真实运行中函数和张量怎么变化。
3. `chapters/000-overview.md` 到 `chapters/004-model-runtime-concepts.md`：先建立一屏心智模型。
4. `chapters/005-runtime-environment-and-processes.md`：理解环境变量、进程和 NPU 选择。
5. `chapters/006-model-loading-quant-config-and-patches.md`：理解模型加载、量化描述和 Ascend patch。
6. `chapters/007-request-engine-scheduler-lifecycle.md`：理解 request、engine、scheduler 到 worker 的生命周期。
7. `chapters/008-attention-metadata-position-and-backend.md`：理解 positions、KV cache 索引和 attention metadata。
8. `chapters/009-parallelism-memory-and-graph-capture.md`：理解 TP、内存 profiling、KV cache 和图捕获。
9. `chapters/010-end-to-end-data-derivation.md`：从 prompt 到输出完整走一遍。
10. `chapters/011-prefill-decode-kv-cache.md`：理解 prefill、decode、KV cache。
11. `chapters/012-transformer-qwen35-layer.md`：理解 Qwen3.5 层结构。
12. `chapters/013-fusedmoe-router-experts.md`：理解 A3B、router、FusedMoE。
13. `chapters/014-w8a8-dynamic-quantization.md`：理解 W8A8_DYNAMIC。
14. `chapters/015-logits-sampling-output.md`：理解 logits 到最终文本。
15. `SOURCE_WALKTHROUGH.md`：进入源码逐跳讲解。
16. `source-walkthrough/000-branch-and-reading-map.md` 到 `source-walkthrough/006-sampling-and-output.md`：按真实源码调用链逐步阅读。

## 编号说明

最早先写了 `000-004` 作为基础扫盲，再把第二轮数学深挖放在 `010-015`，中间 `005-009` 原本预留给运行系统和模型数学之间的桥梁层。现在这些中间章节已经补齐，所以阅读路径是连续的：

```text
000-004 基础心智模型
005-009 运行时桥梁层
010-015 核心数学和源码深挖
```

## 文件说明

- `MISSION.md`：本教学记录的目标。
- `RESOURCES.md`：源码和资料索引。
- `NOTES.md`：用户要求、教学偏好和注意事项。
- `RUNBOOK.md`：实际运行命令和每条命令的作用。
- `RUN_LOG.md`：真实运行结果、警告、追踪结果和关键数据。
- `TRACE_SUMMARY.md`：动态调用链和数据形状摘要。
- `OUTLINE.md`：教学大纲。
- `DEEP_OUTLINE.md`：深度版教学路线和模型配置事实。
- `SOURCE_WALKTHROUGH.md`：源码逐跳讲解入口。
- `trace_qwen_run.py`：动态追踪用临时脚本。
- `inspect_cached_config.py`：远端容器内读取模型 `config.json` 的只读脚本。
- `inspect_quant_description.py`：远端容器内读取 `quant_model_description.json` 的只读脚本。
- `chapters/`：按章节展开的讲解。
- `source-walkthrough/`：按代码调用顺序展开的源码讲解。
- `reference/`：速查表和术语表。
- `learning-records/`：后续记录用户真正掌握的知识点。
