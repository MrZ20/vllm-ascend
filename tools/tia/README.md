# TIA（Test Impact Analysis）在 CI 中的工作流与使用教程

本文档说明 vllm-ascend 在 CI 中如何进行 **测试影响分析（TIA）**，包括：
- 如何生成 “源码函数/文件 -> 测试用例” 映射表（Mapper）
- PR 中如何基于 diff 推荐测试用例列表 A（Analyzer）
- PR 仍运行 full 列表 B，并在影子模式下校验 A 是否覆盖 B 中的失败用例（Verifier）

> 相关实现文件：
> - 映射生成与合并：`tools/tia/generate_mapping.py`
> - PR 变更分析：`tools/tia/analyze_changes.py`
> - AST 解析：`tools/tia/ast_parser.py`
> - 影子模式校验：`tools/tia/verify_coverage.py`
> - 映射生成工作流：`.github/workflows/schedule_tia_mapping_generation.yaml`
> - PR E2E 工作流（复用）：`.github/workflows/_e2e_test.yaml`

---

## 1. 整体目标与术语

- **列表 A（Recommended Tests）**：根据 PR 改动自动推荐的测试用例集合，用于评估能否覆盖改动。
- **列表 B（Full Tests）**：CI 仍然执行的全量 E2E Full 测试集合（真实 gate）。
- **影子模式（Shadow Mode）**：CI 运行 B 的同时，生成并记录 A，最后用失败用例集合去验证 A 的覆盖率；不改变 gate 逻辑。

---

## 2. Mapper：生成“源码 -> 测试用例”映射表

### 2.1 触发方式

工作流文件：`.github/workflows/schedule_tia_mapping_generation.yaml`

触发：
- 定时：每周日 02:00 UTC
- 手动：GitHub Actions 中 `workflow_dispatch`

### 2.2 生成过程（3 个维度）

工作流会分别在以下测试集合上生成 `mapping_*.json`：
1. 单卡：`tests/e2e/singlecard/`  -> `mapping_singlecard.json`
2. 双卡：`tests/e2e/multicard/2-cards/` -> `mapping_2cards.json`
3. 四卡：`tests/e2e/multicard/4-cards/` -> `mapping_4cards.json`

每个 job 会：
1. 运行 pytest 并启用 coverage context（test_function）：
   - 核心约束：必须能从 `.coverage` 中取到 `(file, line, test_context)`
2. 执行 `tools/tia/generate_mapping.py generate`，把 coverage 行覆盖转换成：
   - `file_mapping`: 文件级映射（utility/config 类文件）
   - `func_mapping`: 函数级映射（核心逻辑文件）

### 2.3 合并映射

在 workflow 的 `merge-mappings` job 中，会执行：
- `tools/tia/generate_mapping.py merge mapping_singlecard.json mapping_2cards.json mapping_4cards.json --output mapping.json`

并上传 artifact：`tia-mapping`（包含 `mapping.json`）。

---

## 3. Analyzer：在 PR 中推荐测试用例列表 A

脚本：`tools/tia/analyze_changes.py`

输入：
- `mapping.json`（由上一步产生的 artifact）
- `git diff`（默认 base 为 `origin/main` vs `HEAD`）

输出：
- `recommended_tests.json`
  - `all_tests=true` 或 `recommended_tests=[...]`

规则摘要：
1. 若改动包含 `csrc/`、`CMakeLists.txt`、`setup.py` 等（构建/算子链路），直接返回 `ALL_TESTS`。
2. 对 Python 改动：
   - 解析 diff 得到变更行号集合
   - 对当前 workspace 最新版本做 AST 解析，定位变更落在哪些函数/方法区间
3. 查表推荐：
   - utility/config 文件：查 `file_mapping`
   - 其他文件：查 `func_mapping`
   - 新文件/无法定位：记录 unmapped，必要时建议回退到模块级全量（策略可按团队口径调整）

---

## 4. Verifier：影子模式验证 A 是否覆盖 B 的失败用例

脚本：`tools/tia/verify_coverage.py`

做法：
1. Full E2E（列表 B）按原 CI 逻辑执行（不减少测试）。
2. 将 Full E2E 的测试结果输出为 JUnit XML（`--junitxml=...`）。
3. 从 JUnit XML 中提取失败/错误用例列表 `failed_tests`。
4. 判断 `failed_tests ⊆ recommended_tests`：
   - 覆盖率 = covered_failures / total_failures
   - 输出 `tia_report.json`，并写入 GitHub Step Summary

> 注意：为了匹配成功，Verifier 会尽量把 pytest 的 `classname` 归一化成 `tests/.../*.py::test_name` 风格，再与推荐列表进行比对。

---

## 5. CI 集成范围（仅 3 个 E2E Full Job）

在 `.github/workflows/_e2e_test.yaml` 中，仅对以下 job 做 TIA（影子模式）：
1. `e2e-full`（singlecard-full）：`run_suite.py --suite e2e-singlecard`，分片 `part: [0,1]`
2. `e2e-2-cards-full`（multicard-2-full）：`run_suite.py --suite e2e-multicard-2-cards`
3. `e2e-4-cards-full`（multicard-4-full）：`run_suite.py --suite e2e-multicard-4-cards`

---

## 6. 教学：如何本地/CI 验证链路

### 6.0 无 NPU 自检（推荐先跑）

该自检不依赖 NPU，也不依赖 pytest（仅使用 Python 标准库），用于快速验证 TIA 三段核心逻辑：

```bash
python3 tools/tia/selfcheck.py
```

### 6.1 本地生成映射（示例）

```bash
# 注意：必须使用 pytest-cov 的动态 context 才能得到“测试函数级别”的映射。
pytest -sv tests/e2e/singlecard/ \
   --cov=vllm_ascend --cov-context=test_function --cov-report=

python3 tools/tia/generate_mapping.py generate \
   --coverage-db .coverage \
   --source-dir vllm_ascend \
   --project-root . \
   --output mapping_singlecard.json
```

### 6.2 本地分析当前分支推荐用例

```bash
git fetch origin main --depth=1
python tools/tia/analyze_changes.py --mapping mapping.json --base-ref origin/main --output recommended_tests.json
cat recommended_tests.json
```

### 6.3 本地影子验证（需 junitxml）

```bash
pytest tests/e2e/singlecard/ --junitxml=test-results.xml || true
python tools/tia/verify_coverage.py --recommended recommended_tests.json --junit-xml test-results.xml --output tia_report.json
cat tia_report.json
```

---

## 7. 常见问题（Troubleshooting）

1. **推荐列表为空或大量 unmapped**
   - 检查 `mapping.json` 是否存在且内容量足够（artifact 是否更新）
   - 检查 coverage 是否启用了 `--context=test_function`
   - 检查 `.coverage` 行号解码是否正确（建议使用 coverage 官方 numbits 解码）

2. **Verifier 报告显示“missed failures”但你认为应该覆盖**
   - 多数是测试 ID 归一化差异：`classname` vs `tests/.../*.py`
   - 确保 junitxml 的 testcase 能转成路径风格进行匹配

3. **Full suite 没有生成 JUnit XML**
   - 需要确保 `run_suite.py` 支持并透传 pytest 参数（例如 `--junitxml`）

---
