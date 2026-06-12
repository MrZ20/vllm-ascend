# PR E2E 临时验证使用说明

本文档用于临时说明 `.github/workflows/pr_test.yaml` 中的 PR E2E 验证开关。使用方式是：在每次 push 前，按需要修改 `pr_test.yaml` 顶部的 `env` 配置，然后推送代码触发 PR CI。

## 开关位置

在 `.github/workflows/pr_test.yaml` 顶部找到：

```yaml
env:
  TEMP_E2E_TESTS: ''
  TEMP_SKIP_DEFAULT_CPU_UT: 'false'
  TEMP_ENABLE_E2E_TIME_SHARDING: 'false'
  TEMP_E2E_MAX_GROUP_SECONDS: '3600'
```

## 配置含义

- `TEMP_E2E_TESTS`
  - 空字符串：保持原有逻辑，根据 PR diff 自动选择测试。
  - `all`：打开 `--run-all-modules`，跑所有已配置模块。
  - 测试路径列表：手动指定要跑的 pytest 目标，支持文件、目录、nodeid，空格或换行分隔。
- `TEMP_SKIP_DEFAULT_CPU_UT`
  - `false`：保留默认 CPU UT。
  - `true`：跳过默认 CPU UT，适合只验证指定 E2E。
- `TEMP_ENABLE_E2E_TIME_SHARDING`
  - `false`：关闭按时间分组，同一个 runner 的用例放在一个 matrix group。
  - `true`：开启按时间分组，同一个 runner 内的 E2E 用例会按时间阈值拆成多个 matrix group 并行跑。
- `TEMP_E2E_MAX_GROUP_SECONDS`
  - 每个分组的目标最大耗时，单位秒，仅在 `TEMP_ENABLE_E2E_TIME_SHARDING: 'true'` 时生效。
  - 单个用例估算时间超过该值时无法继续拆分，会单独成组并打印 warning。

## 常用示例

### 只跑指定 E2E，不跑默认 CPU UT，不分组

```yaml
env:
  TEMP_E2E_TESTS: |
    tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py
    tests/e2e/pull_request/four_card/test_qwen3_5.py
    tests/e2e/pull_request/four_card/spec_decode/test_mtp_qwen3_next.py
  TEMP_SKIP_DEFAULT_CPU_UT: 'true'
  TEMP_ENABLE_E2E_TIME_SHARDING: 'false'
  TEMP_E2E_MAX_GROUP_SECONDS: '3600'
```

### 只跑指定 E2E，并按时间分组并行

```yaml
env:
  TEMP_E2E_TESTS: |
    tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py
    tests/e2e/pull_request/four_card/test_qwen3_5.py
    tests/e2e/pull_request/four_card/spec_decode/test_mtp_qwen3_next.py
  TEMP_SKIP_DEFAULT_CPU_UT: 'true'
  TEMP_ENABLE_E2E_TIME_SHARDING: 'true'
  TEMP_E2E_MAX_GROUP_SECONDS: '3600'
```

### 跑所有已配置模块，并开启分组

```yaml
env:
  TEMP_E2E_TESTS: 'all'
  TEMP_SKIP_DEFAULT_CPU_UT: 'false'
  TEMP_ENABLE_E2E_TIME_SHARDING: 'true'
  TEMP_E2E_MAX_GROUP_SECONDS: '3600'
```

### 恢复默认 PR 选择逻辑

```yaml
env:
  TEMP_E2E_TESTS: ''
  TEMP_SKIP_DEFAULT_CPU_UT: 'false'
  TEMP_ENABLE_E2E_TIME_SHARDING: 'false'
  TEMP_E2E_MAX_GROUP_SECONDS: '3600'
```

## 用例路径格式

支持以下格式：

```text
tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py
tests/e2e/pull_request/four_card/
tests/e2e/pull_request/two_card/test_flashcomm_distributed.py::test_qwen3_dense_fc1_tp2
```

路径会先由 `.github/workflows/scripts/select_tests.py` 按目录规则路由到对应 runner：

- `one_card` -> A2 1 卡
- `two_card` -> A3 2 卡
- `four_card` -> A3 4 卡
- 文件名包含 `_310p` -> 310P runner

## 时间映射表

按时间分组使用 `.github/workflows/scripts/pr_e2e_test_times.yaml`。

这张表应以 `select_tests.py --skip-default-cpu-ut --run-all-modules` 的最终选择结果为准，只保留非 `tests/ut` 目标。配置里精确到 `::` 的用例要保留 nodeid，不要退化成整个 `.py` 文件。

匹配优先级：

1. 完整 nodeid，例如 `test_file.py::test_case`
2. 测试文件路径，例如 `test_file.py`
3. `default_estimated_time`

如果发现某个分组明显过重，可以把对应文件或 nodeid 的估算时间调大；如果分组过轻，可以调小。修改后再次 push 即可让 PR CI 使用新的分组结果。

## 失败处理

每个 matrix group 内部都会使用 `run_selected_tests.py --continue-on-error`。这表示：

- 单个用例失败后不会立刻停止该 group。
- 同一个 group 内剩余用例会继续执行。
- 全部执行完成后，如果有任意失败，最终 job 仍会失败并标红。
