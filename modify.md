# PR 10334 修改说明

## 背景

本次 PR 主要用于验证 precision testing / coverage 相关流程，目前仍处于测试阶段，因此临时通过 `pull_request` 事件触发。后续流程稳定后，可以按计划移除 PR 触发，仅保留定时或手动触发入口。

当前 CI 失败并不是测试用例本身失败，而是新增 full-test coverage workflow 的执行链路还没有完全打通。

## 当前失败原因

失败 job 为 `E2E / select-full-tests`：

```text
pip: command not found
Process completed with exit code 127
```

对应链接：

https://github.com/vllm-project/vllm-ascend/actions/runs/27333213406/job/80750591966

根因是 `select-full-tests` 运行在裸 `linux-amd64-cpu-8-hk` runner 上，没有使用 lint container，也没有显式安装 Python/pip，但步骤里直接执行了：

```bash
pip install regex pyyaml
```

因此 job 在选择测试用例前就失败，后续 `run-full-tests` 被跳过。

## 已修复内容

### 1. 修复 `select-full-tests` 缺少 pip 的问题

文件：`.github/workflows/pr_test.yaml`

给 `select-full-tests` 增加 lint 容器：

```yaml
container:
  image: quay.io/ascend-ci/vllm-ascend:lint
```

原因：

- 复用现有 `lint-and-select-tests` 的运行环境。
- 避免裸 runner 缺少 `pip`、`pyyaml` 等 Python 工具导致 workflow 提前失败。
- 保持 test selection 阶段和现有 lint/select 流程环境一致。

### 2. 修复 PR 触发时误测 `main` 的问题

文件：`.github/workflows/pr_test.yaml`

原逻辑：

```yaml
ref: ${{ inputs.vllm_ascend_ref || 'main' }}
```

在 `pull_request` 事件下没有 `workflow_dispatch` inputs，因此会默认 checkout `main`。这样 PR 临时触发时，实际验证的是 `main`，不是 PR 中的 workflow/script 修改。

修复后，PR 事件使用 PR head SHA，手动或定时触发仍使用输入 ref 或 `main`：

```yaml
ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || inputs.vllm_ascend_ref || 'main' }}
```

同时 `Resolve refs` 输出也保持同样逻辑，确保后续 `run-full-tests` 测试同一个 ref。

### 3. 修复 schedule 触发条件

文件：`.github/workflows/pr_test.yaml`

`on.schedule` 已经添加，但 `select-full-tests` 的 job 条件原本只允许：

```yaml
github.event_name == 'pull_request' || github.event_name == 'workflow_dispatch'
```

这会导致后续删除 PR 触发后，定时任务不会真正执行 full tests。

修复后增加 `schedule`：

```yaml
if: ${{ github.event_name == 'pull_request' || github.event_name == 'workflow_dispatch' || github.event_name == 'schedule' }}
```

### 4. 修复 `--enable-coverage` 参数传递顺序

文件：

- `.github/workflows/_selected_tests.yaml`
- `.github/workflows/scripts/run_selected_tests.sh`

原问题：

`run_selected_tests.sh` 只在脚本参数最前面解析 `--enable-coverage`，但 workflow 把它放在：

```text
<npu_type> <num_npus> <mode> --enable-coverage <tests...>
```

这样 `--enable-coverage` 会被当作 pytest target，而不是 coverage 开关。

修复方式：

- workflow 中把 `--enable-coverage` 放到脚本参数最前面。
- 脚本自身也增强为可识别任意位置的 `--enable-coverage`，避免后续调用方再次踩同类问题。

### 5. 上传 coverage 数据

文件：`.github/workflows/_selected_tests.yaml`

脚本会把 coverage 数据写入：

```text
tests/outputs/**/covdata/**
```

之前 workflow 只上传了 selected test logs，没有上传 coverage 数据。即使测试成功，也无法从 Actions artifact 中拿到覆盖率原始数据。

本次新增 artifact 上传：

```yaml
- name: Upload coverage data
  if: ${{ always() && inputs.enable-coverage }}
  continue-on-error: true
  uses: actions/upload-artifact@v7
  with:
    name: selected-test-coverage-vllm-${{ inputs.vllm }}-${{ matrix.group.npu_type }}-${{ matrix.group.num_npus }}card
    path: tests/outputs/**/covdata/**
    if-no-files-found: ignore
    retention-days: 14
    compression-level: 0
```

原因：

- 便于后续下载、汇总、分析 coverage 数据。
- 即使测试失败，也尽量保留已生成的 coverage 数据。

### 6. 清理 `tests/coveragerc` 中不通用的路径

文件：`tests/coveragerc`

原配置包含个人硬编码路径：

```ini
data_file = /mnt/share/s00837289/covdata/coverage
```

这在 CI 或其他开发者环境中不一定存在，也不一定可写。虽然脚本中会通过 `COVERAGE_FILE` 覆盖输出路径，但配置文件中保留个人路径仍然容易误导，也不适合进入主线。

修复后：

```ini
data_file = .coverage
```

同时移除了错误的绝对 include 路径：

```ini
/vllm_ascend/*
```

保留通用匹配：

```ini
*/vllm_ascend/*
```

## 验证情况

本地已完成以下轻量验证：

```bash
bash -n .github/workflows/scripts/run_selected_tests.sh
```

- `run_selected_tests.sh` 语法检查通过。
- 使用 Ruby YAML 解析 `.github/workflows/pr_test.yaml` 和 `.github/workflows/_selected_tests.yaml` 通过。
- 使用 fake `python/pytest` 验证 `--enable-coverage` 放在参数前面或中间都能被正确识别。
- `git diff --check` 通过，没有尾随空白或补丁格式问题。

未完成项：

- 本地没有安装 `actionlint`，未做完整 GitHub Actions 语义检查。
- 本地虚拟环境缺少 `pyyaml` / `coverage`，未完整运行 `.github/workflows/scripts/coverage.py` 或真实 coverage 测试。
- NPU full tests 需要依赖 CI 或真实 NPU runner 继续验证。

## 后续建议

1. 将本次修复 push 到 PR 分支，重新触发 CI。
2. 观察 `select-full-tests` 是否能成功输出 `test_groups`。
3. 观察 `run-full-tests` 是否正确 checkout PR head SHA，并开启 coverage。
4. 确认 Actions artifact 中是否生成 `selected-test-coverage-*`。
5. PR 稳定后，按计划移除临时 `pull_request` 触发。
6. 合入前建议整理 commit message，使用 Conventional Commits 并保留 sign-off。
