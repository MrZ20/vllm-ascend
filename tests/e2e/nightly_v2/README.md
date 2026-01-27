# vLLM-Ascend Nightly Test Framework V2

## 概述

此目录包含重构后的 vLLM-Ascend 端到端测试框架。该框架统一了单节点和多节点测试的配置和执行方式，使用 YAML 配置文件驱动测试，提高了测试的可维护性和可扩展性。

## 目录结构

```
nightly_v2/
├── __init__.py
├── README.md                    # 本文档
├── unified_config.py            # 统一配置加载器
├── unified_runner.py            # 统一测试运行器
├── scripts/
│   └── run.sh                   # 统一运行脚本
├── single_node/
│   ├── __init__.py
│   ├── test_single_node.py      # 单节点测试入口
│   └── config/                  # 单节点配置文件
│       ├── Qwen3-32B.yaml
│       ├── DeepSeek-R1-0528-W8A8.yaml
│       ├── Qwen3-235B-A22B-W8A8.yaml
│       └── ...
└── multi_node/
    ├── __init__.py
    ├── test_multi_node.py       # 多节点测试入口
    └── config/                  # 多节点配置文件
        ├── DeepSeek-V3.yaml
        ├── Qwen3-235B-A22B.yaml
        └── ...
```

## 设计原则

### 1. 统一配置格式

单节点和多节点测试使用相同的 YAML 配置格式，主要区别在于：

- **单节点**: `num_nodes=1`，使用 `server_args` 列表配置服务器参数
- **多节点**: `num_nodes>1`，使用 `deployment` 列表为每个节点配置独立的命令

### 2. 配置驱动测试

所有测试逻辑由配置文件驱动，无需修改 Python 代码即可添加新的模型测试：

```yaml
# 单节点配置示例
test_name: "Qwen3-32B single node test"
model: "Qwen/Qwen3-32B"
num_nodes: 1
npu_per_node: 4

env_common:
  TASK_QUEUE_ENABLE: "1"

server_args:
  - "--tensor-parallel-size"
  - "4"
  - "--max-model-len"
  - "36864"

benchmarks:
  - case_type: accuracy
    dataset_path: vllm-ascend/gsm8k-lite
    ...
```

### 3. 向后兼容

多节点配置格式与原有的 `multi_node/config` 保持兼容，只需要将配置文件放到新目录即可使用。

## 使用方法

### 运行单节点测试

```bash
# 方式1: 使用环境变量
CONFIG_YAML_PATH=Qwen3-32B.yaml pytest tests/e2e/nightly_v2/single_node/test_single_node.py

# 方式2: 使用运行脚本
CONFIG_YAML_PATH=Qwen3-32B.yaml ./tests/e2e/nightly_v2/scripts/run.sh single
```

### 运行多节点测试

多节点测试通常在 Kubernetes 环境中运行：

```bash
# 在 K8s LeaderWorkerSet 环境中
CONFIG_YAML_PATH=DeepSeek-V3.yaml pytest tests/e2e/nightly_v2/multi_node/test_multi_node.py

# 使用运行脚本 (自动检测环境)
CONFIG_YAML_PATH=DeepSeek-V3.yaml ./tests/e2e/nightly_v2/scripts/run.sh
```

## 配置文件格式

### 通用字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `test_name` | string | 否 | 测试名称 |
| `model` | string | 是 | 模型路径或 ID |
| `num_nodes` | int | 否 | 节点数量，默认 1 |
| `npu_per_node` | int | 否 | 每节点 NPU 数量 |
| `env_common` | dict | 否 | 环境变量 |
| `benchmarks` | list | 否 | 基准测试配置 |

### 单节点特有字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `server_args` | list | vllm serve 命令参数列表 |

### 多节点特有字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `deployment` | list | 各节点的部署配置 |
| `disaggregated_prefill` | dict | 分离式预填充配置 |
| `cluster_hosts` | list | 非 K8s 环境的节点 IP 列表 |

### 基准测试配置

```yaml
benchmarks:
  - case_type: accuracy|performance
    dataset_path: "vllm-ascend/gsm8k-lite"
    request_conf: "vllm_api_general_chat"
    dataset_conf: "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt"
    max_out_len: 32768
    batch_size: 32
    baseline: 95        # 基线值
    threshold: 5        # 阈值（精度）或比例（性能）
    num_prompts: 100    # 仅性能测试
    request_rate: 0     # 仅性能测试
```

## 添加新测试

### 添加单节点测试

1. 在 `single_node/config/` 创建新的 YAML 配置文件
2. 参考现有配置文件格式填写配置
3. 运行测试验证配置

### 添加多节点测试

1. 在 `multi_node/config/` 创建新的 YAML 配置文件
2. 配置各节点的 `deployment` 和必要的环境变量
3. 在 CI workflow 中添加对应的测试作业

## 从 Python 测试文件迁移

原有的 Python 测试文件可以按以下步骤迁移为 YAML 配置：

1. 提取 `env_dict` 为 `env_common`
2. 提取 `server_args` 列表
3. 提取 `aisbench_cases` 为 `benchmarks`
4. 如果有多个模式（如 single/aclgraph），创建多个配置文件

### 迁移示例

原 Python 文件:
```python
MODELS = ["Qwen/Qwen3-32B"]
env_dict = {"TASK_QUEUE_ENABLE": "1"}
server_args = ["--tensor-parallel-size", "4", ...]
aisbench_cases = [{"case_type": "accuracy", ...}]
```

迁移后的 YAML:
```yaml
model: "Qwen/Qwen3-32B"
env_common:
  TASK_QUEUE_ENABLE: "1"
server_args:
  - "--tensor-parallel-size"
  - "4"
benchmarks:
  - case_type: accuracy
    ...
```

## 与原测试框架的对比

| 特性 | 原框架 | V2 框架 |
|------|--------|---------|
| 配置方式 | Python 硬编码 | YAML 配置文件 |
| 添加新测试 | 需要编写 Python | 只需添加 YAML |
| 单/多节点统一 | 分离的代码路径 | 统一的配置加载器 |
| 参数组合测试 | pytest.mark.parametrize | 多个配置文件 |
| 可维护性 | 需要理解 Python | 只需理解 YAML 格式 |

## CI 集成

在 GitHub Actions workflow 中使用：

```yaml
jobs:
  single_node_test:
    runs-on: linux-aarch64-a2-1
    strategy:
      matrix:
        config:
          - Qwen3-32B.yaml
          - DeepSeek-R1-0528-W8A8.yaml
    steps:
      - name: Run test
        env:
          CONFIG_YAML_PATH: ${{ matrix.config }}
        run: |
          pytest -sv tests/e2e/nightly_v2/single_node/test_single_node.py
```
