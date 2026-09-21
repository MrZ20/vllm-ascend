# 从新 nightly 镜像开始的 V2 验证方案

## 1. 目标与当前状态

本方案供接手验证的 agent 使用。用户将重建 nightly 镜像；当前停止所有重试，
待新镜像就绪后，按「镜像检查 → 代码迁移 → CPU 回归 → 小资源 NPU smoke」执行。
开发修改仍在本地 `refactor_test_framework` 工作区，未提交、未推送。

当前证据只属于旧验证环境：

| 项目 | 结果 |
| --- | --- |
| 本地相关 UT | 133 passed、3 skipped；缺 Linux `/proc` 或本机 vLLM 的用例跳过 |
| Python / workflow / Shell 检查 | 通过 |
| 单机普通服务 | `smoke-single-20260921-02` 通过，AISBench 2/2 成功 |
| 双机 Internal DP | `smoke-internal-20260921-04` 启动失败，benchmark 未执行 |
| Mixed PD、EPD、2P2D、external PD | 未执行 |
| 当前进程 / NPU | 两端本轮 owned 进程已清理，目标端口无监听，NPU 12、13 空闲；没有重试 05 |
| 真实 Actions / LWS | 未部署、未验证；没有开启 cron |

新镜像需要重新跑单机普通服务和 Internal DP，不能沿用旧镜像的通过结论。

### 版本与结果边界

旧环境曾出现 engine 源码与安装包不匹配，因此新镜像先完成原始环境的版本和
导入核对，再迁移测试代码。旧环境的通过结论不能替代新镜像验证。
单机成功退出仍有 `resource_tracker` semaphore/shared-memory 警告，尚未修复。

原始证据：

- `/private/tmp/schedule-v2-evidence/single-02/`
- `/private/tmp/schedule-v2-evidence/internal-04/`
- `/private/tmp/schedule-v2-final-related-ut.log`

## 2. 新镜像与容器准备

### 用户提供 / agent 记录

- 新镜像地址及不可变 digest；两台使用同一镜像。
- 镜像实际的 vLLM、vLLM-Ascend 源码 commit、安装版本。
- 两台宿主机：直接 `ssh a3-node0`、`ssh a3-node2`。
- 容器名：`zsl_nightly`；由用户重建，接手 agent 先确认容器确为新镜像。

容器启动条件：

- 沿用用户已验证的 NPU 设备、驱动挂载和网络设置；两端能互通。
- `/root/.cache` 挂载已有模型/数据缓存，双机应为同一个共享卷。
- `/dev/shm` 建议按测试 workflow 使用 `--shm-size=64g`。
- privileged 环境始终 `SCHEDULE_CLEANUP_PROCESSES=0`，禁止全局 kill。

确认容器实际运行的是新 image ID，再开始验证。

### 检查顺序

1. 在宿主机查看 `docker inspect zsl_nightly` 的 image ID、mounts、ShmSize。
2. 容器内检查 `npu-smi info`，确认设备可见和可用。
3. 记录 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend` 的 HEAD 和 dirty 状态。
   若镜像目录不同，以下所有路径使用实际目录。
4. 从**镜像原始源码目录**先验证 CLI 导入，尚不迁移代码：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python - <<'PY'
import importlib.metadata
import acl
import vllm
import vllm_ascend
import vllm.entrypoints.cli.main
for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "ais-bench", "pytest"):
    print(name, importlib.metadata.version(name))
print("vllm:", vllm.__file__)
print("ascend:", vllm_ascend.__file__)
print("acl:", acl.__file__)
PY
```

依赖的发行包名若与镜像实际名称不同，按 `pip list` 核实后记录；不能因此虚报导入失败。
同时确认 AISBench CLI 和 benchmark source checkout 可用。
镜像原始环境导入失败时停止本阶段，反馈镜像 commit/包组合；不要先拉 main、
复制旧 engine 源码或安装另一版 vLLM 来掩盖问题。

## 3. 迁移当前测试代码

### 3.1 冻结本地待测内容

本地仓库：`/Users/user/work/MrZ20/vllm-ascend`。
分支 HEAD：`977d4c856`；开发基线：`0deca318102a07f9d8e027047652a2a0baab4fbc`。

迁移包：`/private/tmp/schedule-v2-fresh-nightly/migration.tar.gz`。
它包含基线到当前工作区的测试改动及新增文件，不能只迁移 `git diff HEAD`：
此前已提交的 Remote Server 重构也是本次框架的依赖。

包内：

| 内容 | 用途 |
| --- | --- |
| `changes.patch` | 原有文件相对开发基线的改动，含已提交的 Remote* 重构和当前未提交修改 |
| `new/` | 新增 V2 Python、UT、YAML、workflow、文档 |
| `smoke/` | 最新临时小请求 YAML；包含随后修正的 Internal DP 主节点参数 |
| `base/`、`candidate/` | 冲突时对照基线、当前候选文件，按语义合并 |
| `manifest.json` | 本地 HEAD、基线和各文件 SHA256；描述测试源码，不是镜像构建证明 |

本地后续若有开发修改，重新生成迁移包并记录新 hash；不要混用旧 `source-v3/v4/v5`。

### 3.2 保留新镜像的 engine 版本

在每台新容器创建独立验证副本，HEAD 使用**各自新镜像原始 Ascend HEAD**。
不要固定为旧的 `c173a64a4`，不要执行 `git reset --hard`，不要 checkout 本地分支
去覆盖新镜像的 `vllm_ascend/`。

在各容器创建独立验证副本：

```bash
cd /vllm-workspace/vllm-ascend
IMAGE_ASCEND_SHA=$(git rev-parse HEAD)
VALIDATION_ROOT=/tmp/schedule-v2-fresh
# 使用新路径；已存在时先核实归属、HEAD 和内容，不盲目覆盖。
git worktree add --detach "$VALIDATION_ROOT" "$IMAGE_ASCEND_SHA"
```

独立副本需要使用镜像原生构建产物：

- 链接原源码树中的 `.so`、生成的 `_version.py`、`_cann_ops_custom/` 和 `lib/` 所需文件。
- 保持这些产物与新镜像源码配套；不能从旧容器拷贝 native 二进制。
- 不要把整个 `vllm_ascend` package 链接回旧路径，否则验证可能导入错误源码。
- `benchmark` 可以链接到新镜像已有的 benchmark checkout，记录其 commit。
- 如果镜像使用普通 wheel 而非 editable 源码安装，先核对真实导入和 native 布局，
  按实际安装方式适配，不能假设文件路径与旧镜像相同。

### 3.3 应用补丁并处理版本差异

将迁移包传入两端，解压到验证副本之外，例如本机 `/tmp/schedule-v2-migration`。
先比较包的 SHA256，确认两端相同。

1. 在验证副本执行 `git apply --check /tmp/schedule-v2-migration/changes.patch`。
2. 无冲突再 `git apply ...`，然后复制 `new/` 中不存在于目标的新增文件。
3. 有冲突时，对照 `base/`、`candidate/` 与镜像文件做语义合并。
   尤其检查 Remote* 生命周期、设备计数、CLI parser 和 AISBench 接口是否已变化。
4. 新文件若已存在，也应比较和合并，不能直接覆盖。
5. 把包内 `smoke/` 放到验证副本 `tests/e2e/schedule/smoke/`。
6. 迁移后查看 diff；`vllm/`、`vllm_ascend/` 和 native 库不应出现本次迁移造成的修改。

主要已有文件：

- `tests/e2e/utils.py`
- `tests/e2e/conftest.py`
- `tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py`
- `tests/e2e/nightly/multi_node/external_dp/scripts/runtime.py`
- `tests/ut/test_e2e_utils.py`
- `tools/aisbench.py`

`tools/send_request.py`、`tools/send_mm_request.py`、`tools/spec_decode_metrics.py` 和
公共 KV Pool config 是使用中的现有依赖。先检查新镜像的接口；无需为凑齐旧环境
而全量覆盖这些未修改文件。

最后从验证副本检查 `vllm_ascend.__file__`、CLI 导入、关键 Python 文件 hash。
两端的最终框架代码和 smoke YAML 必须相同；源码根路径不同不影响配置 digest。
所有冲突适配都记录在验证报告，开发修复不能交给 Luna；Luna high/max 可执行测试。

## 4. CPU 回归与模型缓存

本地检查使用仓库 `.venv`；容器内使用准备好的 Python 测试环境。

```bash
python -m pytest --noconftest tests/ut/schedule tests/ut/test_e2e_utils.py -q -rs
```

记录准确的 passed/skipped 原因，不能要求新镜像机械复现本机的 3 个 skip。
先让 parser/planner、Remote* 和 runner 回归通过，再启动模型。
发生依赖缺失时，区分轻量测试依赖与 engine/native 版本问题。

复用已有缓存，临时 YAML 中路径仍需逐项确认：

| 内容 | 缓存路径 |
| --- | --- |
| Qwen32B | `/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-32B-W8A8` |
| Qwen30B | `/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-30B-A3B-W8A8` |
| Qwen2.5-VL | `/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct` |
| GSM8K | `/root/.cache/modelscope/hub/datasets/gsm8k` |
| 图片 | `/root/.cache/modelscope/hub/datasets/vllm-ascend/mm_request/test_mm2.jpg` |
| TextVQA perf | `/root/.cache/modelscope/hub/datasets/vllm-ascend/textvqa-perf-1080p` |

Qwen2.5-VL 路径中的三个下划线是 ModelScope 缓存的实际命名。
EPD 配置已包含 `mm_request.images: [null]`，功能请求会加载图片；benchmark 使用
TextVQA base64 配置。不能用纯文本请求代替图片路径验证。

## 5. 新镜像验证矩阵

按顺序执行，一次一个场景；双机同一场景的两端并发启动。
全部使用临时 YAML：2 条 benchmark 请求、16 输出 token、batch size 1，另有短功能请求。
目标是流程打通，不验证全量精度或性能基线。

| 顺序 | 优先级 | 场景 | 资源 | `tests/e2e/schedule/smoke/` 下文件 |
| --- | --- | --- | --- | --- |
| 1 | P0 | 单机普通 Qwen32B | 1 × 4 NPU | `single.yaml` |
| 2 | P0 | 双机 Internal DP Qwen30B | 2 × 2 NPU | `Qwen3-30B-internal-dp.yaml` |
| 3 | P0 | 双机 explicit P + external D | 2 × 2 NPU | `Qwen3-30B-mixed-pd.yaml` |
| 4 | P1 | EPD：Encode + combined PD + proxy | 1 × 4 NPU | `Qwen2.5-VL-7B-EPD.yaml` |
| 5 | P2 | 单机 2P2D | 1 × 4 NPU | `Qwen3-30B-2p2d.yaml` |
| 6 | P2 | 双机 external P + external D | 2 × 2 NPU | `Qwen3-30B-external-pd.yaml` |

Internal DP 主节点已移除 `--data-parallel-start-rank 0`：旧环境的 vLLM 会据此
推断 hybrid/external LB 并拒绝远端 headless。worker 保留 `--headless` 和起始 rank 1。
新镜像若 CLI 语义变化，先查其源码，显式调整 YAML；Planner 不应猜测并补写参数。

TP2 若存在已定位的模型/后端限制，可以改 2 × 4 / TP4 并记录原因；不能仅为绕过
环境错误扩大资源。Mixed PD 的 Service/rank 拓扑保持不变。

## 6. 每轮环境与启动

先检查 `npu-smi info`，选空闲 Ascend RT IDs。旧轮次用过 `12,13` / `12,13,14,15`，
这些 ID **不是预留资源**。同时检查固定端口 `18380–18390`、`23380–23381`、`30000–30400`。
发现他人占用就调整本轮配置，不清理他人的进程。

以下双机示例假设两端验证副本都是 `/tmp/schedule-v2-fresh`。
在本地 `.venv` 中运行 Python，使用直接 SSH 和 `shlex.join`，避免多层引号错误。

```python
from datetime import datetime, timezone
import shlex
import subprocess

run_id = "smoke-internal-fresh-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
config = "tests/e2e/schedule/smoke/Qwen3-30B-internal-dp.yaml"
repo = "/tmp/schedule-v2-fresh"
log_root = "/root/.cache/schedule-v2-validation"
for host, index in [("a3-node0", 0), ("a3-node2", 1)]:
    env = {
        "REPO_ROOT": repo,
        "CONFIG_YAML_PATH": config,
        "CLUSTER_HOSTS": "172.22.0.218,172.22.0.188",
        "LWS_WORKER_INDEX": str(index),
        "NUM_NODES": "2",
        "NPU_PER_NODE": "2",
        "ASCEND_RT_VISIBLE_DEVICES": "12,13",  # 先确认空闲
        "RUN_ID": run_id,
        "LOG_PREFIX": log_root,
        "COORD_DIR": log_root,
        "SCHEDULE_CLEANUP_PROCESSES": "0",
        "VLLM_LOGGING_LEVEL": "ERROR",
        "VLLM_USE_MODELSCOPE": "True",
        "HF_HUB_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    node_dir = f"{log_root}/{run_id}/node-{index}"
    invoke = shlex.join(["env", *[f"{k}={v}" for k, v in env.items()],
                         "bash", f"{repo}/tests/e2e/schedule/scripts/run.sh"])
    script = f"""
set -u
mkdir -p {shlex.quote(node_dir)}
printf '%s\\n' "$$" > {shlex.quote(node_dir + '/launcher.pid')}
{invoke} > {shlex.quote(node_dir + '/launcher.log')} 2>&1
status=$?
printf '%s\\n' "$status" > {shlex.quote(node_dir + '/launcher.rc')}
exit "$status"
"""
    command = shlex.join(["docker", "exec", "-d", "zsl_nightly", "bash", "-lc", script])
    subprocess.run(["ssh", "-o", "ClearAllForwardings=yes", host, command], check=True, timeout=30)
print(run_id)
```

`docker exec -d` 返回只说明已提交启动，继续检查两个节点日志和退出码。
`run.sh` 会 source Ascend/ATB 并保留 PYTHONPATH；不要手动覆盖整个 CANN PYTHONPATH。
新镜像的宿主机 IP 若变化，先更新 CLUSTER_HOSTS。
每次重试都使用新的 RUN_ID。

单机配置：

- 只启动 a3-node0，`LWS_WORKER_INDEX=0`。
- `NUM_NODES=1`、`NPU_PER_NODE=4`、可见空卡 4 张。
- `CLUSTER_HOSTS=172.22.0.218`，便于 EPD/2P2D 解析实际网卡。
- 换成矩阵中的对应 config；COORD_DIR 可以保留。

双机 Mixed / external PD 沿用双机环境，更换 config 和 RUN_ID 即可。
不要给框架偷偷增加 `--headless`、DP rank 或 connector 参数。

## 7. 验收与失败处理

每轮根目录：`/root/.cache/schedule-v2-validation/<RUN_ID>/`。

1. NPU preflight 通过，资源数量和实际可见卡符合预期。
2. 两端 metadata 的配置 digest、框架代码及 resources 一致。
3. 检查 plan/effective 文件中的进程数量、PID/PGID、rank、端口和实际分卡。
4. 所有 backend ready 后 proxy 才启动；功能请求成功。
5. benchmark JSON 显示 Total=2、Success=2、Failed=0。
6. 两端 launcher.rc 为 0，case result、cleanup 均通过。
7. 共享 `cases/000-*/final.json` 通过，所有 node result 存在。
8. 本轮 owned 进程退出，目标端口释放，选中 NPU 恢复空闲。
9. 保存 server/proxy/tests logs、JSON、版本、镜像 digest 和启动命令。

失败时停止推进后续场景，先看**最早的根因**：

- 镜像原始导入失败：回到镜像准备。
- 端口 / NPU 忙：选择空闲资源，并同步调整本轮配置。
- 原生 CLI 语义不符：对照新镜像源码调整临时 YAML。
- 模型 / connector 不支持：保留证据，按明确限制调整模型或资源。
- Planner、readiness、runner、cleanup 错误：由开发 agent 修复并补对应 UT。

所有失败只清理本轮拥有的进程：

- `SCHEDULE_CLEANUP_PROCESSES=0` 始终保持。
- 禁止 pkill/killall 或按 python/vllm 名称批量 kill。
- 人工中断前确认本轮 pytest 的 PID、父子关系和 RUN_ID，再发送 TERM，等待 finally。
- 必须升级清理时，仅针对确认属于本轮的 PID/PGID。
- 中断节点清理后立即返回的回归已修复；不要改回等待未启动 peer 的结果。

`resource_tracker` 警告单独记录。请求成功和 NPU 释放不能证明该警告已修复。

## 8. 交付结果与后续 CI

每个 case 报告：镜像/commit、配置 digest、资源、ready/request/benchmark、退出码、
cleanup、NPU 释放、warning，以及最终 PASS / FAIL / NOT_RUN。
变更过的 YAML、框架修复、镜像兼容调整分别列出。

本地 NPU smoke 完成后，再在获授权的测试环境验证 A3 V2 单机 workflow 和 LWS 双机
workflow 的调度、退出码、artifact、Pod 清理。当前不操作生产集群、不启用 cron。
全量精度/性能、KV Pool 硬件路径、speculative acceptance、其他 SoC 属于后续独立范围。
