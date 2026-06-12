#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Determine which tests to run based on changed files in a PR.

Pipeline:
  1. Diff       -- get changed files from git.
  2. Match      -- identify affected modules via test_config.yaml.
  3. Collect    -- gather test paths (always resolved to individual files).
  4. Route      -- determine runner for each test file by convention:
                      UT:  directory path pattern (a2/, a3_2/, 310p/, etc.)
                      E2E: directory path pattern (one_card, two_card, four_card, 310p)
  5. Output     -- write test_groups / has_tests / matched_modules.

Directory conventions for UT runner routing:
  tests/ut/<module>/            -> CPU runner (default)
  tests/ut/<module>/a2/         -> A2 NPU x1
  tests/ut/<module>/a2_2/       -> A2 NPU x2
  tests/ut/<module>/a3_2/       -> A3 NPU x2
  tests/ut/<module>/a3_4/       -> A3 NPU x4
  tests/ut/<module>/310p/       -> 310P NPU x1

Directory conventions for E2E runner routing:
  tests/e2e/pull_request/one_card/    -> A2 NPU x1
  tests/e2e/pull_request/two_card/    -> A3 NPU x2
  tests/e2e/pull_request/four_card/   -> A3 NPU x4
  *_310p.py under one/two-card paths  -> 310P NPU x1
  *_310p.py under four-card paths     -> 310P NPU x4

Usage:
    python select_tests.py --diff-base origin/main
    python select_tests.py --changed-files file1.py file2.py

Flags:
    --run-all-modules   Run tests for all configured modules regardless of
                        changed files
    --skip-default-cpu-ut
                        Skip the always-on default CPU UT module
    --enable-time-sharding
                        Split non-CPU runner groups by estimated time
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import regex as re
import yaml

_SCRIPT_DIR = Path(__file__).parent
_CONFIG_PATH = _SCRIPT_DIR / "test_config.yaml"
_RUNNER_LABEL_PATH = _SCRIPT_DIR / "runner_label.json"
_DEFAULT_TEST_TIME_MAP_PATH = _SCRIPT_DIR / "pr_e2e_test_times.yaml"
_DEFAULT_CPU_UT_MODULE = "default_cpu_ut"
_DEFAULT_ESTIMATED_TIME = 600.0


class NpuType(str, Enum):
    A2 = "a2"
    A3 = "a3"
    _310P = "310p"
    CPU = "cpu"


@dataclass
class RunnerInfo:
    num_npus: int
    npu_type: NpuType
    label: str
    image_tag: str = ""


@dataclass
class TestTimeMap:
    default_estimated_time: float
    tests: dict[str, float]


RunnerKey = tuple[int, NpuType]
_DEFAULT_KEY: RunnerKey = (0, NpuType.CPU)

_UT_DIR_PATTERNS: list[tuple[re.Pattern, NpuType, int]] = [
    # Order matters: longer/more-specific patterns first (e.g. /a2_2/ before /a2/).
    (re.compile(r"/a2_2/"), NpuType.A2, 2),
    (re.compile(r"/a2/"), NpuType.A2, 1),
    (re.compile(r"/a3_4/"), NpuType.A3, 4),
    (re.compile(r"/a3_2/"), NpuType.A3, 2),
    # /310p/ matches the convention subdir (e.g. tests/ut/<module>/310p/).
    # Note: tests/ut/_310p/ (top-level module, underscore prefix) is NOT matched
    # by this pattern — those tests run on CPU in mock mode, which is intentional.
    (re.compile(r"/310p/"), NpuType._310P, 1),
]

_E2E_DIR_PATTERNS: list[tuple[re.Pattern, NpuType, int]] = [
    (re.compile(r"/four_card/"), NpuType.A3, 4),
    (re.compile(r"/two_card/"), NpuType.A3, 2),
    (re.compile(r"/one_card/"), NpuType.A2, 1),
]


def _as_posix_path(path: str) -> str:
    return path.replace("\\", "/")


def _pytest_node_file_path(path: str) -> str:
    """Return the real file path for a pytest nodeid target."""
    return path.split("::", 1)[0]


def _route_e2e_file(file_path: str) -> RunnerKey | None:
    route_path = _as_posix_path(_pytest_node_file_path(file_path))
    if "_310p" in Path(route_path).name:
        if "/four_card/" in route_path:
            return (4, NpuType._310P)
        return (1, NpuType._310P)
    return _route_e2e_dir(route_path)


def _load_runners() -> list[RunnerInfo]:
    with open(_RUNNER_LABEL_PATH) as f:
        raw = json.load(f)
    return [
        RunnerInfo(
            num_npus=info["npu_num"],
            npu_type=NpuType(info["chip"]),
            label=label,
            image_tag=info.get("image_tag", ""),
        )
        for label, info in raw.items()
    ]


def _load_test_time_map(path: Path) -> TestTimeMap:
    if not path.exists():
        print(
            f"Warning: test time map {path} does not exist; using default estimates.",
            file=sys.stderr,
        )
        return TestTimeMap(default_estimated_time=_DEFAULT_ESTIMATED_TIME, tests={})

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Test time map must be a YAML mapping: {path}")

    default_estimated_time = float(data.get("default_estimated_time", _DEFAULT_ESTIMATED_TIME))
    if default_estimated_time <= 0:
        raise SystemExit("default_estimated_time must be greater than 0")

    raw_tests = data.get("tests", {}) or {}
    if not isinstance(raw_tests, dict):
        raise SystemExit("tests in test time map must be a mapping of pytest target to seconds")

    tests: dict[str, float] = {}
    for target, estimated_time in raw_tests.items():
        value = float(estimated_time)
        if value <= 0:
            raise SystemExit(f"Estimated time for {target} must be greater than 0")
        tests[str(target).rstrip("/")] = value

    return TestTimeMap(default_estimated_time=default_estimated_time, tests=tests)


def _estimated_time_for_target(target: str, test_time_map: TestTimeMap) -> float:
    normalized = target.rstrip("/")
    if normalized in test_time_map.tests:
        return test_time_map.tests[normalized]

    file_path = _pytest_node_file_path(normalized).rstrip("/")
    if file_path in test_time_map.tests:
        return test_time_map.tests[file_path]

    return test_time_map.default_estimated_time


def _get_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def _matches_path_dependency(file_path: str, dependency: str) -> bool:
    dep = dependency.rstrip("/")
    return file_path == dep or file_path.startswith(dep + "/")


def _as_base_list(base: str | list[str] | None) -> list[str]:
    if base is None:
        return []
    if isinstance(base, str):
        return [base]
    return base


def _merge_unique(parent: list, child: list) -> list:
    result = list(parent)
    for item in child:
        if item not in result:
            result.append(item)
    return result


def _resolve_config_inheritance(config: list[dict]) -> list[dict]:
    module_map = {m["name"]: m for m in config}
    resolved: dict[str, dict] = {}
    resolving: set[str] = set()
    inherited_fields = (
        "source_file_dependencies",
        "exclude_source_file_dependencies",
        "tests",
        "skip_tests",
    )

    def resolve(name: str) -> dict:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"Circular test config inheritance detected for module: {name}")
        if name not in module_map:
            raise ValueError(f"Unknown base module in test config: {name}")

        resolving.add(name)
        module = dict(module_map[name])
        inherited_values = {field: [] for field in inherited_fields}
        for base_name in _as_base_list(module.get("base")):
            base_module = resolve(base_name)
            for field in inherited_fields:
                inherited_values[field] = _merge_unique(inherited_values[field], base_module.get(field, []))
        for field in inherited_fields:
            module[field] = _merge_unique(inherited_values[field], module.get(field, []))
        resolving.remove(name)
        resolved[name] = module
        return module

    return [resolve(module["name"]) for module in config]


def _match_modules(
    changed_files: list[str],
    config: list[dict],
) -> list[str]:
    if not changed_files:
        return []
    matched: list[str] = []
    for module in config:
        if not module.get("optional", True):
            matched.append(module["name"])
            continue
        deps = module.get("source_file_dependencies", [])
        exclude_deps = module.get("exclude_source_file_dependencies", [])
        if any(
            _matches_path_dependency(f, dep)
            and not any(_matches_path_dependency(f, exclude) for exclude in exclude_deps)
            for f in changed_files
            for dep in deps
        ):
            matched.append(module["name"])
    return matched


def _collect_test_dirs(
    module_names: list[str],
    config: list[dict],
) -> tuple[list[str], list[str]]:
    """Collect test paths (directories or files) for the given modules.

    Returns (normal_dirs, cpu_only_dirs). *cpu_only_dirs* are from modules
    with ``cpu_only: true`` and should skip NPU convention subdirectories.

    Deduplicates parent/child paths: if both ``a/b`` and ``a/b/c`` are
    present, only ``a/b`` is kept.
    """
    module_map = {m["name"]: m for m in config}
    normal: set[str] = set()
    cpu_only: set[str] = set()
    for name in module_names:
        mod = module_map[name]
        target = cpu_only if mod.get("cpu_only") else normal
        for path in mod.get("tests", []):
            target.add(path.rstrip("/"))
    normal_list = _dedup_paths(normal)
    cpu_only_list = _dedup_paths(cpu_only)
    # Remove cpu_only paths that are already covered by a normal parent path
    cpu_only_list = [p for p in cpu_only_list if not any(p.startswith(n + "/") for n in normal_list)]
    return normal_list, cpu_only_list


def _dedup_paths(paths: set[str]) -> list[str]:
    sorted_paths = sorted(paths)
    result: list[str] = []
    for path in sorted_paths:
        if not any(path != other and path.startswith(other + "/") for other in sorted_paths):
            result.append(path)
    return result


def _configured_nodeid_targets_for_file(file_path: str, config: list[dict]) -> list[str]:
    file_path = _as_posix_path(_pytest_node_file_path(file_path))
    targets: list[str] = []
    seen: set[str] = set()
    for module in config:
        for test_target in module.get("tests", []):
            test_target = test_target.rstrip("/")
            if "::" not in test_target:
                continue
            target_file = _as_posix_path(_pytest_node_file_path(test_target))
            if target_file == file_path and test_target not in seen:
                targets.append(test_target)
                seen.add(test_target)
    return targets


def _is_skipped_test_target(target: str, skip_tests: set[str]) -> bool:
    target = target.rstrip("/")
    return target in skip_tests or _pytest_node_file_path(target) in skip_tests


def _route_ut_dir(dir_path: str) -> RunnerKey:
    dir_path = _pytest_node_file_path(dir_path)
    normalized = dir_path if dir_path.endswith("/") else dir_path + "/"
    normalized = _as_posix_path(normalized)
    for pattern, npu_type, num_npus in _UT_DIR_PATTERNS:
        if pattern.search(normalized):
            return (num_npus, npu_type)
    return _DEFAULT_KEY


def _route_e2e_dir(dir_path: str) -> RunnerKey | None:
    dir_path = _as_posix_path(dir_path)
    for pattern, npu_type, num_npus in _E2E_DIR_PATTERNS:
        if pattern.search(dir_path):
            return (num_npus, npu_type)
    return None


def _is_ut_path(path: str) -> bool:
    return path == "tests/ut" or path.startswith("tests/ut/")


def _is_e2e_path(path: str) -> bool:
    return path == "tests/e2e" or path.startswith("tests/e2e/")


def _collect_explicit_test_targets(changed_files: list[str]) -> list[str]:
    """Return test paths passed directly via --changed-files.

    Git diffs normally list files, but the temporary CI override passes pytest
    paths by hand. Support files, directories and nodeids so those targets can
    be routed without needing a source-file module match.
    """
    targets: list[str] = []
    for changed_file in changed_files:
        if not (_is_ut_path(changed_file) or _is_e2e_path(changed_file)):
            continue

        path = Path(_pytest_node_file_path(changed_file))
        if "::" in changed_file:
            if path.exists():
                targets.append(changed_file.rstrip("/"))
            continue

        if path.is_dir() or (path.is_file() and path.name.startswith("test_")):
            targets.append(changed_file.rstrip("/"))

    return targets


def _scan_ut_test_dir(
    dir_path: str,
    groups: dict[RunnerKey, list[str]],
    cpu_only: bool = False,
) -> None:
    """Scan a UT directory and route tests by directory convention.

    Walks the directory tree. Each test file is routed individually based on
    its path — files under convention directories (e.g. ``a2/``, ``a3_2/``)
    go to the corresponding NPU runner, others go to the CPU group.

    If *cpu_only* is True, files under NPU convention directories are skipped.

    Always emits individual file paths to avoid test pollution when pytest
    runs a whole directory.
    """
    path = Path(_pytest_node_file_path(dir_path))
    if not path.exists():
        groups[_DEFAULT_KEY].append(dir_path)
        return

    if path.is_file():
        key = _route_ut_dir(dir_path)
        if cpu_only and key != _DEFAULT_KEY:
            return
        groups[key].append(dir_path)
        return

    for f in sorted(path.rglob("test_*.py")):
        if any(part in ("__pycache__",) for part in f.parts):
            continue
        key = _route_ut_dir(str(f))
        if cpu_only and key != _DEFAULT_KEY:
            continue
        groups[key].append(str(f))


def _scan_e2e_test_dir(
    dir_path: str,
    groups: dict[RunnerKey, list[str]],
) -> None:
    """Scan an E2E directory or single file and route by directory convention.

    *dir_path* may be either a directory (all ``test_*.py`` under it are
    collected) or a single test file.
    """
    path = Path(_pytest_node_file_path(dir_path))
    if not path.exists():
        return

    if path.is_file():
        key = _route_e2e_file(dir_path)
        if key is not None:
            groups[key].append(dir_path)
        else:
            print(
                f"Warning: E2E test file {dir_path} does not match any runner pattern, skipping.",
                file=sys.stderr,
            )
        return

    key = _route_e2e_dir(dir_path + "/")
    if key is not None:
        test_files = sorted(str(f) for f in path.rglob("test_*.py"))
        if test_files:
            for f in test_files:
                f_key = _route_e2e_file(f)
                if f_key is not None:
                    groups[f_key].append(f)
        return

    for entry in sorted(path.iterdir()):
        if entry.is_dir():
            sub_key = _route_e2e_dir(str(entry) + "/")
            if sub_key is not None:
                test_files = sorted(str(f) for f in entry.rglob("test_*.py"))
                if test_files:
                    for f in test_files:
                        f_key = _route_e2e_file(f)
                        if f_key is not None:
                            groups[f_key].append(f)
            else:
                _scan_e2e_test_dir(str(entry), groups)


def _dedup_groups(groups: dict[RunnerKey, list[str]]) -> None:
    for key in groups:
        seen: set[str] = set()
        deduped: list[str] = []
        for target in groups[key]:
            if target not in seen:
                deduped.append(target)
                seen.add(target)
        groups[key] = deduped


def _find_runner(
    num_npus: int,
    npu_type: NpuType,
    runners: list[RunnerInfo],
) -> RunnerInfo | None:
    if npu_type == NpuType.CPU:
        candidates = [r for r in runners if r.npu_type == NpuType.CPU]
    else:
        candidates = [r for r in runners if r.npu_type == npu_type and r.num_npus == num_npus]
    return candidates[0] if candidates else None


def _partition_by_estimated_time(
    targets: list[str],
    test_time_map: TestTimeMap,
    max_group_seconds: float,
) -> list[tuple[list[str], float]]:
    if not targets:
        return []
    if max_group_seconds <= 0:
        raise SystemExit("--max-group-seconds must be greater than 0")

    weighted_targets = [(target, _estimated_time_for_target(target, test_time_map)) for target in targets]
    for target, estimated_time in weighted_targets:
        if estimated_time > max_group_seconds:
            print(
                f"Warning: {target} estimated time {estimated_time:.1f}s exceeds "
                f"the shard threshold {max_group_seconds:.1f}s and cannot be split further.",
                file=sys.stderr,
            )

    total_estimated_time = sum(estimated_time for _, estimated_time in weighted_targets)
    min_shard_count = max(1, math.ceil(total_estimated_time / max_group_seconds))
    max_shard_count = len(weighted_targets)

    indexed = list(enumerate(weighted_targets))
    indexed.sort(key=lambda item: (-item[1][1], item[0]))

    best_buckets: list[list[int]] = []
    best_sums: list[float] = []
    for shard_count in range(min(min_shard_count, max_shard_count), max_shard_count + 1):
        buckets: list[list[int]] = [[] for _ in range(shard_count)]
        sums = [0.0] * shard_count
        for original_index, (_, estimated_time) in indexed:
            lightest = sums.index(min(sums))
            buckets[lightest].append(original_index)
            sums[lightest] += estimated_time

        best_buckets = buckets
        best_sums = sums
        if max(sums) <= max_group_seconds or shard_count == max_shard_count:
            break

    partitions: list[tuple[list[str], float]] = []
    for bucket, estimated_time in zip(best_buckets, best_sums):
        if not bucket:
            continue
        bucket_targets = [weighted_targets[index][0] for index in sorted(bucket)]
        partitions.append((bucket_targets, estimated_time))
    return partitions


def _runner_group(
    num_npus: int,
    npu_type: NpuType,
    runner: RunnerInfo,
    tests: list[str],
    shard_id: int = 0,
    shard_count: int = 1,
    estimated_time: float | None = None,
) -> dict:
    shard_label = f"{shard_id + 1}of{shard_count}"
    group: dict = {
        "num_npus": num_npus,
        "npu_type": npu_type.value,
        "runner": runner.label,
        "tests": " ".join(tests),
        "shard_id": shard_id,
        "shard_count": shard_count,
        "shard_label": shard_label,
        "shard_suffix": f"-shard-{shard_label}" if shard_count > 1 else "",
    }
    if estimated_time is not None:
        group["estimated_time"] = round(estimated_time, 1)
    if runner.image_tag:
        group["image_tag"] = runner.image_tag
    return group


def _resolve_to_runners(
    all_groups: dict[RunnerKey, list[str]],
    runners: list[RunnerInfo],
    enable_time_sharding: bool = False,
    test_time_map: TestTimeMap | None = None,
    max_group_seconds: float = 3600.0,
) -> list[dict]:
    result: list[dict] = []
    errors: list[str] = []

    for (num_npus, npu_type), tests in sorted(all_groups.items()):
        if not tests:
            continue
        runner = _find_runner(num_npus, npu_type, runners)
        if runner is None:
            available = [f"{r.label} ({r.npu_type.value} x{r.num_npus})" for r in runners if r.npu_type == npu_type]
            header = f"\n  Runner key ({npu_type.value} x{num_npus}) -- no runner available."
            runners_line = (
                f"\n    Available {npu_type.value} runners: {', '.join(available)}"
                if available
                else f'\n    No runners defined for chip type "{npu_type.value}".'
            )
            tests_line = "\n    Affected tests:\n" + "\n".join(f"      - {t}" for t in sorted(tests))
            errors.append(header + runners_line + tests_line)
            continue

        sorted_tests = sorted(tests)
        if enable_time_sharding and npu_type != NpuType.CPU:
            if test_time_map is None:
                raise SystemExit("--enable-time-sharding requires --test-time-map")
            partitions = _partition_by_estimated_time(sorted_tests, test_time_map, max_group_seconds)
            shard_count = len(partitions)
            for shard_id, (partition_tests, estimated_time) in enumerate(partitions):
                result.append(
                    _runner_group(
                        num_npus,
                        npu_type,
                        runner,
                        partition_tests,
                        shard_id=shard_id,
                        shard_count=shard_count,
                        estimated_time=estimated_time,
                    )
                )
            continue

        result.append(_runner_group(num_npus, npu_type, runner, sorted_tests))

    if errors:
        print(
            "\nERROR: The following test groups cannot be routed to any runner"
            " in runner_label.json:" + "".join(errors) + "\n\nPlease fix the directory structure or add the"
            " missing runner to runner_label.json.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return result


def _write_output(
    test_groups: list[dict],
    matched_modules: list[str],
) -> None:
    has_tests = len(test_groups) > 0
    groups_json = json.dumps(test_groups, separators=(",", ":"))

    outputs = {
        "test_groups": groups_json,
        "has_tests": str(has_tests).lower(),
        "matched_modules": ",".join(matched_modules),
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")

    _print_summary(test_groups, matched_modules, has_tests)


def _print_summary(
    test_groups: list[dict],
    matched_modules: list[str],
    has_tests: bool,
) -> None:
    divider = "=" * 60
    print(f"\n{divider}", file=sys.stderr)
    print("Selective Test Scope Summary", file=sys.stderr)
    print(divider, file=sys.stderr)
    print(f"Matched modules: {matched_modules or '(none)'}", file=sys.stderr)
    print(f"Has tests to run: {has_tests}", file=sys.stderr)

    for group in test_groups:
        npu_type = group["npu_type"]
        num_npus = group["num_npus"]
        runner = group["runner"]
        tests = group["tests"].split()
        shard_suffix = group.get("shard_suffix", "")
        estimated_time = group.get("estimated_time")
        estimate_text = f", est {estimated_time:.1f}s" if isinstance(estimated_time, (float, int)) else ""
        if npu_type == "cpu":
            header = f"### CPU{shard_suffix} ({len(tests)} tests{estimate_text}) -> `{runner}`"
        else:
            header = (
                f"### {npu_type.upper()} x{num_npus}{shard_suffix} ({len(tests)} tests{estimate_text}) -> `{runner}`"
            )
        print(f"\n  {header}", file=sys.stderr)
        for t in tests:
            print(f"    - {t}", file=sys.stderr)

    print(f"{divider}\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Determine test scope based on changed files",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--changed-files",
        nargs="+",
        help="List of changed file paths",
    )
    input_group.add_argument(
        "--diff-base",
        type=str,
        help="Git ref to diff against (e.g. origin/main)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to test_config.yaml",
    )
    parser.add_argument(
        "--run-all-modules",
        action="store_true",
        help="Run tests for all configured modules regardless of changed files",
    )
    parser.add_argument(
        "--skip-default-cpu-ut",
        action="store_true",
        help="Skip the always-on default_cpu_ut module.",
    )
    parser.add_argument(
        "--enable-time-sharding",
        action="store_true",
        help="Split non-CPU runner groups by estimated runtime.",
    )
    parser.add_argument(
        "--test-time-map",
        type=Path,
        default=_DEFAULT_TEST_TIME_MAP_PATH,
        help="YAML map of pytest target to estimated seconds.",
    )
    parser.add_argument(
        "--max-group-seconds",
        type=float,
        default=3600.0,
        help="Maximum target seconds for each time-sharded group.",
    )

    args = parser.parse_args()
    config = _resolve_config_inheritance(yaml.safe_load(args.config.read_text()))

    changed_files = _get_changed_files(args.diff_base) if args.diff_base else args.changed_files
    matched_modules = (
        [module["name"] for module in config] if args.run_all_modules else _match_modules(changed_files, config)
    )
    if args.skip_default_cpu_ut:
        matched_modules = [name for name in matched_modules if name != _DEFAULT_CPU_UT_MODULE]
    test_dirs, cpu_only_dirs = _collect_test_dirs(matched_modules, config)

    skip_tests: set[str] = set()
    for module in config:
        for s in module.get("skip_tests", []):
            skip_tests.add(s.rstrip("/"))

    explicit_test_targets = _collect_explicit_test_targets(changed_files)

    ut_dirs = [d for d in test_dirs if _is_ut_path(d)]
    cpu_only_ut_dirs = [d for d in cpu_only_dirs if _is_ut_path(d)]
    e2e_dirs = [d for d in test_dirs if _is_e2e_path(d)]

    all_groups: dict[RunnerKey, list[str]] = defaultdict(list)

    for dir_path in ut_dirs:
        p = Path(_pytest_node_file_path(dir_path))
        if p.is_file():
            key = _route_ut_dir(dir_path)
            all_groups[key].append(dir_path)
        else:
            _scan_ut_test_dir(dir_path, all_groups)
    for dir_path in cpu_only_ut_dirs:
        p = Path(_pytest_node_file_path(dir_path))
        if p.is_file():
            key = _route_ut_dir(dir_path)
            if key == _DEFAULT_KEY:
                all_groups[key].append(dir_path)
        else:
            _scan_ut_test_dir(dir_path, all_groups, cpu_only=True)

    for dir_path in e2e_dirs:
        _scan_e2e_test_dir(dir_path, all_groups)

    for explicit_test_target in explicit_test_targets:
        target_path = Path(_pytest_node_file_path(explicit_test_target))
        if "::" in explicit_test_target or target_path.is_dir():
            changed_targets = [explicit_test_target]
        else:
            changed_targets = _configured_nodeid_targets_for_file(explicit_test_target, config) or [
                explicit_test_target
            ]
        for f in changed_targets:
            if _is_skipped_test_target(f, skip_tests):
                continue
            if _is_ut_path(f):
                p = Path(_pytest_node_file_path(f))
                if p.is_dir():
                    _scan_ut_test_dir(f, all_groups)
                else:
                    key = _route_ut_dir(f)
                    all_groups[key].append(f)
            elif _is_e2e_path(f):
                _scan_e2e_test_dir(f, all_groups)

    _dedup_groups(all_groups)

    if skip_tests:
        for key in list(all_groups.keys()):
            filtered: list[str] = []
            for t in all_groups[key]:
                if _is_skipped_test_target(t, skip_tests):
                    continue
                p = Path(_pytest_node_file_path(t))
                if p.is_dir():
                    sub = [
                        str(f) for f in sorted(p.rglob("test_*.py")) if not _is_skipped_test_target(str(f), skip_tests)
                    ]
                    if sub:
                        filtered.extend(sub)
                else:
                    filtered.append(t)
            all_groups[key] = filtered
        _dedup_groups(all_groups)

    runners = _load_runners()
    test_time_map = _load_test_time_map(args.test_time_map) if args.enable_time_sharding else None
    test_groups = _resolve_to_runners(
        all_groups,
        runners,
        enable_time_sharding=args.enable_time_sharding,
        test_time_map=test_time_map,
        max_group_seconds=args.max_group_seconds,
    )

    _write_output(test_groups, matched_modules)


if __name__ == "__main__":
    main()
