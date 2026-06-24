#!/usr/bin/env python3
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
"""Parse schedule_config.yaml and output periodic test matrices."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PeriodicCase:
    name: str
    config_path: str
    test_directory: str  # "model" | "accuracy" | "ops"
    framework: str  # "single_node" | "multi_node" | "accuracy" | "ops"
    device_type: str  # "a2" | "a3" | "310p"
    device_scale: str  # "card" | "node"
    device_num: int
    runner: str
    group: str


RESOURCE_DIRS: dict[str, tuple[str, int]] = {
    "one_card": ("card", 1),
    "two_card": ("card", 2),
    "four_card": ("card", 4),
    "eight_card": ("card", 8),
    "one_node": ("node", 1),
    "two_node": ("node", 2),
    "four_node": ("node", 4),
}
RESOURCE_NAME_BY_VALUE = {value: key for key, value in RESOURCE_DIRS.items()}

RUNNER_MAPPING_TABLE: dict[tuple[str, str, int], str] = {
    ("a2", "card", 1): "linux-aarch64-a2b3-1",
    ("a2", "card", 2): "linux-aarch64-a2b3-2",
    ("a2", "card", 4): "linux-aarch64-a2b3-4",
    ("a2", "card", 8): "linux-aarch64-a2b3-8",
    ("a2", "node", 1): "linux-aarch64-a2b3-8",
    ("a2", "node", 2): "linux-amd64-cpu-8-hk",
    ("a2", "node", 4): "linux-amd64-cpu-8-hk",
    ("a3", "card", 1): "linux-aarch64-a3-2",
    ("a3", "card", 2): "linux-aarch64-a3-2",
    ("a3", "card", 4): "linux-aarch64-a3-4",
    ("a3", "card", 8): "linux-aarch64-a3-8",
    ("a3", "node", 1): "linux-aarch64-a3-16",
    ("a3", "node", 2): "linux-aarch64-a3-0",
    ("a3", "node", 4): "linux-aarch64-a3-0",
    ("310p", "card", 1): "linux-aarch64-310p-2",
    ("310p", "card", 2): "linux-aarch64-310p-2",
    ("310p", "card", 4): "linux-aarch64-310p-4",
}

CHIP_ORDER = {"a2": 0, "a3": 1, "310p": 2}
CHIP_310P = re.compile(r"(?<![A-Za-z0-9])v?310p?(?![A-Za-z0-9])")
CHIP_A2 = re.compile(r"(?<![A-Za-z0-9])[Aa]2(?![A-Za-z0-9])")
CHIP_A3 = re.compile(r"(?<![A-Za-z0-9])[Aa]3(?![A-Za-z0-9])")


def normalize_path(raw_path: Any) -> str:
    return str(raw_path).strip().replace("\\", "/").rstrip("/")


def list_files(raw_path: Any, patterns: tuple[str, ...]) -> list[str]:
    path = normalize_path(raw_path)
    last_part = path.rsplit("/", 1)[-1]
    if not str(raw_path).rstrip().endswith("/") and not Path(path).is_dir() and "." in last_part:
        return [path]

    if not Path(path).is_dir():
        return []

    files: set[str] = set()
    for pattern in patterns:
        files.update(str(file_path).replace("\\", "/") for file_path in Path(path).rglob(pattern))
    return sorted(files)


def make_case(
    config_path: str, test_directory: str, framework: str, group: str, default_device_type: str = "a3"
) -> PeriodicCase:
    resource_dir = next(resource for resource in RESOURCE_DIRS if resource in config_path)
    device_scale, device_num = RESOURCE_DIRS[resource_dir]

    def select_device_type(default_device: str = "a3") -> str:
        if CHIP_310P.search(config_path):
            device_type = "310p"
        elif CHIP_A2.search(config_path):
            device_type = "a2"
        elif CHIP_A3.search(config_path):
            device_type = "a3"
        else:
            device_type = default_device
        return device_type

    device_type = select_device_type(default_device_type)

    file_name = config_path.rsplit("/", 1)[-1]
    name = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    runner = RUNNER_MAPPING_TABLE.get((device_type, device_scale, device_num), "")

    return PeriodicCase(
        name=name,
        config_path=config_path,
        test_directory=test_directory,
        framework=framework,
        device_type=device_type,
        device_scale=device_scale,
        device_num=device_num,
        runner=runner,
        group=group,
    )


def single_node_framework(raw_path: Any, group: str, single_node_list: list[PeriodicCase]) -> bool:
    config_path = normalize_path(raw_path)
    if "model" not in config_path:
        return False

    resource_dir = next(resource for resource in RESOURCE_DIRS if resource in config_path)
    device_scale, device_num = RESOURCE_DIRS[resource_dir]
    if not (device_scale == "card" or (device_scale == "node" and device_num == 1)):
        return False

    single_node_list.append(make_case(config_path, "model", "single_node", group))
    return True


def multi_node_framework(raw_path: Any, group: str, multi_node_list: list[PeriodicCase]) -> bool:
    config_path = normalize_path(raw_path)
    if "model" not in config_path:
        return False

    resource_dir = next(resource for resource in RESOURCE_DIRS if resource in config_path)
    device_scale, device_num = RESOURCE_DIRS[resource_dir]
    if not (device_scale == "node" and device_num > 1):
        return False

    multi_node_list.append(make_case(config_path, "model", "multi_node", group))
    return True


def ops_framework(raw_path: Any, group: str, ops_list: list[PeriodicCase]) -> bool:
    config_path = normalize_path(raw_path)
    if "ops" not in config_path:
        return False

    ops_list.append(make_case(config_path, "ops", "ops", group))
    return True


def accuracy_framework(raw_path: Any, group: str, accuracy_list: list[PeriodicCase]) -> bool:
    config_path = normalize_path(raw_path)
    if "accuracy" not in config_path:
        return False

    accuracy_list.append(make_case(config_path, "accuracy", "accuracy", group))
    return True


def matches_filter(case: PeriodicCase, test_filter: str) -> bool:
    filters = [item.strip() for item in test_filter.split(",") if item.strip()] or ["all"]
    if "all" in filters:
        return True

    item_values = [
        case.name,
        case.config_path,
        case.test_directory,
        case.framework,
        case.device_type,
        case.device_scale,
        str(case.device_num),
        case.runner,
        case.group,
    ]
    for filter_text in filters:
        for value in item_values:
            if filter_text == value or filter_text in value:
                return True
    return False


def filter_cases(cases: list[PeriodicCase], test_filter: str) -> list[PeriodicCase]:
    filtered_cases = []
    seen_cases: set[tuple[str, str, str]] = set()
    for case in cases:
        if not matches_filter(case, test_filter):
            continue

        case_key = (case.framework, case.config_path, case.group)
        if case_key in seen_cases:
            continue

        seen_cases.add(case_key)
        filtered_cases.append(case)
    return filtered_cases


def build_single_node_matrix(single_node_list: list[PeriodicCase]) -> list[dict[str, Any]]:
    return [
        {
            "name": case.name,
            "chip": case.device_type,
            "runner": case.runner,
            "config_path": case.config_path,
            "tests": "",
            "extra_components": False,
            "group": case.group,
        }
        for case in single_node_list
    ]


def build_multi_node_matrix(multi_node_list: list[PeriodicCase]) -> list[dict[str, Any]]:
    multi_node_matrix = [
        {
            "name": case.name,
            "chip": case.device_type,
            "runner": case.runner,
            "config_path": case.config_path,
            "extra_components": False,
            "size": case.device_num,
            "group": case.group,
        }
        for case in multi_node_list
    ]
    return sorted(multi_node_matrix, key=lambda item: -int(item["size"]))


def build_accuracy_matrix(accuracy_list: list[PeriodicCase]) -> list[dict[str, Any]]:
    # Accuracy configs sharing the same group/chip/resource run together. This
    # replaces separate lm_eval grouping files with path-routed config lists.
    grouped_cases: OrderedDict[tuple[str, str, str, int, str], list[PeriodicCase]] = OrderedDict()
    for case in accuracy_list:
        key = (case.group, case.device_type, case.device_scale, case.device_num, case.runner)
        grouped_cases.setdefault(key, []).append(case)

    accuracy_matrix = []
    for (group_name, device_type, device_scale, device_num, runner), cases in grouped_cases.items():
        config_paths = list(OrderedDict((case.config_path, None) for case in cases))
        resource_name = RESOURCE_NAME_BY_VALUE.get((device_scale, device_num), f"{device_num}_{device_scale}")
        accuracy_matrix.append(
            {
                "name": cases[0].name if len(config_paths) == 1 else f"{resource_name}-{device_type}",
                "chip": device_type,
                "runner": runner,
                "config_paths": config_paths,
                "group": group_name,
            }
        )
    return accuracy_matrix


def build_ops_matrix(ops_list: list[PeriodicCase]) -> list[dict[str, Any]]:
    # Ops tests are grouped by the resource marker after all paths have been
    # collected, so framework matching remains a simple case collector.
    grouped_cases: OrderedDict[tuple[str, str, str, int, str, str], list[PeriodicCase]] = OrderedDict()
    for case in ops_list:
        resource_dir = next(resource for resource in RESOURCE_DIRS if resource in case.config_path)
        key = (case.group, case.device_type, case.device_scale, case.device_num, case.runner, resource_dir)
        grouped_cases.setdefault(key, []).append(case)

    ops_matrix = []
    for (group_name, device_type, _device_scale, _device_num, runner, resource_dir), cases in grouped_cases.items():
        test_paths = list(OrderedDict((case.config_path, None) for case in cases))
        ops_matrix.append(
            {
                "name": cases[0].name if len(test_paths) == 1 else f"{resource_dir}-{device_type}",
                "chip": device_type,
                "runner": runner,
                "tests": " ".join(test_paths),
                "group": group_name,
            }
        )
    return ops_matrix


def select_schedule_sections(config: dict[str, Any], event_name: str, cron: str, group: str) -> list[dict[str, Any]]:
    schedule_sections = config.get("periodic_tests", [])
    if event_name == "schedule" and cron:
        return [section for section in schedule_sections if section["cron"] == cron]
    if group and group != "manual":
        return [section for section in schedule_sections if section["group"] == group]
    return schedule_sections


def write_outputs(
    warnings: list[str],
    single_node_matrix: list[dict[str, Any]],
    multi_node_matrix: list[dict[str, Any]],
    accuracy_matrix: list[dict[str, Any]],
    ops_matrix: list[dict[str, Any]],
) -> None:
    all_items = [
        item for matrix in (single_node_matrix, multi_node_matrix, accuracy_matrix, ops_matrix) for item in matrix
    ]
    image_build_targets = sorted({item["chip"] for item in all_items}, key=lambda chip: CHIP_ORDER.get(chip, 99))

    if warnings:
        print("WARNING: unrecognized schedule paths:", file=sys.stderr)
        print("\n".join(warnings), file=sys.stderr)
        print(file=sys.stderr)

    print("=== Selected periodic test cases ===", file=sys.stderr)
    for item in all_items:
        path_or_tests = item.get("config_path") or item.get("tests") or item.get("config_paths")
        print(
            f"  [{item['group']}] [{item['chip']}] [{item.get('runner', 'workflow-default')}] "
            f"{item['name']} ({path_or_tests})",
            file=sys.stderr,
        )
    print(
        "\nTotals: "
        f"{len(single_node_matrix)} single-node, "
        f"{len(multi_node_matrix)} multi-node, "
        f"{len(accuracy_matrix)} accuracy, "
        f"{len(ops_matrix)} ops",
        file=sys.stderr,
    )

    output_lines = [
        f"single_node_matrix={json.dumps(single_node_matrix)}",
        f"multi_node_matrix={json.dumps(multi_node_matrix)}",
        f"accuracy_matrix={json.dumps(accuracy_matrix)}",
        f"ops_matrix={json.dumps(ops_matrix)}",
        f"image_build_targets={json.dumps(image_build_targets)}",
    ]

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write("\n".join(output_lines) + "\n")
    else:
        print("\n=== Outputs ===")
        for name, matrix in (
            ("single_node_matrix", single_node_matrix),
            ("multi_node_matrix", multi_node_matrix),
            ("accuracy_matrix", accuracy_matrix),
            ("ops_matrix", ops_matrix),
        ):
            print(f"\n{name} ({len(matrix)} entries):")
            print(json.dumps(matrix, indent=2))
        print(f"\nimage_build_targets: {json.dumps(image_build_targets)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to schedule_config.yaml")
    parser.add_argument("--event-name", default="workflow_dispatch")
    parser.add_argument("--cron", default="")
    parser.add_argument("--group", default="")
    parser.add_argument("--test-filter", default="all")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    single_node_list: list[PeriodicCase] = []
    multi_node_list: list[PeriodicCase] = []
    accuracy_list: list[PeriodicCase] = []
    ops_list: list[PeriodicCase] = []
    warnings: list[str] = []

    for section in select_schedule_sections(config, args.event_name, args.cron, args.group):
        group = section["group"]
        expanded_files = [
            expanded_path
            for raw_path in section.get("files", [])
            for expanded_path in list_files(raw_path, ("*.yaml", "*.yml", "test_*.py"))
        ]

        for raw_path in expanded_files:
            config_path = normalize_path(raw_path)

            if single_node_framework(raw_path, group, single_node_list):
                continue
            if multi_node_framework(raw_path, group, multi_node_list):
                continue
            if ops_framework(raw_path, group, ops_list):
                continue
            if accuracy_framework(raw_path, group, accuracy_list):
                continue

            warnings.append(config_path)

    single_node_list = filter_cases(single_node_list, args.test_filter)
    multi_node_list = filter_cases(multi_node_list, args.test_filter)
    accuracy_list = filter_cases(accuracy_list, args.test_filter)
    ops_list = filter_cases(ops_list, args.test_filter)

    single_node_matrix = build_single_node_matrix(single_node_list)
    multi_node_matrix = build_multi_node_matrix(multi_node_list)
    accuracy_matrix = build_accuracy_matrix(accuracy_list)
    ops_matrix = build_ops_matrix(ops_list)

    write_outputs(warnings, single_node_matrix, multi_node_matrix, accuracy_matrix, ops_matrix)


if __name__ == "__main__":
    main()
