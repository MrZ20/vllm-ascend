# SPDX-License-Identifier: Apache-2.0
"""Resolve the independent V2 matrix without importing model/runtime code."""

import argparse
import json
import os
import re
from pathlib import Path

import yaml

CATEGORIES = ("single_node", "double_node", "multi_node")


def positive(value, name):
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def resolve(matrix_path: Path, repo: Path, soc: str, selected: str = "all") -> dict:
    raw = yaml.safe_load(matrix_path.read_text())
    if soc not in raw:
        raise ValueError(f"No V2 matrix for {soc}")
    platform = raw[soc]
    if set(platform) - set(CATEGORIES):
        raise ValueError(f"Unknown V2 matrix categories: {set(platform) - set(CATEGORIES)}")
    requested = None if selected.strip() == "all" else set(filter(None, map(str.strip, selected.split(","))))
    if requested == set():
        raise ValueError("Select all or a comma-separated list of test names")
    seen, result = set(), {}
    for category in CATEGORIES:
        group = platform.get(category, {"test_config": []})
        if set(group) != {"test_config"} or not isinstance(group["test_config"], list):
            raise ValueError(f"{category} requires test_config: []")
        entries = []
        for item in group["test_config"]:
            if set(item) - {"name", "config_path", "num_nodes", "npu_per_node", "os"}:
                raise ValueError(f"Unknown fields in matrix entry {item}")
            name = item["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in seen:
                raise ValueError(f"Invalid or duplicate test name {name!r}")
            seen.add(name)
            nodes = positive(item["num_nodes"], "num_nodes")
            if (category == "single_node" and nodes != 1) or (category == "double_node" and nodes != 2):
                raise ValueError(f"{name}: num_nodes conflicts with {category}")
            if category == "multi_node" and nodes < 3:
                raise ValueError(f"{name}: multi_node requires at least three nodes")
            config = Path(item["config_path"])
            if config.is_absolute() or not (repo / config).resolve().is_relative_to(repo.resolve()):
                raise ValueError("config_path must be repository-relative")
            if not (repo / config).is_file():
                raise ValueError(f"Missing model YAML: {config}")
            if category == "single_node" and not item.get("os"):
                raise ValueError(f"{name}: direct runner requires os")
            entry = {**item, "num_nodes": nodes, "npu_per_node": ""}
            if "npu_per_node" in item:
                entry["npu_per_node"] = str(positive(item["npu_per_node"], "npu_per_node"))
            if requested is None or name in requested:
                entries.append(entry)
        result[category] = entries
    if requested is not None and requested - seen:
        raise ValueError(f"Unknown test names: {sorted(requested - seen)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--soc", required=True)
    parser.add_argument("--select", default="all")
    args = parser.parse_args()
    result = resolve(args.matrix, args.repo, args.soc, args.select)
    print(json.dumps(result, indent=2))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={json.dumps(value)}\n")


if __name__ == "__main__":
    main()
