#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""
Analyze PR changes and recommend test cases based on mapping.json.

Usage:
    python tools/tia/analyze_changes.py \
        --mapping mapping.json \
        --base-ref origin/main \
        --output recommended_tests.json

This script:
1. Gets changed files from git diff between base-ref and HEAD.
2. For C++/build file changes, recommends ALL_TESTS.
3. For Python source changes, uses AST to identify changed functions.
4. Looks up affected tests in the mapping.
5. Outputs a recommended test list.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from ast_parser import find_function_for_line, is_utility_file, parse_source_functions


# File patterns that trigger ALL_TESTS
ALL_TESTS_PATTERNS = [
    r"^csrc/",
    r"^CMakeLists\.txt$",
    r"^setup\.py$",
    r"^pyproject\.toml$",
    r"^requirements.*\.txt$",
]

# File patterns to completely ignore (docs, CI config, etc.)
IGNORE_PATTERNS = [
    r"^docs/",
    r"^\.github/",
    r"^\.gitignore$",
    r"^\.gitmodules$",
    r"^\.pre-commit-config\.yaml$",
    r"^\.readthedocs\.yaml$",
    r"^\.markdownlint\.yaml$",
    r"^\.ruff_cache/",
    r"^\.gemini/",
    r"^typos\.toml$",
    r"^codecov\.yml$",
    r"^actionlint$",
    r"^format\.sh$",
    r"^LICENSE$",
    r"^CODE_OF_CONDUCT\.md$",
    r"^CONTRIBUTING\.md$",
    r"^DCO$",
    r"^README",
    r"^Dockerfile",
    r"^collect_env\.py$",
    r"^tools/(?!tia/)",
    r"^benchmarks/",
]


def get_changed_files(base_ref: str) -> list[str]:
    """
    Get list of changed files between base_ref and HEAD using git diff.

    Returns list of relative file paths.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return files
    except subprocess.CalledProcessError as e:
        print(f"WARNING: git diff failed: {e.stderr}")
        # Fallback: try with merge-base
        try:
            merge_base = subprocess.run(
                ["git", "merge-base", base_ref, "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMR", merge_base, "HEAD"],
                capture_output=True, text=True, check=True,
            )
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return files
        except subprocess.CalledProcessError as e2:
            print(f"ERROR: git diff fallback also failed: {e2.stderr}")
            return []


def get_changed_lines(base_ref: str, file_path: str) -> list[int]:
    """
    Get the specific line numbers changed in a file (in the new version).

    Parses unified diff output to extract added/modified line numbers.
    """
    try:
        # Use merge-base for accurate diff
        try:
            merge_base = subprocess.run(
                ["git", "merge-base", base_ref, "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            merge_base = base_ref

        result = subprocess.run(
            ["git", "diff", "-U0", merge_base, "HEAD", "--", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    changed_lines = []
    # Parse unified diff hunk headers: @@ -old_start,old_count +new_start,new_count @@
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in result.stdout.split("\n"):
        match = hunk_pattern.match(line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            # count == 0 means deletion at this point, no new lines
            if count > 0:
                changed_lines.extend(range(start, start + count))

    return changed_lines


def should_ignore_file(file_path: str) -> bool:
    """Check if a file should be completely ignored for TIA."""
    for pattern in IGNORE_PATTERNS:
        if re.match(pattern, file_path):
            return True
    return False


def triggers_all_tests(file_path: str) -> bool:
    """Check if changing this file should trigger all tests."""
    for pattern in ALL_TESTS_PATTERNS:
        if re.match(pattern, file_path):
            return True
    return False


def is_test_file(file_path: str) -> bool:
    """Check if this is a test file."""
    normalized = file_path.replace("\\", "/")
    return normalized.startswith("tests/") or "/tests/" in normalized


def analyze_changes(
    base_ref: str,
    mapping: dict,
    project_root: str = ".",
) -> dict:
    """
    Analyze PR changes and return recommended tests.

    Returns:
        Dictionary with:
        - "all_tests": bool - whether all tests should run
        - "recommended_tests": list[str] - specific test identifiers
        - "changed_test_files": list[str] - test files that were directly changed
        - "analysis": dict - detailed analysis for debugging
    """
    changed_files = get_changed_files(base_ref)
    print(f"Found {len(changed_files)} changed files")

    file_mapping = mapping.get("file_mapping", {})
    func_mapping = mapping.get("func_mapping", {})

    recommended_tests: set[str] = set()
    changed_test_files: list[str] = []
    unmapped_files: list[str] = []
    unmapped_functions: list[str] = []
    analysis: dict[str, list[str]] = defaultdict(list)

    for file_path in changed_files:
        # Skip ignored files
        if should_ignore_file(file_path):
            analysis["ignored"].append(file_path)
            continue

        # C++/build files -> ALL_TESTS
        if triggers_all_tests(file_path):
            print(f"  ALL_TESTS triggered by: {file_path}")
            return {
                "all_tests": True,
                "recommended_tests": ["ALL_TESTS"],
                "changed_test_files": [],
                "trigger_file": file_path,
                "analysis": {"all_tests_trigger": [file_path]},
            }

        # Test files: include them directly
        if is_test_file(file_path):
            changed_test_files.append(file_path)
            # Also extract the test module path for recommendation
            # e.g., tests/e2e/singlecard/test_models.py -> tests/e2e/singlecard/test_models.py
            recommended_tests.add(file_path)
            analysis["changed_tests"].append(file_path)
            continue

        # Non-Python files that aren't caught above
        if not file_path.endswith(".py"):
            analysis["non_python_skipped"].append(file_path)
            continue

        # Python source files: analyze at appropriate granularity
        abs_path = os.path.join(project_root, file_path)

        # Check file-level mapping first
        if is_utility_file(file_path):
            if file_path in file_mapping:
                tests = file_mapping[file_path]
                recommended_tests.update(tests)
                analysis["file_level_mapped"].append(
                    f"{file_path} -> {len(tests)} tests"
                )
            else:
                unmapped_files.append(file_path)
                analysis["unmapped_file_level"].append(file_path)
            continue

        # Function-level analysis
        if not os.path.exists(abs_path):
            # New file or deleted file
            if file_path in file_mapping:
                recommended_tests.update(file_mapping[file_path])
                analysis["new_file_file_mapped"].append(file_path)
            else:
                unmapped_files.append(file_path)
                analysis["new_or_missing_file"].append(file_path)
            continue

        changed_lines = get_changed_lines(base_ref, file_path)
        if not changed_lines:
            analysis["no_changed_lines"].append(file_path)
            continue

        # Parse current version of the file with AST
        functions = parse_source_functions(abs_path)

        changed_functions: set[str] = set()
        orphan_lines: list[int] = []

        for line_no in changed_lines:
            func_name = find_function_for_line(functions, line_no)
            if func_name:
                changed_functions.add(func_name)
            else:
                orphan_lines.append(line_no)

        # Look up each changed function in func_mapping
        file_had_mapping = False
        for func_name in changed_functions:
            func_key = f"{file_path}::{func_name}"
            if func_key in func_mapping:
                tests = func_mapping[func_key]
                recommended_tests.update(tests)
                file_had_mapping = True
                analysis["func_level_mapped"].append(
                    f"{func_key} -> {len(tests)} tests"
                )
            else:
                unmapped_functions.append(func_key)
                analysis["unmapped_functions"].append(func_key)

        # Handle orphan lines (module-level changes) via file mapping
        if orphan_lines:
            if file_path in file_mapping:
                recommended_tests.update(file_mapping[file_path])
                analysis["module_level_mapped"].append(
                    f"{file_path} lines {orphan_lines[:5]}..."
                )
            elif not file_had_mapping:
                unmapped_files.append(file_path)
                analysis["module_level_unmapped"].append(file_path)

    # Report unmapped items
    if unmapped_files:
        print(f"\n  WARNING: {len(unmapped_files)} files have no mapping:")
        for f in unmapped_files[:10]:
            print(f"    - {f}")

    if unmapped_functions:
        print(f"\n  WARNING: {len(unmapped_functions)} functions have no mapping:")
        for f in unmapped_functions[:10]:
            print(f"    - {f}")

    return {
        "all_tests": False,
        "recommended_tests": sorted(recommended_tests),
        "changed_test_files": sorted(changed_test_files),
        "unmapped_files": sorted(set(unmapped_files)),
        "unmapped_functions": sorted(set(unmapped_functions)),
        "analysis": {k: v for k, v in analysis.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PR changes and recommend test cases"
    )
    parser.add_argument(
        "--mapping",
        default="mapping.json",
        help="Path to mapping.json (default: mapping.json)",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git base reference (default: origin/main)",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--output",
        default="recommended_tests.json",
        help="Output JSON file for recommended tests",
    )

    args = parser.parse_args()

    # Load mapping
    if not os.path.exists(args.mapping):
        print(f"WARNING: Mapping file not found: {args.mapping}")
        print("Falling back to ALL_TESTS")
        result = {
            "all_tests": True,
            "recommended_tests": ["ALL_TESTS"],
            "reason": "mapping_not_found",
        }
    else:
        with open(args.mapping, "r") as f:
            mapping = json.load(f)

        print(f"Loaded mapping: {len(mapping.get('file_mapping', {}))} file mappings, "
              f"{len(mapping.get('func_mapping', {}))} function mappings")

        result = analyze_changes(args.base_ref, mapping, args.project_root)

    # Output results
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TIA ANALYSIS RESULT")
    print("=" * 60)

    if result.get("all_tests"):
        print(f"Decision: RUN ALL TESTS")
        if "trigger_file" in result:
            print(f"Reason: Change in {result['trigger_file']}")
        elif "reason" in result:
            print(f"Reason: {result['reason']}")
    else:
        tests = result.get("recommended_tests", [])
        print(f"Decision: RUN {len(tests)} SELECTED TESTS")

        if tests:
            print("\nRecommended tests:")
            for t in tests[:30]:
                print(f"  - {t}")
            if len(tests) > 30:
                print(f"  ... and {len(tests) - 30} more")

        changed = result.get("changed_test_files", [])
        if changed:
            print(f"\nDirectly changed test files ({len(changed)}):")
            for t in changed:
                print(f"  - {t}")

        unmapped = result.get("unmapped_files", [])
        if unmapped:
            print(f"\nUnmapped source files ({len(unmapped)}):")
            for f in unmapped[:10]:
                print(f"  ⚠ {f}")

    print("=" * 60)
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
