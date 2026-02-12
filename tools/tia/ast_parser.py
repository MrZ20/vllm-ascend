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
AST parser for extracting top-level function and class method ranges.

Used by both generate_mapping.py and analyze_changes.py to map
line numbers to function-level identifiers.
"""

import ast
import os
from typing import Optional


def parse_source_functions(file_path: str) -> dict[tuple[int, int], str]:
    """
    Parse a Python file and return a mapping of (start_line, end_line) -> function identifier.

    For top-level functions: "function_name"
    For class methods: "ClassName.method_name"

    Nested functions and inner classes are ignored - only the outermost
    function/method is recorded.

    Args:
        file_path: Path to the Python source file.

    Returns:
        Dictionary mapping (start_line, end_line) tuples to function identifier strings.
        Returns empty dict if file cannot be parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return {}

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return {}

    functions: dict[tuple[int, int], str] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            end_line = _get_end_lineno(node, source)
            functions[(node.lineno, end_line)] = node.name

        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                    end_line = _get_end_lineno(item, source)
                    functions[(item.lineno, end_line)] = f"{class_name}.{item.name}"


    return functions


def find_function_for_line(
    functions: dict[tuple[int, int], str], line_number: int
) -> Optional[str]:
    """
    Find which function contains the given line number.

    Args:
        functions: Output of parse_source_functions().
        line_number: 1-based line number to look up.

    Returns:
        Function identifier string, or None if the line is not inside any function.
    """
    for (start, end), name in functions.items():
        if start <= line_number <= end:
            return name
    return None


def _get_end_lineno(node: ast.AST, source: str) -> int:
    """
    Get the end line number of an AST node.

    Python 3.8+ provides end_lineno directly. For older versions,
    we fall back to scanning child nodes.
    """
    if hasattr(node, "end_lineno") and node.end_lineno is not None:
        return node.end_lineno

    # Fallback: find the maximum line number among all child nodes
    max_line = node.lineno
    for child in ast.walk(node):
        if hasattr(child, "lineno") and child.lineno is not None:
            max_line = max(max_line, child.lineno)
    return max_line


def is_utility_file(file_path: str) -> bool:
    """
    Determine if a file should be mapped at file-level rather than function-level.

    Utility files, configs, and __init__.py files are mapped at file level
    because changes to them can affect many different components.

    Args:
        file_path: Relative path from project root.

    Returns:
        True if the file should use file-level mapping.
    """
    normalized = file_path.replace("\\", "/")
    basename = os.path.basename(normalized)

    if basename in {"__init__.py", "conftest.py"}:
        return True

    if normalized.startswith("vllm_ascend/utils/"):
        return True
    if normalized.startswith("vllm_ascend/config/"):
        return True

    # Keep a conservative fallback for known utility single files
    if basename in {"utils.py", "envs.py"}:
        return True

    return False
