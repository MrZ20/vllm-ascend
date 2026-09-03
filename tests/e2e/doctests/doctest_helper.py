#!/usr/bin/env python3

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MKDOCS_PATH = Path("mkdocs.yml")
MARKER_RE = re.compile(r"^[ \t]*<!--\s*doctest:\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*-->[ \t]*$")
FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)```(?P<language>bash|python)[ \t]*$")
EXTRA_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>.*)$")
MACRO_RE = re.compile(r"{{\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*}}")


class DoctestError(ValueError):
    pass


def extract_block(text: str, marker: str, source: str = "input") -> str | None:
    lines = text.splitlines()
    marker_lines = [
        index for index, line in enumerate(lines) if (match := MARKER_RE.match(line)) and match.group(1) == marker
    ]
    if not marker_lines:
        return None
    if len(marker_lines) > 1:
        raise DoctestError(f"Duplicate doctest marker '{marker}' in {source}.")

    opening_index = marker_lines[0] + 1
    while opening_index < len(lines) and not lines[opening_index].strip():
        opening_index += 1
    if opening_index >= len(lines):
        raise DoctestError(f"Doctest marker '{marker}' in {source} is not followed by a bash or python code block.")

    opening = FENCE_RE.match(lines[opening_index])
    if opening is None:
        raise DoctestError(f"Doctest marker '{marker}' in {source} is not followed by a bash or python code block.")

    indent = opening.group("indent")
    body: list[str] = []
    for line in lines[opening_index + 1 :]:
        if line.strip() == "```":
            content = "\n".join(line[len(indent) :] if line.startswith(indent) else line for line in body)
            return content + ("\n" if body else "")
        body.append(line)
    raise DoctestError(f"Code block for doctest marker '{marker}' in {source} is not closed.")


def _strip_yaml_comment(value: str) -> str:
    quote = ""
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote and character == "\\" and quote == '"':
            escaped = True
            continue
        if character in ("'", '"'):
            if not quote:
                quote = character
            elif quote == character:
                quote = ""
            continue
        if character == "#" and not quote and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def _parse_scalar(value: str, key: str) -> str:
    value = _strip_yaml_comment(value).strip()
    if not value or value[0] in "![]{}|>":
        raise DoctestError(f"mkdocs.yml extra key '{key}' is not a simple scalar.")
    if value[0] in ("'", '"'):
        if len(value) < 2 or value[-1] != value[0]:
            raise DoctestError(f"mkdocs.yml extra key '{key}' has an invalid quoted scalar.")
        value = value[1:-1]
    return value


def load_extra(text: str, source: str = "mkdocs.yml") -> dict[str, str]:
    lines = text.splitlines()
    extra_line = next((index for index, line in enumerate(lines) if re.match(r"^extra\s*:\s*(?:#.*)?$", line)), None)
    if extra_line is None:
        raise DoctestError(f"No top-level extra mapping found in {source}.")

    values: dict[str, str] = {}
    seen: set[str] = set()
    extra_indent: int | None = None
    for line in lines[extra_line + 1 :]:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if extra_indent is None:
            if indent == 0:
                break
            extra_indent = indent
        if indent < extra_indent:
            break
        if indent != extra_indent:
            continue
        match = EXTRA_RE.match(stripped)
        if match is None:
            continue
        key = match.group("name")
        if key in seen:
            raise DoctestError(f"Duplicate mkdocs.yml extra key '{key}' in {source}.")
        seen.add(key)
        raw_value = _strip_yaml_comment(match.group("value")).strip()
        if not raw_value or raw_value[0] in "![]{}|>":
            continue
        values[key] = _parse_scalar(raw_value, key)
    return values


def release_values(extra: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in extra.items()
        if key.startswith("release_") or key in {"vllm_version", "vllm_ascend_version"}
    }


def release_changed(base_text: str, head_text: str) -> bool:
    return release_values(load_extra(base_text)) != release_values(load_extra(head_text))


def expand_macros(content: str, extra: dict[str, str], marker: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in extra:
            raise DoctestError(f"Unknown mkdocs.yml macro '{{{{ {name} }}}}' in doctest marker '{marker}'.")
        return extra[name]

    return MACRO_RE.sub(replace, content)


def _repo_path(path: Path) -> str:
    absolute = path if path.is_absolute() else REPO_ROOT / path
    try:
        return absolute.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise DoctestError(f"Path is outside the repository: {path}") from error


def _check_ref(ref: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise DoctestError(f"Unknown git ref '{ref}'.")


def read_text(path: Path, ref: str | None = None, *, allow_missing: bool = False) -> str | None:
    if ref is None:
        absolute = path if path.is_absolute() else REPO_ROOT / path
        if not absolute.is_file():
            if allow_missing:
                return None
            raise DoctestError(f"File not found: {path}")
        return absolute.read_text(encoding="utf-8")

    _check_ref(ref)
    repo_path = _repo_path(path)
    result = subprocess.run(
        ["git", "show", f"{ref}:{repo_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        if allow_missing:
            return None
        raise DoctestError(f"File '{repo_path}' does not exist at git ref '{ref}'.")
    return result.stdout


def block_changed(base_text: str | None, head_text: str | None, marker: str, source: str) -> bool:
    base_block = extract_block(base_text, marker, f"{source} at base") if base_text is not None else None
    head_block = extract_block(head_text, marker, f"{source} at head") if head_text is not None else None
    if base_block is None and head_block is None:
        raise DoctestError(f"Doctest marker '{marker}' does not exist at either ref in {source}.")
    return base_block != head_block


def _required_block(path: Path, marker: str, ref: str | None) -> str:
    text = read_text(path, ref)
    assert text is not None
    content = extract_block(text, marker, f"{path}{f' at {ref}' if ref else ''}")
    if content is None:
        raise DoctestError(f"Doctest marker '{marker}' was not found in {path}{f' at {ref}' if ref else ''}.")
    return content


def _extra_value(key: str, ref: str | None) -> str:
    text = read_text(MKDOCS_PATH, ref)
    assert text is not None
    values = load_extra(text, f"mkdocs.yml{f' at {ref}' if ref else ''}")
    if key not in values:
        raise DoctestError(f"No simple scalar named '{key}' under mkdocs.yml extra.")
    return values[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read marked documentation code blocks and mkdocs.yml extra values.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    block_parser = subparsers.add_parser("block")
    block_parser.add_argument("--ref")
    block_parser.add_argument("--expand", action="store_true")
    block_parser.add_argument("path", type=Path)
    block_parser.add_argument("marker")

    changed_parser = subparsers.add_parser("changed")
    changed_parser.add_argument("base")
    changed_parser.add_argument("head")
    changed_parser.add_argument("path", type=Path)
    changed_parser.add_argument("marker")

    extra_parser = subparsers.add_parser("extra")
    extra_parser.add_argument("--ref")
    extra_parser.add_argument("key")

    release_parser = subparsers.add_parser("release-changed")
    release_parser.add_argument("base")
    release_parser.add_argument("head")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "block":
        content = _required_block(args.path, args.marker, args.ref)
        if args.expand:
            mkdocs_text = read_text(MKDOCS_PATH, args.ref)
            assert mkdocs_text is not None
            content = expand_macros(content, load_extra(mkdocs_text), args.marker)
        sys.stdout.write(content)
        return 0
    if args.command == "changed":
        base_text = read_text(args.path, args.base, allow_missing=True)
        head_text = read_text(args.path, args.head, allow_missing=True)
        return 0 if block_changed(base_text, head_text, args.marker, str(args.path)) else 1
    if args.command == "release-changed":
        base_text = read_text(MKDOCS_PATH, args.base)
        head_text = read_text(MKDOCS_PATH, args.head)
        assert base_text is not None and head_text is not None
        return 0 if release_changed(base_text, head_text) else 1
    print(_extra_value(args.key, args.ref))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DoctestError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
