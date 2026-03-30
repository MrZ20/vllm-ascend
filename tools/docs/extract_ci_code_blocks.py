#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SHELL_LANGUAGES = {"bash", "sh", "shell"}
DEFAULT_EXEC_CLASS = "doc-exec"
CODE_BLOCK_START_RE = re.compile(r"^\s*```{code-block}\s+([^\s}]+)\s*$")
CODE_BLOCK_END_RE = re.compile(r"^\s*```\s*$")
OPTION_RE = re.compile(r"^\s*:([A-Za-z0-9_-]+):\s*(.*?)\s*$")


class DocCodeExtractionError(ValueError):

    def __init__(self, doc_path: Path, line_no: int, message: str,
                 metadata: dict[str, str] | None = None) -> None:
        details = f"{doc_path}:{line_no}: {message}"
        if metadata:
            details = f"{details} metadata={metadata}"
        super().__init__(details)
        self.doc_path = doc_path
        self.line_no = line_no
        self.metadata = metadata or {}


@dataclass(frozen=True)
class CodeBlock:
    doc_path: str
    language: str
    name: str
    group: str
    classes: tuple[str, ...]
    content: str
    start_line: int


@dataclass(frozen=True)
class ScriptPlan:
    doc_path: str
    group: str
    output_path: str
    blocks: tuple[CodeBlock, ...]


def parse_markdown_code_blocks(doc_path: Path) -> list[tuple[str, int, dict[str,
                                                                        str], str, int]]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, int, dict[str, str], str, int]] = []
    index = 0
    while index < len(lines):
        match = CODE_BLOCK_START_RE.match(lines[index])
        if not match:
            index += 1
            continue

        language = match.group(1)
        start_line = index + 1
        index += 1

        metadata: dict[str, str] = {}
        while index < len(lines):
            option_match = OPTION_RE.match(lines[index])
            if option_match is None:
                break
            key = option_match.group(1)
            value = option_match.group(2)
            if key in metadata:
                raise DocCodeExtractionError(doc_path, index + 1,
                                             f"duplicate code block option '{key}'",
                                             metadata)
            metadata[key] = value
            index += 1

        body_lines: list[str] = []
        while index < len(lines) and not CODE_BLOCK_END_RE.match(lines[index]):
            body_lines.append(lines[index])
            index += 1

        if index >= len(lines):
            raise DocCodeExtractionError(doc_path, start_line,
                                         "unterminated code block", metadata)

        blocks.append((language, start_line, metadata, "\n".join(body_lines),
                       index + 1))
        index += 1

    return blocks


def collect_executable_blocks(
    doc_paths: list[Path],
    required_classes: set[str] | None = None,
    name_filters: set[str] | None = None,
    group_filters: set[str] | None = None,
) -> list[CodeBlock]:
    required_classes = required_classes or {DEFAULT_EXEC_CLASS}
    name_filters = name_filters or set()
    group_filters = group_filters or set()

    executable_blocks: list[CodeBlock] = []
    for doc_path in doc_paths:
        seen_names: dict[str, int] = {}
        for language, start_line, metadata, content, _ in parse_markdown_code_blocks(
                doc_path):
            classes = tuple(token for token in metadata.get("class", "").split()
                            if token)
            is_executable = required_classes.issubset(set(classes))
            if not is_executable:
                continue

            if language not in SHELL_LANGUAGES:
                raise DocCodeExtractionError(
                    doc_path, start_line,
                    "executable code block must use a shell language", metadata)

            name = metadata.get("name", "").strip()
            if not name:
                raise DocCodeExtractionError(doc_path, start_line,
                                             "executable code block is missing ':name:'",
                                             metadata)

            group = metadata.get("group", "").strip()
            if not group:
                raise DocCodeExtractionError(doc_path, start_line,
                                             "executable code block is missing ':group:'",
                                             metadata)

            if not content.strip():
                raise DocCodeExtractionError(doc_path, start_line,
                                             "executable code block is empty",
                                             metadata)

            if name in seen_names:
                raise DocCodeExtractionError(
                    doc_path, start_line,
                    f"duplicate executable code block name '{name}' "
                    f"(first defined at line {seen_names[name]})", metadata)
            seen_names[name] = start_line

            if name_filters and name not in name_filters:
                continue
            if group_filters and group not in group_filters:
                continue

            executable_blocks.append(
                CodeBlock(
                    doc_path=str(doc_path),
                    language=language,
                    name=name,
                    group=group,
                    classes=classes,
                    content=content,
                    start_line=start_line,
                ))

    return executable_blocks


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-_.").lower()


def build_output_filename(doc_path: Path, group: str) -> str:
    group_slug = slugify(group)
    return f"{group_slug}.sh"


def build_default_output_dir(doc_path: Path) -> Path:
    return Path("doctest") / doc_path.stem


def build_script_plans(blocks: list[CodeBlock],
                       output_dir: Path | None = None) -> list[ScriptPlan]:
    grouped: dict[tuple[str, str], list[CodeBlock]] = {}
    for block in blocks:
        key = (block.doc_path, block.group)
        grouped.setdefault(key, []).append(block)

    plans: list[ScriptPlan] = []
    for (doc_path, group), group_blocks in grouped.items():
        base_output_dir = output_dir or build_default_output_dir(Path(doc_path))
        output_path = base_output_dir / build_output_filename(
            Path(doc_path), group)
        plans.append(
            ScriptPlan(doc_path=doc_path,
                       group=group,
                       output_path=str(output_path),
                       blocks=tuple(group_blocks)))

    plans.sort(key=lambda plan: (plan.doc_path, plan.group))
    return plans


def render_shell_script(plan: ScriptPlan) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "set -u",
        "set -o pipefail",
        "",
        f"# source doc: {plan.doc_path}",
        f"# group: {plan.group}",
    ]

    for block in plan.blocks:
        lines.extend([
            "",
            f"printf '%s\\n' '==> source doc: {block.doc_path}'",
            f"printf '%s\\n' '==> group: {block.group}'",
            f"printf '%s\\n' '==> code block name: {block.name}'",
            f"# begin code block: {block.name} ({block.doc_path}:{block.start_line})",
            block.content,
            f"# end code block: {block.name}",
        ])

    return "\n".join(lines) + "\n"


def write_shell_scripts(plans: list[ScriptPlan], dry_run: bool) -> None:
    for plan in plans:
        if dry_run:
            continue
        output_path = Path(plan.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_shell_script(plan), encoding="utf-8")
        output_path.chmod(0o755)


def expand_doc_inputs(path_inputs: list[str]) -> list[Path]:
    doc_paths: list[Path] = []
    for path_input in path_inputs:
        candidate = Path(path_input)
        if candidate.is_file():
            doc_paths.append(candidate.resolve())
            continue

        if not candidate.is_absolute():
            matches = sorted(Path().glob(path_input))
            if matches:
                doc_paths.extend(path.resolve() for path in matches
                                 if path.is_file())
                continue

        raise FileNotFoundError(f"no markdown documents matched '{path_input}'")

    unique_paths = sorted({path.resolve() for path in doc_paths})
    return unique_paths


def pretty_print(plans: list[ScriptPlan]) -> None:
    payload = []
    for plan in plans:
        plan_dict = asdict(plan)
        payload.append(plan_dict)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CI-marked MyST code blocks into shell scripts.")
    parser.add_argument("paths",
                        nargs="*",
                        help="Markdown file paths or glob patterns.")
    parser.add_argument("--doc",
                        action="append",
                        default=[],
                        help="Additional markdown document path or glob.")
    parser.add_argument("--name",
                        action="append",
                        default=[],
                        help="Only include code blocks with this name.")
    parser.add_argument("--class",
                        dest="classes",
                        action="append",
                        default=[],
                        help="Require executable blocks to include this class. "
                        f"Defaults to '{DEFAULT_EXEC_CLASS}'.")
    parser.add_argument("--group",
                        action="append",
                        default=[],
                        help="Only include code blocks from this group.")
    parser.add_argument("--output-dir",
                        default=None,
                        help="Directory where shell scripts will be written.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Validate and print extraction results without "
                        "writing scripts.")
    parser.add_argument("--pretty",
                        action="store_true",
                        help="Print parsed plans as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    doc_inputs = [*args.paths, *args.doc]
    if not doc_inputs:
        parser.error("at least one markdown path must be provided")

    required_classes = set(args.classes or [DEFAULT_EXEC_CLASS])
    doc_paths = expand_doc_inputs(doc_inputs)
    blocks = collect_executable_blocks(
        doc_paths=doc_paths,
        required_classes=required_classes,
        name_filters=set(args.name),
        group_filters=set(args.group),
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    plans = build_script_plans(blocks, output_dir)

    if args.pretty or args.dry_run:
        pretty_print(plans)

    write_shell_scripts(plans, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DocCodeExtractionError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
