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
import fnmatch
import json
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

HEADER_PREFIXES = {
    "# source doc:": "source_doc",
    "# group:": "group",
}


class ScriptRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class ScriptMetadata:
    path: str
    group: str
    source_doc: str


@dataclass(frozen=True)
class ScriptResult:
    metadata: ScriptMetadata
    log_path: str
    returncode: int

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def parse_script_metadata(script_path: Path) -> ScriptMetadata:
    source_doc = ""
    group = ""
    with script_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no > 40:
                break
            stripped = line.rstrip("\n")
            if stripped.startswith("#"):
                for prefix, key in HEADER_PREFIXES.items():
                    if stripped.startswith(prefix):
                        value = stripped[len(prefix):].strip()
                        if key == "source_doc":
                            source_doc = value
                        elif key == "group":
                            group = value

    if not group:
        group = script_path.stem
    return ScriptMetadata(path=str(script_path), group=group, source_doc=source_doc)


def expand_script_inputs(path_inputs: list[str]) -> list[Path]:
    scripts: list[Path] = []
    for path_input in path_inputs:
        candidate = Path(path_input)
        if candidate.is_file():
            if candidate.suffix != ".sh":
                raise ScriptRunnerError(
                    f"script input must be a .sh file: {candidate}")
            scripts.append(candidate.resolve())
            continue

        if candidate.is_dir():
            matches = sorted(
                path.resolve() for path in candidate.glob("*.sh")
                if path.is_file())
            if not matches:
                raise ScriptRunnerError(
                    f"script directory contains no .sh files: {candidate}")
            scripts.extend(matches)
            continue

        if not candidate.is_absolute():
            matches = sorted(
                path.resolve() for path in Path().glob(path_input)
                if path.is_file() and path.suffix == ".sh")
            if matches:
                scripts.extend(matches)
                continue

        raise ScriptRunnerError(f"no shell scripts matched '{path_input}'")

    unique = sorted({path for path in scripts})
    if not unique:
        raise ScriptRunnerError("no shell scripts selected")
    return unique


def match_patterns(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    script_name = Path(path).name
    return any(
        fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(script_name, pattern)
        for pattern in patterns)


def filter_scripts(script_paths: list[Path],
                   groups: set[str] | None = None,
                   docs: set[str] | None = None,
                   patterns: list[str] | None = None) -> list[ScriptMetadata]:
    groups = groups or set()
    docs = docs or set()
    patterns = patterns or []

    selected: list[ScriptMetadata] = []
    for script_path in script_paths:
        metadata = parse_script_metadata(script_path)
        if groups and metadata.group not in groups:
            continue

        if docs:
            doc_stem = Path(metadata.source_doc).stem if metadata.source_doc else ""
            if metadata.source_doc not in docs and doc_stem not in docs:
                continue

        if not match_patterns(metadata.path, patterns):
            continue
        selected.append(metadata)

    if not selected:
        raise ScriptRunnerError("no shell scripts matched the provided filters")
    return selected


def default_log_dir(script_metadata: ScriptMetadata) -> Path:
    script_path = Path(script_metadata.path)
    return script_path.parent / "logs"


def build_log_path(script_metadata: ScriptMetadata, log_dir: Path | None) -> Path:
    base_dir = log_dir or default_log_dir(script_metadata)
    return base_dir / f"{Path(script_metadata.path).stem}.log"


def print_script_banner(metadata: ScriptMetadata) -> None:
    print("=" * 80)
    print(f"Script: {metadata.path}")
    print(f"Group: {metadata.group}")
    print(f"Source doc: {metadata.source_doc or 'unknown'}")
    print("=" * 80)


def stream_process_output(proc: subprocess.Popen[str],
                          handle: TextIO) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        handle.write(line)


def run_script(script_metadata: ScriptMetadata, log_dir: Path | None) -> ScriptResult:
    log_path = build_log_path(script_metadata, log_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("w", encoding="utf-8") as handle:
            print_script_banner(script_metadata)
            handle.write(f"Script: {script_metadata.path}\n")
            handle.write(f"Group: {script_metadata.group}\n")
            handle.write(
                f"Source doc: {script_metadata.source_doc or 'unknown'}\n")
            handle.write("=" * 80 + "\n")

            proc = subprocess.Popen(
                ["/usr/bin/env", "bash", script_metadata.path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stream_process_output(proc, handle)
            returncode = proc.wait()
    except OSError as exc:
        raise ScriptRunnerError(
            f"failed to write log for {script_metadata.path}: {exc}") from exc

    return ScriptResult(metadata=script_metadata,
                        log_path=str(log_path),
                        returncode=returncode)


def print_summary(results: list[ScriptResult]) -> None:
    total = len(results)
    failed = [result for result in results if not result.succeeded]
    succeeded = total - len(failed)

    print("\nSummary")
    print(f"Total scripts: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {len(failed)}")

    for result in failed:
        print(
            f"FAILED: {result.metadata.path} (group={result.metadata.group}) log={result.log_path}"
        )


def build_summary_payload(results: list[ScriptResult]) -> dict[str, object]:
    failed = [result for result in results if not result.succeeded]
    return {
        "total": len(results),
        "succeeded": len(results) - len(failed),
        "failed": len(failed),
        "results": [
            {
                "metadata": asdict(result.metadata),
                "log_path": result.log_path,
                "returncode": result.returncode,
                "succeeded": result.succeeded,
            } for result in results
        ],
    }


def write_summary_json(results: list[ScriptResult], summary_json: Path | None) -> None:
    if summary_json is None:
        return
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_json.write_text(json.dumps(build_summary_payload(results), indent=2),
                                encoding="utf-8")
    except OSError as exc:
        raise ScriptRunnerError(
            f"failed to write summary json {summary_json}: {exc}") from exc


def list_scripts(metadata_list: list[ScriptMetadata], log_dir: Path | None) -> None:
    for metadata in metadata_list:
        print(f"script={metadata.path}")
        print(f"group={metadata.group}")
        print(f"source_doc={metadata.source_doc or 'unknown'}")
        print(f"log={build_log_path(metadata, log_dir)}")
        print("")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run shell scripts generated by the docs code-block extractor."
    )
    parser.add_argument("paths",
                        nargs="+",
                        help="Shell script paths, directories, or glob patterns.")
    parser.add_argument("--group",
                        action="append",
                        default=[],
                        help="Only run scripts whose group matches this value.")
    parser.add_argument("--doc",
                        action="append",
                        default=[],
                        help="Only run scripts whose source doc stem or full path matches this value.")
    parser.add_argument("--pattern",
                        action="append",
                        default=[],
                        help="Only run scripts whose path or filename matches this glob pattern.")
    parser.add_argument("--log-dir",
                        default=None,
                        help="Directory for runner log files. Defaults to <script-dir>/logs.")
    parser.add_argument("--summary-json",
                        default=None,
                        help="Optional path to a JSON summary file.")
    parser.add_argument("--keep-going",
                        action="store_true",
                        help="Continue running remaining scripts after a failure.")
    parser.add_argument("--list",
                        action="store_true",
                        help="List matching scripts without executing them.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Alias for --list.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    script_paths = expand_script_inputs(args.paths)
    selected = filter_scripts(script_paths,
                              groups=set(args.group),
                              docs=set(args.doc),
                              patterns=args.pattern)
    log_dir = Path(args.log_dir) if args.log_dir else None
    summary_json = Path(args.summary_json) if args.summary_json else None

    if args.list or args.dry_run:
        list_scripts(selected, log_dir)
        return 0

    results: list[ScriptResult] = []
    for metadata in selected:
        result = run_script(metadata, log_dir)
        results.append(result)
        if not result.succeeded and not args.keep_going:
            write_summary_json(results, summary_json)
            print_summary(results)
            return 1

    write_summary_json(results, summary_json)
    print_summary(results)
    return 0 if all(result.succeeded for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptRunnerError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
