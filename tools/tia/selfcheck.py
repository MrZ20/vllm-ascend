#!/usr/bin/env python3
"""Stdlib-only self-check for TIA core logic.

This script is designed to run without NPU and without pytest being installed.
It validates:
- coverage SQLite parsing -> mapping generation (function-level)
- git diff changed-lines parsing -> recommendation lookup
- junitxml failure id normalization (classname -> file path)

Usage:
  python3 tools/tia/selfcheck.py

Exit code:
  0 on success, 1 on failure.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

# Make sibling modules importable
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import analyze_changes  # noqa: E402
import generate_mapping  # noqa: E402
import verify_coverage  # noqa: E402


def _lines_to_numbits(lines: list[int]) -> bytes:
    if not lines:
        return b""
    max_line = max(lines)
    size = (max_line + 7) // 8
    buf = bytearray(size)
    for ln in lines:
        if ln <= 0:
            continue
        idx = (ln - 1) // 8
        bit = (ln - 1) % 8
        buf[idx] |= 1 << bit
    return bytes(buf)


def _make_coverage_db(db_path: Path, file_path: str, contexts_to_lines: dict[str, list[int]]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    cur.execute("CREATE TABLE context (id INTEGER PRIMARY KEY, context TEXT)")
    cur.execute("CREATE TABLE line_bits (file_id INTEGER, context_id INTEGER, numbits BLOB)")

    cur.execute("INSERT INTO file (id, path) VALUES (?, ?)", (1, file_path))

    ctx_id = 1
    for ctx, lines in contexts_to_lines.items():
        cur.execute("INSERT INTO context (id, context) VALUES (?, ?)", (ctx_id, ctx))
        numbits = sqlite3.Binary(_lines_to_numbits(lines))
        cur.execute(
            "INSERT INTO line_bits (file_id, context_id, numbits) VALUES (?, ?, ?)",
            (1, ctx_id, numbits),
        )
        ctx_id += 1

    conn.commit()
    conn.close()


def _run(cmd: list[str], cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_mapping_generation(tmp: Path) -> None:
    src_root = tmp / "vllm_ascend"
    src_root.mkdir(parents=True)
    src_file = src_root / "sample.py"
    src_file.write_text(
        """\
def top():
    x = 1
    return x


class C:
    def m(self):
        return 2
""",
        encoding="utf-8",
    )

    db_path = tmp / ".coverage"
    _make_coverage_db(
        db_path,
        file_path=str(src_file),
        contexts_to_lines={
            "tests/ut/test_a.py::test_a|run": [2],
            "tests/ut/test_b.py::test_b|call": [8],
        },
    )

    mapping = generate_mapping.generate_mapping(
        coverage_db_path=str(db_path),
        source_dir="vllm_ascend",
        project_root=str(tmp),
    )

    func_mapping = mapping["func_mapping"]
    assert func_mapping["vllm_ascend/sample.py::top"] == ["tests/ut/test_a.py::test_a"]
    assert func_mapping["vllm_ascend/sample.py::C.m"] == ["tests/ut/test_b.py::test_b"]


def check_analyze_changes(tmp: Path) -> None:
    repo = tmp / "repo"
    repo.mkdir(parents=True)

    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "ci@example.com"], repo)
    _run(["git", "config", "user.name", "CI"], repo)

    (repo / "vllm_ascend").mkdir()
    src = repo / "vllm_ascend" / "mod.py"
    src.write_text(
        """\
def foo():
    a = 1
    return a
""",
        encoding="utf-8",
    )
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "base"], repo)

    src.write_text(
        """\
def foo():
    a = 2
    return a
""",
        encoding="utf-8",
    )
    _run(["git", "add", "vllm_ascend/mod.py"], repo)
    _run(["git", "commit", "-m", "change"], repo)

    mapping = {
        "file_mapping": {},
        "func_mapping": {"vllm_ascend/mod.py::foo": ["tests/ut/test_mod.py::test_foo"]},
    }

    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        result = analyze_changes.analyze_changes(base_ref="HEAD~1", mapping=mapping, project_root=str(repo))
    finally:
        os.chdir(old_cwd)

    assert result["all_tests"] is False
    assert "tests/ut/test_mod.py::test_foo" in result["recommended_tests"]


def check_verify_coverage(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    rec_path = tmp / "recommended.json"
    rec_path.write_text(
        json.dumps({"all_tests": False, "recommended_tests": ["tests/ut/test_mod.py::test_foo"]}),
        encoding="utf-8",
    )

    junit_path = tmp / "results.xml"
    junit_path.write_text(
        """\
<testsuite failures="1" errors="0" tests="1">
    <testcase classname="tests.ut.test_mod" name="test_foo">
        <failure message="boom"/>
    </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    report = verify_coverage.verify(
        recommended_path=str(rec_path),
        test_results_path="",
        junit_xml_path=str(junit_path),
    )

    assert report.missed_failures == []
    assert report.covered_failures


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        check_mapping_generation(tmp / "mapping")
        check_analyze_changes(tmp / "analyze")
        check_verify_coverage(tmp / "verify")

    print("TIA selfcheck OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"TIA selfcheck FAILED: {e}")
        raise SystemExit(1)
