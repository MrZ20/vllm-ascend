import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIA_DIR = REPO_ROOT / "tools" / "tia"
SCRIPTS_DIR = REPO_ROOT / ".github" / "workflows" / "scripts"

# Import TIA modules as plain scripts (tools/ is not a Python package).
sys.path.insert(0, str(TIA_DIR))
import analyze_changes  # noqa: E402
import ast_parser  # noqa: E402
import generate_mapping  # noqa: E402
import verify_coverage  # noqa: E402

# Import CI helper scripts similarly.
sys.path.insert(0, str(SCRIPTS_DIR))
import ci_utils  # noqa: E402


def _lines_to_numbits(lines: list[int]) -> bytes:
    """Encode 1-based line numbers into a coverage.py-like numbits blob."""
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
    """Create a minimal .coverage SQLite DB with file/context/line_bits tables."""
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


def test_normalize_test_context_strips_phase_suffixes():
    assert generate_mapping.normalize_test_context("a.py::test_x|run") == "a.py::test_x"
    assert generate_mapping.normalize_test_context("a.py::test_x|setup") == "a.py::test_x"
    assert generate_mapping.normalize_test_context("a.py::test_x|call") == "a.py::test_x"


def test_generate_mapping_minimal_sqlite_db(tmp_path: Path):
    # Arrange: create a minimal source file with stable line numbers.
    src_root = tmp_path / "vllm_ascend"
    src_root.mkdir(parents=True)
    src_file = src_root / "sample.py"

    # Line numbers matter for mapping.
    src_file.write_text(
        """
def top():
    x = 1
    return x


class C:
    def m(self):
        return 2
""".lstrip(),
        encoding="utf-8",
    )

    # Build coverage DB: execute line 2 in top() and line 8 in C.m()
    db_path = tmp_path / ".coverage"
    contexts_to_lines = {
        "tests/ut/test_a.py::test_a|run": [2],
        "tests/ut/test_b.py::test_b|call": [8],
    }
    _make_coverage_db(
        db_path,
        file_path=str(src_file),
        contexts_to_lines=contexts_to_lines,
    )

    # Act
    mapping = generate_mapping.generate_mapping(
        coverage_db_path=str(db_path),
        source_dir="vllm_ascend",
        project_root=str(tmp_path),
    )

    # Assert
    func_mapping = mapping["func_mapping"]
    assert "vllm_ascend/sample.py::top" in func_mapping
    assert "vllm_ascend/sample.py::C.m" in func_mapping

    assert func_mapping["vllm_ascend/sample.py::top"] == ["tests/ut/test_a.py::test_a"]
    assert func_mapping["vllm_ascend/sample.py::C.m"] == ["tests/ut/test_b.py::test_b"]


def test_analyze_changes_recommends_tests_from_func_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a tiny git repo to test diff parsing without any NPU dependency.
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    subprocess.run(["git", "init", "-b", "main"], check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "CI"], check=True)

    (repo / "vllm_ascend").mkdir()
    src = repo / "vllm_ascend" / "mod.py"
    src.write_text(
        """
def foo():
    a = 1
    return a
""".lstrip(),
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "base"], check=True)

    # Change a line inside foo()
    src.write_text(
        """
def foo():
    a = 2
    return a
""".lstrip(),
        encoding="utf-8",
    )

    subprocess.run(["git", "add", str(src)], check=True)
    subprocess.run(["git", "commit", "-m", "change foo"], check=True)

    mapping = {
        "file_mapping": {},
        "func_mapping": {
            "vllm_ascend/mod.py::foo": ["tests/ut/test_mod.py::test_foo"],
        },
    }

    result = analyze_changes.analyze_changes(base_ref="HEAD~1", mapping=mapping, project_root=str(repo))
    assert result["all_tests"] is False
    assert "tests/ut/test_mod.py::test_foo" in result["recommended_tests"]


def test_verify_coverage_matches_classname_to_path(tmp_path: Path):
    # recommended contains file-based nodeid
    rec = {
        "all_tests": False,
        "recommended_tests": ["tests/ut/test_mod.py::test_foo"],
    }
    rec_path = tmp_path / "recommended.json"
    rec_path.write_text(__import__("json").dumps(rec), encoding="utf-8")

    # junitxml without file attr but with dotted classname
    junit = (
        "<testsuite failures=\"1\" errors=\"0\" tests=\"1\">"
        "  <testcase classname=\"tests.ut.test_mod\" name=\"test_foo\">"
        "    <failure message=\"boom\"/>"
        "  </testcase>"
        "</testsuite>"
    )
    junit_path = tmp_path / "results.xml"
    junit_path.write_text(junit, encoding="utf-8")

    report = verify_coverage.verify(
        recommended_path=str(rec_path),
        test_results_path="",
        junit_xml_path=str(junit_path),
    )
    assert report.missed_failures == []
    assert report.covered_failures != []


def test_run_e2e_files_passes_pytest_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Stub Popen to capture args without actually running pytest.
    captured = {}

    class _Proc:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(ci_utils.subprocess, "Popen", _Proc)

    tf = ci_utils.TestFile(name="tests/ut/test_dummy.py", estimated_time=1)
    # The file doesn't need to exist since we don't really execute pytest.
    ret = ci_utils.run_e2e_files([tf], continue_on_error=False, pytest_args="--junitxml=out.xml -k smoke")
    assert ret == 0
    assert "--junitxml=out.xml" in captured["args"]
    assert "-k" in captured["args"]
    assert "smoke" in captured["args"]
