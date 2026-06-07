#!/usr/bin/env python3
"""Run a list of selected pytest targets with colored logs, per-test timing and a summary.

This is the Python replacement for the legacy ``run_selected_tests.sh``. It keeps the same
positional CLI and the same on-disk log layout (so the workflow's artifact upload keeps working),
while adding per-target wall-clock timing, a final colored summary of every target, and a
``--continue-on-error`` switch to keep running after a failure instead of stopping at the first one.

Usage:
    run_selected_tests.py <npu_type> <num_npus> <with-device|without-device> <test> [test ...] \
        [--continue-on-error]
"""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# This target must run last, after triton is uninstalled (see the with-device branch).
ACLGRAPH_CAPTURE_REPLAY = "tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py"


class _Color:
    HEADER = "\033[95m"
    BLUE = "\033[1;34m"
    YELLOW = "\033[33m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


@dataclass
class TestRecord:
    target: str
    passed: bool
    elapsed: float
    log_file: Path


def _escape_github_actions_value(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _print_github_actions_group_start(title: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::group::{_escape_github_actions_value(title)}", flush=True)


def _print_github_actions_group_end() -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::endgroup::", flush=True)


def _print_github_actions_annotation(annotation: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{annotation}::{_escape_github_actions_value(message)}", flush=True)


def log_file_for(target: str, index: int, log_dir: Path) -> Path:
    """Build the per-target log path, mirroring the legacy bash sanitization."""
    name = target
    if name.startswith("tests/"):
        name = name[len("tests/"):]
    if name.endswith(".py"):
        name = name[: -len(".py")]
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return log_dir / f"{index}-{name}.log"


def print_test_info(npu_type: str, num_npus: str, targets: list[str]) -> None:
    print(f"{_Color.BLUE}=== TEST INFO ==={_Color.RESET}")
    print(f"  {_Color.YELLOW}Device:{_Color.RESET} {npu_type}")
    if npu_type != "cpu":
        print(f"  {_Color.YELLOW}NPU count:{_Color.RESET} {num_npus}")
    print(f"  {_Color.YELLOW}Targets:{_Color.RESET}")
    for target in targets:
        print(f"    {_Color.GREEN}-{_Color.RESET} {target}")
    print(f"{_Color.BLUE}{'=' * 20}{_Color.RESET}", flush=True)


def run_pytest(pytest_targets: list[str], log_file: Path, label: str) -> tuple[int, float]:
    """Run pytest on ``pytest_targets``, tee-ing output to stdout and ``log_file``.

    Returns ``(returncode, elapsed_seconds)``. ``--color=yes`` keeps ANSI colors in both the live
    console output and the saved log file, just like the legacy ``pytest ... | tee`` pipeline.
    """
    _print_github_actions_group_start(label)
    print(f"{_Color.BLUE}=== Running: {label} ==={_Color.RESET}", flush=True)
    start = time.perf_counter()
    try:
        with open(log_file, "wb") as f:
            proc = subprocess.Popen(
                ["pytest", "-sv", "--color=yes", *pytest_targets],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                f.write(line)
            returncode = proc.wait()
    finally:
        _print_github_actions_group_end()
    elapsed = time.perf_counter() - start
    return returncode, elapsed


def print_summary(records: list[TestRecord]) -> None:
    passed_count = sum(1 for r in records if r.passed)
    total_elapsed = sum(r.elapsed for r in records)
    all_passed = passed_count == len(records)

    print(f"\n{_Color.BLUE}=== TEST SUMMARY ==={_Color.RESET}")
    for r in records:
        status = f"{_Color.GREEN}✅ PASSED{_Color.RESET}" if r.passed else f"{_Color.RED}❌ FAILED{_Color.RESET}"
        print(f"  {status}  {r.target}  ({r.elapsed:.1f}s)")
        print(f"    {_Color.YELLOW}log:{_Color.RESET} {r.log_file}")
    summary_color = _Color.GREEN if all_passed else _Color.RED
    print(
        f"{summary_color}Summary: {passed_count}/{len(records)} passed  (total {total_elapsed:.1f}s){_Color.RESET}",
        flush=True,
    )

    failed = [r for r in records if not r.passed]
    if failed:
        print(f"{_Color.RED}=== FAILED TEST LOGS ==={_Color.RESET}")
        for r in failed:
            _print_github_actions_group_start(f"{r.target} failure log")
            try:
                sys.stdout.write(r.log_file.read_text(encoding="utf-8", errors="replace"))
            except OSError as e:
                print(f"(could not read log {r.log_file}: {e})")
            _print_github_actions_group_end()
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected pytest targets with colored logs, per-test timing and a summary.",
    )
    parser.add_argument("npu_type", help="Device type, e.g. 'cpu', '310p', 'a2'")
    parser.add_argument("num_npus", help="Number of NPUs (ignored for cpu)")
    parser.add_argument("mode", choices=["with-device", "without-device"])
    parser.add_argument("targets", nargs="+", help="One or more pytest targets")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run every target even if some fail (default: stop at the first failure).",
    )
    return parser.parse_args()


def main() -> None:
    # Force UTF-8 so the colored ✅/❌ summary never crashes on non-UTF-8 locales (e.g. Windows GBK).
    # The tee writes raw pytest bytes to sys.stdout.buffer, so this only affects our own text output.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    log_dir = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"selected-tests-{args.npu_type}-{args.num_npus}card"
    log_dir.mkdir(parents=True, exist_ok=True)

    print_test_info(args.npu_type, args.num_npus, args.targets)

    records: list[TestRecord] = []
    index = 0

    def run_one(pytest_targets: list[str], label: str, log_file: Path) -> None:
        returncode, elapsed = run_pytest(pytest_targets, log_file, label)
        passed = returncode == 0
        records.append(TestRecord(target=label, passed=passed, elapsed=elapsed, log_file=log_file))
        if not passed:
            _print_github_actions_annotation(
                "error",
                f"FAILED {label}. See the TEST SUMMARY / FAILED TEST LOGS section for details.",
            )

    def should_stop() -> bool:
        return not args.continue_on_error and any(not r.passed for r in records)

    if args.npu_type == "cpu":
        # cpu runs all targets in a single batched pytest invocation (one process, one log).
        index += 1
        run_one(args.targets, f"cpu-ut ({len(args.targets)} targets)", log_dir / f"{index}-cpu-ut.log")
    elif args.mode == "with-device":
        run_aclgraph = ACLGRAPH_CAPTURE_REPLAY in args.targets
        for target in args.targets:
            if target == ACLGRAPH_CAPTURE_REPLAY:
                continue
            index += 1
            run_one([target], target, log_file_for(target, index, log_dir))
            if should_stop():
                break
        if run_aclgraph and not should_stop():
            # This test is incompatible with triton; remove it before running the test.
            subprocess.run(["pip", "uninstall", "-y", "triton-ascend", "triton"])
            index += 1
            run_one([ACLGRAPH_CAPTURE_REPLAY], ACLGRAPH_CAPTURE_REPLAY, log_file_for(ACLGRAPH_CAPTURE_REPLAY, index, log_dir))
    else:
        for target in args.targets:
            index += 1
            run_one([target], target, log_file_for(target, index, log_dir))
            if should_stop():
                break

    print_summary(records)
    sys.exit(0 if all(r.passed for r in records) else 1)


if __name__ == "__main__":
    main()
