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
Verify TIA recommendations by comparing with actual test failures.

Shadow mode: Run full tests but compare results with TIA recommendations
to validate coverage.

Usage:
    python tools/tia/verify_coverage.py \
        --recommended recommended_tests.json \
        --test-results test-results.json \
        --output tia_report.json

Input:
    - recommended_tests.json: Output of analyze_changes.py
    - test-results.json: Output of run_suite.py --failed-tests-log

Output:
    - tia_report.json: Coverage verification report
    - Logs to stdout with clear PASS/FAIL indication
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class TestResult:
    """Represents a single test result from JUnit XML."""
    name: str
    classname: str
    status: str  # "passed", "failed", "error", "skipped"
    message: str = ""
    file_path: str = ""

    @property
    def full_id(self) -> str:
        """Get full test identifier like tests/e2e/singlecard/test_models.py::test_func"""
        file_path = self.file_path or _classname_to_py_path(self.classname)
        if file_path:
            return f"{file_path}::{self.name}"
        if self.classname:
            return f"{self.classname}::{self.name}"
        return self.name


def _classname_to_py_path(classname: str) -> str:
    """
    Convert pytest junitxml classname (e.g. tests.e2e.singlecard.test_models)
    to a file path (tests/e2e/singlecard/test_models.py).
    """
    if not classname:
        return ""
    cls = classname.split("::", 1)[0]
    return cls.replace(".", "/") + ".py"


@dataclass
class TIAReport:
    """TIA verification report."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0

    recommended_count: int = 0
    all_tests_recommended: bool = False

    # Failed tests that WERE in the recommended list (correctly predicted)
    covered_failures: list[str] = field(default_factory=list)
    # Failed tests that were NOT in the recommended list (missed by TIA)
    missed_failures: list[str] = field(default_factory=list)

    coverage_rate: float = 0.0
    verdict: str = ""


def parse_junit_xml(xml_path: str) -> list[TestResult]:
    """
    Parse JUnit XML file to extract test results.
    """
    if not os.path.exists(xml_path):
        print(f"ERROR: JUnit XML not found: {xml_path}")
        return []

    results = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML: {e}")
        return []

    testsuites = root.findall(".//testcase")

    for tc in testsuites:
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        file_path = tc.get("file", "")

        if tc.find("failure") is not None:
            status = "failed"
            failure = tc.find("failure")
            message = failure.get("message", "") if failure is not None else ""
        elif tc.find("error") is not None:
            status = "error"
            error = tc.find("error")
            message = error.get("message", "") if error is not None else ""
        elif tc.find("skipped") is not None:
            status = "skipped"
            message = ""
        else:
            status = "passed"
            message = ""

        results.append(TestResult(
            name=name,
            classname=classname,
            status=status,
            message=message,
            file_path=file_path,
        ))

    return results


def parse_run_suite_results(json_path: str) -> tuple[list[str], list[str]]:
    """
    Parse the JSON output from run_suite.py --failed-tests-log.

    Returns (passed_files, failed_files) where each is a list of
    test file paths like "tests/e2e/singlecard/test_models.py".
    """
    if not os.path.exists(json_path):
        print(f"ERROR: Test results JSON not found: {json_path}")
        return [], []

    with open(json_path, "r") as f:
        data = json.load(f)

    return data.get("passed", []), data.get("failed", [])


def normalize_test_id(test_id: str) -> tuple[str, str, str]:
    """
    Normalize a test identifier for matching.

    Returns (path_part, test_part, base_test_part)
    """
    if "::" in test_id:
        path_part, test_part = test_id.split("::", 1)
    else:
        path_part = test_id
        test_part = ""

    path_part = path_part.replace("\\", "/")
    base_test_part = re.sub(r"\[.*\]$", "", test_part) if test_part else ""

    return path_part, test_part, base_test_part


def test_is_recommended(test_id: str, recommended: set[str]) -> bool:
    """
    Check if a test ID matches any entry in the recommended set.

    Matching is done at multiple levels:
    1. Exact match
    2. File-level match (test file path matches)
    3. Base function match (ignoring parametrize params)
    4. Substring containment for file paths
    """
    if not recommended or "ALL_TESTS" in recommended:
        return True

    # Normalize the test_id
    test_id_normalized = test_id.replace("\\", "/")

    # Direct match
    if test_id_normalized in recommended:
        return True

    path_part, test_part, base_test_part = normalize_test_id(test_id_normalized)

    for rec in recommended:
        rec_normalized = rec.replace("\\", "/")

        # Direct match with normalized rec
        if test_id_normalized == rec_normalized:
            return True

        rec_path, rec_test, rec_base = normalize_test_id(rec_normalized)

        # File-level match: recommended is a file, test belongs to it
        if not rec_test:
            # rec is just a file path
            if path_part == rec_path:
                return True
            if path_part.endswith(rec_path) or rec_path.endswith(path_part):
                return True

        # Full match including test part
        if path_part == rec_path and test_part and test_part == rec_test:
            return True

        # Base function match (ignore parametrize)
        if path_part == rec_path and base_test_part and base_test_part == rec_base:
            return True

        # Partial path + test match
        if rec_path in path_part or path_part in rec_path:
            if rec_test and (test_part == rec_test or base_test_part == rec_base):
                return True

    return False


def verify(
    recommended_path: str,
    test_results_path: str = "",
    junit_xml_path: str = "",
) -> TIAReport:
    """
    Verify TIA recommendations against actual test results.

    Args:
        recommended_path: Path to recommended_tests.json
        test_results_path: Path to run_suite.py JSON results (preferred)
        junit_xml_path: Path to JUnit XML results (fallback)

    Returns:
        TIAReport with verification results
    """
    report = TIAReport()

    # Load recommendations
    with open(recommended_path, "r") as f:
        rec_data = json.load(f)

    recommended_tests = set(rec_data.get("recommended_tests", []))
    report.all_tests_recommended = rec_data.get("all_tests", False)
    report.recommended_count = len(recommended_tests)

    failed_test_ids: list[str] = []

    # Prefer run_suite.py JSON format (file-level granularity)
    if test_results_path and os.path.exists(test_results_path):
        passed_files, failed_files = parse_run_suite_results(test_results_path)
        report.total_tests = len(passed_files) + len(failed_files)
        report.passed_tests = len(passed_files)
        report.failed_tests = len(failed_files)
        failed_test_ids = failed_files

    elif junit_xml_path and os.path.exists(junit_xml_path):
        test_results = parse_junit_xml(junit_xml_path)
        report.total_tests = len(test_results)
        report.passed_tests = sum(1 for t in test_results if t.status == "passed")
        report.failed_tests = sum(1 for t in test_results if t.status == "failed")
        report.error_tests = sum(1 for t in test_results if t.status == "error")
        report.skipped_tests = sum(1 for t in test_results if t.status == "skipped")

        failed_test_ids = [
            t.full_id for t in test_results
            if t.status in ("failed", "error")
        ]
    else:
        report.verdict = "SKIP - No test results found"
        return report

    if not failed_test_ids:
        report.verdict = "PASS - No failures to verify"
        report.coverage_rate = 1.0
        return report

    # Check coverage
    for failed_id in failed_test_ids:
        if report.all_tests_recommended or test_is_recommended(failed_id, recommended_tests):
            report.covered_failures.append(failed_id)
        else:
            report.missed_failures.append(failed_id)

    total_failures = len(failed_test_ids)
    covered = len(report.covered_failures)
    report.coverage_rate = covered / total_failures if total_failures > 0 else 1.0

    if report.missed_failures:
        report.verdict = f"WARN - TIA missed {len(report.missed_failures)}/{total_failures} failures"
    else:
        report.verdict = f"PASS - TIA covered all {total_failures} failures"

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify TIA recommendations against actual test results"
    )
    parser.add_argument(
        "--recommended",
        required=True,
        help="Path to recommended_tests.json from analyze_changes.py",
    )
    parser.add_argument(
        "--test-results",
        default="",
        help="Path to run_suite.py JSON results (--failed-tests-log output)",
    )
    parser.add_argument(
        "--junit-xml",
        default="",
        help="Path to JUnit XML test results (fallback)",
    )
    parser.add_argument(
        "--output",
        default="tia_report.json",
        help="Output report JSON file",
    )

    args = parser.parse_args()

    if not args.test_results and not args.junit_xml:
        print("ERROR: Must provide either --test-results or --junit-xml")
        sys.exit(1)

    report = verify(
        args.recommended,
        test_results_path=args.test_results,
        junit_xml_path=args.junit_xml,
    )

    # Print report
    print("\n" + "=" * 60)
    print("TIA VERIFICATION REPORT")
    print("=" * 60)
    print(f"Verdict: {report.verdict}")
    print(f"Coverage rate: {report.coverage_rate:.1%}")
    print()
    print(f"Total tests run: {report.total_tests}")
    print(f"  Passed: {report.passed_tests}")
    print(f"  Failed: {report.failed_tests}")
    if report.error_tests:
        print(f"  Errors: {report.error_tests}")
    if report.skipped_tests:
        print(f"  Skipped: {report.skipped_tests}")
    print()
    print(f"TIA recommended {report.recommended_count} tests"
          f"{' (ALL_TESTS)' if report.all_tests_recommended else ''}")
    print()

    if report.covered_failures:
        print(f"✅ Covered failures ({len(report.covered_failures)}):")
        for t in report.covered_failures:
            print(f"  ✓ {t}")

    if report.missed_failures:
        print(f"\n❌ MISSED failures ({len(report.missed_failures)}):")
        for t in report.missed_failures:
            print(f"  ✗ {t}")

    print("=" * 60)

    # Write report
    report_dict = {
        "verdict": report.verdict,
        "coverage_rate": report.coverage_rate,
        "all_tests_recommended": report.all_tests_recommended,
        "recommended_count": report.recommended_count,
        "total_tests": report.total_tests,
        "passed_tests": report.passed_tests,
        "failed_tests": report.failed_tests,
        "error_tests": report.error_tests,
        "skipped_tests": report.skipped_tests,
        "covered_failures": report.covered_failures,
        "missed_failures": report.missed_failures,
    }

    with open(args.output, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
