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

import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "collect_user_first_contribution.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def _run_script(
    tmp_path: Path,
    contributors_file: Path,
    args: list[str],
    *,
    git_stub: str | None = None,
    curl_stub: str | None = None,
    github_token: str | None = "fake-token",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()

    if git_stub is not None or curl_stub is not None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        if git_stub is not None:
            _write_executable(bin_dir / "git", git_stub)

        if curl_stub is not None:
            _write_executable(bin_dir / "curl", curl_stub)

        env["PATH"] = f"{bin_dir}:{env['PATH']}"

    if github_token is None:
        env.pop("GITHUB_TOKEN", None)
    else:
        env["GITHUB_TOKEN"] = github_token

    return subprocess.run(
        [str(SCRIPT), f"--file={contributors_file}", *args],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_sort_only_accepts_missing_and_empty_number_cells(tmp_path: Path) -> None:
    contributors_file = tmp_path / "contributors.md"
    contributors_file.write_text(
        textwrap.dedent("""
            # Test

            ## Contributors
            <!-- last_commit: abc -->

            | Number | Contributor | Date | Commit ID |
            |:------:|:-----------:|:-----:|:---------:|
            | 99 | [@numbered](https://github.com/numbered) | 2026/03/25 | [aaaaaaa](x) |
            | [@missing](https://github.com/missing) | 2026/03/24 | [bbbbbbb](x) |
            |  | [@empty](https://github.com/empty) | 2026/03/23 | [ccccccc](x) |

            tail
            """).lstrip(),
        encoding="utf-8",
    )

    result = _run_script(
        tmp_path,
        contributors_file,
        ["--sort-only"],
        github_token=None,
    )

    assert result.returncode == 0, result.stdout
    content = contributors_file.read_text(encoding="utf-8")
    assert "| 3 | [@numbered](https://github.com/numbered) | 2026/03/25 |" in content
    assert "| 2 | [@missing](https://github.com/missing) | 2026/03/24 |" in content
    assert "| 1 | [@empty](https://github.com/empty) | 2026/03/23 |" in content
    assert content.rstrip().endswith("tail")


def test_incremental_uses_full_history_to_filter_first_contribution(tmp_path: Path) -> None:
    contributors_file = tmp_path / "contributors.md"
    contributors_file.write_text(
        textwrap.dedent("""
            # Test

            ## Contributors

            <!-- last_commit: aaaaaaa -->

            Every release of vLLM Ascend would not have been possible without the following contributors:

            Updated on 2025-01-01:

            | Number | Contributor | Date | Commit ID |
            |:------:|:-----------:|:-----:|:---------:|
            | 1 | [@old](https://github.com/old) | 2025/01/01 | [1111111](x) |
            """).lstrip(),
        encoding="utf-8",
    )

    git_stub = """
        #!/usr/bin/env bash
        set -euo pipefail

        case "$1" in
          rev-parse)
            echo bbbbbbb
            ;;
          merge-base)
            exit 0
            ;;
          rev-list)
            printf "%s\\n" 2222222
            ;;
          log)
            if ! printf "%s\\n" "$*" | grep -q -- "--all"; then
              echo "incremental did not request full history" >&2
              exit 9
            fi
            printf "%s\\n" \\
              "100|1111111|old@example.com|2025-01-01T00:00:00+00:00|Old" \\
              "150|3333333|historical@example.com|2025-01-01T12:00:00+00:00|Historical" \\
              "200|2222222|new@example.com|2025-01-02T00:00:00+00:00|New"
            ;;
          *)
            exit 1
            ;;
        esac
    """
    curl_stub = """
        #!/usr/bin/env bash
        set -euo pipefail

        url="${*: -1}"
        case "$url" in
          */commits/1111111)
            printf "%s\\n" '{"author":{"login":"old"}}'
            ;;
          */commits/2222222)
            printf "%s\\n" '{"author":{"login":"new"}}'
            ;;
          */commits/3333333)
            printf "%s\\n" '{"author":{"login":"historical"}}'
            ;;
          *)
            printf "%s\\n" '{"author":{"login":null}}'
            ;;
        esac
    """

    result = _run_script(
        tmp_path,
        contributors_file,
        ["--repo=owner/repo"],
        git_stub=git_stub,
        curl_stub=curl_stub,
    )

    assert result.returncode == 0, result.stdout
    assert "Found 1 new contributors" in result.stdout

    content = contributors_file.read_text(encoding="utf-8")
    assert "## Contributors\n<!-- last_commit:" in content
    assert "@new" in content
    assert "@historical" not in content
    assert "| 2 | [@new](https://github.com/new) | 2025/01/02 |" in content
    assert "| 1 | [@old](https://github.com/old) | 2025/01/01 |" in content


def test_full_link_check_reports_invalid_links_and_replacement(tmp_path: Path) -> None:
    contributors_file = tmp_path / "contributors.md"
    contributors_file.write_text(
        textwrap.dedent("""
            # Test

            ## Contributors
            old section
            """).lstrip(),
        encoding="utf-8",
    )

    git_stub = """
        #!/usr/bin/env bash
        set -euo pipefail

        if [ "$1" = "rev-parse" ] && [ "$2" = "HEAD" ]; then
          echo fffffff
        elif [ "$1" = "log" ]; then
          printf "%s\\n" \\
            "100|aaaaaaa|bad@example.com|2025-01-01T00:00:00+00:00|Bad" \\
            "200|bbbbbbb|no@example.com|2025-01-02T00:00:00+00:00|No" \\
            "250|ccccccc|no@example.com|2025-01-03T00:00:00+00:00|No" \\
            "300|ddddddd|replace@example.com|2025-01-04T00:00:00+00:00|Replace" \\
            "400|eeeeeee|replace@example.com|2025-01-05T00:00:00+00:00|Replace" \\
            "500|9999999|good@example.com|2025-01-06T00:00:00+00:00|Good"
        else
          exit 1
        fi
    """
    curl_stub = """
        #!/usr/bin/env bash
        set -euo pipefail

        status=false
        url=""
        for arg in "$@"; do
          if [ "$arg" = "-w" ]; then
            status=true
          fi
          url="$arg"
        done

        if [ "$status" = true ]; then
          case "$url" in
            */users/invalid-profile)
              printf "404"
              ;;
            */users/no-valid)
              printf "200"
              ;;
            */users/valid-replacement)
              printf "200"
              ;;
            */users/all-good)
              printf "200"
              ;;
            */commits/bbbbbbb)
              printf "404"
              ;;
            */commits/ccccccc)
              printf "404"
              ;;
            */commits/ddddddd)
              printf "404"
              ;;
            */commits/eeeeeee)
              printf "200"
              ;;
            */commits/9999999)
              printf "200"
              ;;
            *)
              printf "200"
              ;;
          esac
        else
          case "$url" in
            */commits/aaaaaaa)
              printf "%s\\n" '{"author":{"login":"invalid-profile"}}'
              ;;
            */commits/bbbbbbb)
              printf "%s\\n" '{"author":{"login":"no-valid"}}'
              ;;
            */commits/ddddddd)
              printf "%s\\n" '{"author":{"login":"valid-replacement"}}'
              ;;
            */commits/9999999)
              printf "%s\\n" '{"author":{"login":"all-good"}}'
              ;;
            *)
              printf "%s\\n" '{"author":{"login":null}}'
              ;;
          esac
        fi
    """

    result = _run_script(
        tmp_path,
        contributors_file,
        ["--full", "--link-check", "--repo=owner/repo"],
        git_stub=git_stub,
        curl_stub=curl_stub,
    )

    assert result.returncode == 0, result.stdout
    assert "Checking 4 GitHub profile/commit link groups with 4 jobs" in result.stdout
    assert "all-good -- 200 ok" in result.stdout
    assert "| @invalid-profile | 2025/01/01 | aaaaaaa | ---- Invalid profile(404)" in result.stdout
    assert "| @no-valid | 2025/01/02 | bbbbbbb | ---- Invalid commit(404)" in result.stdout
    assert "| @valid-replacement | 2025/01/04 | ddddddd | ---- Invalid commit(404)" in result.stdout
    assert "=====>\n| @valid-replacement | 2025/01/05 | eeeeeee |" in result.stdout
    assert "Link check completed: 4/4 checked, 3 issue(s) found." in result.stdout

    content = contributors_file.read_text(encoding="utf-8")
    assert "| 3 | [@valid-replacement](https://github.com/valid-replacement) | 2025/01/04 |" in content
    assert "| 3 | [@valid-replacement](https://github.com/valid-replacement) | 2025/01/05 |" not in content
