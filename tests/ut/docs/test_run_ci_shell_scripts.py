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

import importlib.util
from pathlib import Path
import sys


def load_module():
    module_path = (Path(__file__).resolve().parents[3] / "tools" / "docs" /
                   "run_ci_shell_scripts.py")
    spec = importlib.util.spec_from_file_location("run_ci_shell_scripts",
                                                  module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_script(script_path: Path,
                 *,
                 group: str,
                 source_doc: str,
                 body: str) -> Path:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -e",
            "set -u",
            "set -o pipefail",
            "",
            f"# source doc: {source_doc}",
            f"# group: {group}",
            "",
            body,
            "",
        ]),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return script_path


def test_runs_scripts_successfully_and_generates_logs(tmp_path, capsys):
    module = load_module()
    script_dir = tmp_path / "doctest" / "DemoDoc"
    write_script(script_dir / "alpha.sh",
                 group="alpha",
                 source_doc="docs/source/tutorials/models/DemoDoc.md",
                 body="echo alpha-pass")

    exit_code = module.main([str(script_dir)])

    assert exit_code == 0
    log_path = script_dir / "logs" / "alpha.log"
    assert log_path.exists()
    assert "alpha-pass" in log_path.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "Total scripts: 1" in output
    assert "Succeeded: 1" in output


def test_fail_fast_stops_after_first_failure(tmp_path, capsys):
    module = load_module()
    script_dir = tmp_path / "doctest" / "DemoDoc"
    write_script(script_dir / "alpha.sh",
                 group="alpha",
                 source_doc="docs/source/tutorials/models/DemoDoc.md",
                 body="echo failing\nexit 3")
    write_script(script_dir / "beta.sh",
                 group="beta",
                 source_doc="docs/source/tutorials/models/DemoDoc.md",
                 body="echo should-not-run")

    exit_code = module.main([str(script_dir)])

    assert exit_code == 1
    assert (script_dir / "logs" / "alpha.log").exists()
    assert not (script_dir / "logs" / "beta.log").exists()
    output = capsys.readouterr().out
    assert "Failed: 1" in output
    assert "should-not-run" not in output


def test_keep_going_runs_remaining_scripts_and_returns_failure(tmp_path, capsys):
    module = load_module()
    script_dir = tmp_path / "doctest" / "DemoDoc"
    write_script(script_dir / "alpha.sh",
                 group="alpha",
                 source_doc="docs/source/tutorials/models/DemoDoc.md",
                 body="echo alpha-fail\nexit 5")
    write_script(script_dir / "beta.sh",
                 group="beta",
                 source_doc="docs/source/tutorials/models/DemoDoc.md",
                 body="echo beta-pass")

    exit_code = module.main([str(script_dir), "--keep-going"])

    assert exit_code == 1
    assert (script_dir / "logs" / "alpha.log").exists()
    assert (script_dir / "logs" / "beta.log").exists()
    output = capsys.readouterr().out
    assert "Failed: 1" in output
    assert "Succeeded: 1" in output
    assert "beta-pass" in output


def test_list_mode_only_lists_matching_scripts(tmp_path, capsys):
    module = load_module()
    script_dir = tmp_path / "doctest" / "DemoDoc"
    write_script(script_dir / "kimi-k2-single-node.sh",
                 group="kimi-k2-single-node",
                 source_doc="docs/source/tutorials/models/Kimi-K2-Thinking.md",
                 body="echo kimi")
    write_script(script_dir / "other-group.sh",
                 group="other-group",
                 source_doc="docs/source/tutorials/models/Other.md",
                 body="echo other")

    exit_code = module.main(
        [str(script_dir), "--group", "kimi-k2-single-node", "--list"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "kimi-k2-single-node.sh" in output
    assert "other-group.sh" not in output
    assert "log=" in output


def test_doc_filter_uses_source_doc_stem(tmp_path, capsys):
    module = load_module()
    script_dir = tmp_path / "doctest" / "DemoDoc"
    write_script(script_dir / "alpha.sh",
                 group="alpha",
                 source_doc="docs/source/tutorials/models/Kimi-K2-Thinking.md",
                 body="echo alpha")
    write_script(script_dir / "beta.sh",
                 group="beta",
                 source_doc="docs/source/tutorials/models/Other.md",
                 body="echo beta")

    exit_code = module.main([str(script_dir), "--doc", "Kimi-K2-Thinking", "--list"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "alpha.sh" in output
    assert "beta.sh" not in output


def test_summary_json_is_written(tmp_path):
    module = load_module()
    script_dir = tmp_path / "doctest" / "DemoDoc"
    summary_path = tmp_path / "artifacts" / "summary.json"
    write_script(script_dir / "alpha.sh",
                 group="alpha",
                 source_doc="docs/source/tutorials/models/DemoDoc.md",
                 body="echo alpha")

    exit_code = module.main(
        [str(script_dir), "--summary-json", str(summary_path)])

    assert exit_code == 0
    assert summary_path.exists()
    payload = summary_path.read_text(encoding="utf-8")
    assert '"total": 1' in payload
    assert '"group": "alpha"' in payload
