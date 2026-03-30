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

import pytest


def load_module():
    module_path = (Path(__file__).resolve().parents[3] / "tools" / "docs" /
                   "extract_ci_code_blocks.py")
    spec = importlib.util.spec_from_file_location("extract_ci_code_blocks",
                                                  module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_doc(tmp_path: Path, name: str, content: str) -> Path:
    doc_path = tmp_path / name
    doc_path.write_text(content, encoding="utf-8")
    return doc_path


def test_extracts_only_executable_blocks_and_preserves_order(tmp_path: Path):
    module = load_module()
    doc_path = write_doc(
        tmp_path, "sample.md", """
# Demo

```{code-block} bash
echo "display only"
```

```{code-block} bash
:name: first-step
:class: doc-exec
:group: smoke-a
export A=1
```

```{code-block} bash
:name: second-step
:class: doc-exec
:group: smoke-a
echo "$A"
```
""".strip() + "\n")

    blocks = module.collect_executable_blocks([doc_path])
    assert [block.name for block in blocks] == ["first-step", "second-step"]

    plans = module.build_script_plans(blocks, tmp_path / "out")
    assert len(plans) == 1
    script = module.render_shell_script(plans[0])
    assert "display only" not in script
    assert script.index("export A=1") < script.index('echo "$A"')


def test_supports_multiple_groups_in_one_document(tmp_path: Path):
    module = load_module()
    doc_path = write_doc(
        tmp_path, "multi-group.md", """
```{code-block} bash
:name: step-a
:class: doc-exec
:group: alpha
echo alpha
```

```{code-block} bash
:name: step-b
:class: doc-exec
:group: beta
echo beta
```
""".strip() + "\n")

    blocks = module.collect_executable_blocks([doc_path])
    plans = module.build_script_plans(blocks, tmp_path / "out")

    assert len(plans) == 2
    assert {plan.group for plan in plans} == {"alpha", "beta"}
    assert {Path(plan.output_path).name for plan in plans} == {"alpha.sh", "beta.sh"}


def test_generated_script_prints_block_metadata(tmp_path: Path):
    module = load_module()
    doc_path = write_doc(
        tmp_path, "logging.md", """
```{code-block} bash
:name: verify-service
:class: doc-exec
:group: smoke
curl http://127.0.0.1:8000/health
```
""".strip() + "\n")

    blocks = module.collect_executable_blocks([doc_path])
    plan = module.build_script_plans(blocks, tmp_path / "out")[0]
    script = module.render_shell_script(plan)

    assert "==> source doc:" in script
    assert "==> group: smoke" in script
    assert "==> code block name: verify-service" in script


def test_missing_group_fails_fast(tmp_path: Path):
    module = load_module()
    doc_path = write_doc(
        tmp_path, "invalid.md", """
```{code-block} bash
:name: missing-group
:class: doc-exec
echo fail
```
""".strip() + "\n")

    with pytest.raises(module.DocCodeExtractionError,
                       match="missing ':group:'"):
        module.collect_executable_blocks([doc_path])


def test_dry_run_writes_nothing_and_validates_format(tmp_path: Path, capsys):
    module = load_module()
    doc_path = write_doc(
        tmp_path, "dry-run.md", """
```{code-block} bash
:name: start-service
:class: doc-exec
:group: smoke
echo start
```
""".strip() + "\n")

    exit_code = module.main([
        "--doc",
        str(doc_path),
        "--output-dir",
        str(tmp_path / "generated"),
        "--dry-run",
    ])

    assert exit_code == 0
    assert not (tmp_path / "generated").exists()
    captured = capsys.readouterr()
    assert '"group": "smoke"' in captured.out


def test_default_output_path_uses_doctest_doc_stem_and_group_name(tmp_path: Path):
    module = load_module()
    doc_dir = tmp_path / "docs" / "source" / "tutorials" / "models"
    doc_dir.mkdir(parents=True)
    doc_path = write_doc(
        doc_dir, "Kimi-K2-Thinking.md", """
```{code-block} bash
:name: start-service
:class: doc-exec
:group: kimi-k2-single-node
echo start
```
""".strip() + "\n")

    blocks = module.collect_executable_blocks([doc_path])
    plan = module.build_script_plans(blocks)[0]

    assert plan.output_path.endswith(
        "doctest/Kimi-K2-Thinking/kimi-k2-single-node.sh")
