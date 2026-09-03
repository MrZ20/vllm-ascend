from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = REPO_ROOT / "tests/e2e/doctests/doctest_helper.py"
SPEC = spec_from_file_location("doctest_helper", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
doctest_helper = module_from_spec(SPEC)
SPEC.loader.exec_module(doctest_helper)


def test_extracts_bash_block():
    text = """<!-- doctest: sample -->
```bash
echo hello
```
"""
    assert doctest_helper.extract_block(text, "sample") == "echo hello\n"


def test_extracts_python_block():
    text = """<!-- doctest: sample -->

```python
print("hello")
```
"""
    assert doctest_helper.extract_block(text, "sample") == 'print("hello")\n'


def test_removes_markdown_indentation():
    text = """    <!-- doctest: sample -->
    ```bash
    echo hello
      echo nested
    ```
"""
    assert doctest_helper.extract_block(text, "sample") == "echo hello\n  echo nested\n"


def test_missing_marker_returns_none():
    assert doctest_helper.extract_block("plain text\n", "missing") is None


def test_duplicate_marker_is_an_error():
    text = """<!-- doctest: sample -->
```bash
true
```
<!-- doctest: sample -->
```bash
true
```
"""
    with pytest.raises(doctest_helper.DoctestError, match="Duplicate doctest marker"):
        doctest_helper.extract_block(text, "sample")


def test_unclosed_fence_is_an_error():
    text = """<!-- doctest: sample -->
```bash
true
"""
    with pytest.raises(doctest_helper.DoctestError, match="is not closed"):
        doctest_helper.extract_block(text, "sample")


def test_changed_detects_different_block():
    base = "<!-- doctest: sample -->\n```bash\necho base\n```\n"
    head = "<!-- doctest: sample -->\n```bash\necho head\n```\n"
    assert doctest_helper.block_changed(base, head, "sample", "doc.md")


def test_changed_ignores_prose_changes():
    base = "old prose\n<!-- doctest: sample -->\n```bash\necho same\n```\n"
    head = "new prose\n<!-- doctest: sample -->\n```bash\necho same\n```\n"
    assert not doctest_helper.block_changed(base, head, "sample", "doc.md")


def test_changed_preserves_blank_lines_inside_fence():
    base = "<!-- doctest: sample -->\n```bash\necho same\n```\n"
    head = "<!-- doctest: sample -->\n```bash\necho same\n\n```\n"
    assert doctest_helper.block_changed(base, head, "sample", "doc.md")


def test_changed_detects_added_marker():
    head = "<!-- doctest: sample -->\n```bash\ntrue\n```\n"
    assert doctest_helper.block_changed(None, head, "sample", "doc.md")


def test_changed_rejects_marker_missing_from_both_refs():
    with pytest.raises(doctest_helper.DoctestError, match="does not exist at either ref"):
        doctest_helper.block_changed("prose\n", "different prose\n", "sample", "doc.md")


def test_loads_simple_extra_scalars():
    text = """site_name: docs
extra:
  release_version: "1.2.3"
  bare_version: v1.2.3 # comment
  nested:
    value: ignored
plugins: []
"""
    assert doctest_helper.load_extra(text) == {
        "release_version": "1.2.3",
        "bare_version": "v1.2.3",
    }


def test_release_values_selects_release_stack_only():
    assert doctest_helper.release_values(
        {
            "vllm_version": "v1",
            "vllm_ascend_version": "v1",
            "release_cann_version": "1",
            "stable_vllm_ascend_version": "v1",
            "main_cann_version": "2",
            "docs_lang": "en",
        }
    ) == {
        "vllm_version": "v1",
        "vllm_ascend_version": "v1",
        "release_cann_version": "1",
    }


@pytest.mark.parametrize(
    "key",
    [
        "release_vllm_version",
        "release_vllm_ascend_version",
        "release_triton_ascend_version",
        "release_new_dependency_version",
        "vllm_version",
        "vllm_ascend_version",
    ],
)
def test_release_changed_detects_runtime_release_key(key):
    base = f"extra:\n  {key}: old\n"
    head = f"extra:\n  {key}: new\n"
    assert doctest_helper.release_changed(base, head)


@pytest.mark.parametrize("key", ["stable_vllm_ascend_version", "main_cann_version", "docs_lang"])
def test_release_changed_ignores_documentation_metadata(key):
    base = f"extra:\n  {key}: old\n"
    head = f"extra:\n  {key}: new\n"
    assert not doctest_helper.release_changed(base, head)


def test_release_changed_detects_added_or_deleted_release_key():
    without_key = "extra:\n  docs_lang: en\n"
    with_key = "extra:\n  release_new_dependency_version: 1.0\n"
    assert doctest_helper.release_changed(without_key, with_key)
    assert doctest_helper.release_changed(with_key, without_key)


def test_expands_known_macro():
    content = 'pip install "package=={{ release_version }}"\n'
    assert doctest_helper.expand_macros(content, {"release_version": "1.2.3"}, "sample") == (
        'pip install "package==1.2.3"\n'
    )


def test_unknown_macro_is_an_error():
    with pytest.raises(doctest_helper.DoctestError, match="Unknown mkdocs.yml macro"):
        doctest_helper.expand_macros("echo {{ missing }}\n", {}, "sample")
