# SPDX-License-Identifier: Apache-2.0
import copy
import subprocess
import sys

import pytest
import yaml

from tests.e2e.schedule.config import ResourceSpec, parse_cases
from tests.e2e.schedule.deployment import materialize_command, materialize_env


def minimal_case():
    return {
        "name": "simple",
        "model": "model",
        "server_cmd": "vllm serve model --port 18000",
        "endpoint": {"host": "127.0.0.1", "port": 18000},
        "test_content": ["chat_completion"],
    }


def parse(tmp_path, value, resources=None):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(value))
    return parse_cases(path, resources or ResourceSpec(1, 4), tmp_path)


def test_import_is_cpu_safe():
    script = (
        "from tests.e2e.schedule import config, deployment; import sys; "
        "assert not {'torch', 'vllm', 'torch_npu'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_shorthand_and_cases_do_not_share_mutations(tmp_path):
    case = minimal_case()
    other = {**case, "name": "second"}
    cases = parse(tmp_path, {"test_cases": [case, other]})
    cases[0].test.prompts.append("one")
    assert not cases[1].test.prompts
    assert cases[0].deployment.services[0].name == "main"


def test_native_yaml_merge_and_explicit_duplicate(tmp_path):
    path = tmp_path / "anchor.yaml"
    path.write_text("""schema_version: 2
_base: &base
  name: base
  model: model
  server_cmd: vllm serve model --port 18000
  endpoint: {host: 127.0.0.1, port: 18000}
  test_content: [completion]
test_cases:
  - <<: *base
    name: overridden
""")
    assert parse_cases(path, ResourceSpec(1, 2), tmp_path)[0].name == "overridden"
    path.write_text(path.read_text() + "    name: duplicate\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        parse_cases(path, ResourceSpec(1, 2), tmp_path)


@pytest.mark.parametrize(
    "update, message",
    [
        ({"unexpected": 1}, "unknown fields"),
        ({"test_content": ["made_up"]}, "Unsupported test_content"),
        ({"envs": {"BAD": {}}}, "scalar"),
        ({"deployment": {}}, "mutually exclusive"),
        ({"test_content": []}, "must request"),
    ],
)
def test_invalid_case(tmp_path, update, message):
    with pytest.raises(ValueError, match=message):
        parse(tmp_path, {**minimal_case(), **update})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2.0"])
def test_invalid_resources(value):
    with pytest.raises(ValueError, match="positive integer"):
        ResourceSpec(value, 2)


def test_env_order_self_reference_and_nonrecursive_values():
    inherited = {"LD_LIBRARY_PATH": "/original", "DATA": "$HOME quoted 'text'"}
    templates = {"A": "$B", "B": "$DATA", "LD_LIBRARY_PATH": "/extra:$LD_LIBRARY_PATH"}
    assert materialize_env(templates, inherited, {}) == {
        "B": inherited["DATA"],
        "A": inherited["DATA"],
        "LD_LIBRARY_PATH": "/extra:/original",
    }
    with pytest.raises(ValueError, match="cycle"):
        materialize_env({"A": "$B", "B": "$A"}, {}, {})
    with pytest.raises(ValueError, match="Missing environment"):
        materialize_env({"A": "$MISSING"}, {}, {})


def test_quote_aware_command_keeps_substituted_data():
    env = {"MODEL": "/a model's path", "JSON": '{"text": "a \\"quote\\" $HOME"}', "HOSTS": "10.0.0.1 10.0.0.2"}
    argv = materialize_command(
        'vllm serve "$MODEL" --json "$JSON" --hosts $HOSTS --literal \'$TEXT\' --escaped \\$TEXT', env, {}
    )
    assert argv == (
        "vllm",
        "serve",
        env["MODEL"],
        "--json",
        env["JSON"],
        "--hosts",
        "10.0.0.1",
        "10.0.0.2",
        "--literal",
        "$TEXT",
        "--escaped",
        "$TEXT",
    )
    assert materialize_command('cmd "$HOSTS"', env, {}) == ("cmd", env["HOSTS"])
    with pytest.raises(ValueError, match="Embedded unquoted"):
        materialize_command("cmd --hosts=$HOSTS", env, {})


@pytest.mark.parametrize(
    "command",
    [
        "vllm serve m | cat",
        "vllm serve m > file",
        "vllm serve m && true",
        'cmd "$(id)"',
        "cmd `id`",
        "cmd {{ Unknown.value }}",
    ],
)
def test_reject_shell_and_unknown_context(command):
    with pytest.raises(ValueError):
        materialize_command(command, {}, {})


def test_bool_envs_are_lowercase(tmp_path):
    case = minimal_case()
    case["envs"] = {"ENABLED": True, "THREADS": 2}
    launch = parse(tmp_path, case)[0].deployment.services[0].launches[0]
    assert launch.env_template == {"ENABLED": "true", "THREADS": "2"}


def test_same_name_and_wrong_version(tmp_path):
    value = minimal_case()
    with pytest.raises(ValueError, match="Duplicate case"):
        parse(tmp_path, {"test_cases": [value, copy.deepcopy(value)]})
    with pytest.raises(ValueError, match="schema_version"):
        parse(tmp_path, {**value, "schema_version": 1})


def test_env_json_backslashes_and_empty_argv_are_data():
    value = r'{"text":"a \"quote\" and C:\\data", "value":"$VALUE"}'
    env = materialize_env({"JSON": value, "LITERAL": r"\$HOME"}, {"VALUE": "hello"}, {})
    assert env["JSON"] == value.replace("$VALUE", "hello")
    assert env["LITERAL"] == "$HOME"
    assert materialize_command('command "$EMPTY"', {"EMPTY": ""}, {}) == ("command", "")
