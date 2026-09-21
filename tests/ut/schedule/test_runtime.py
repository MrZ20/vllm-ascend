# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import socket
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import pytest

from tests.e2e import utils
from tests.e2e.schedule import runtime
from tests.e2e.schedule.config import ResourceSpec
from tests.e2e.schedule.deployment import build_plan
from tests.ut.schedule.test_config import parse
from tests.ut.schedule.test_deployment import cluster, pd_case, service

HTTP_SCRIPT = """
import http.server, socket, sys, time
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
server = http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler)
if len(sys.argv) > 2:
    deadline = time.monotonic() + 3
    while True:
        try:
            socket.create_connection(('127.0.0.1', int(sys.argv[2])), timeout=.1).close()
            break
        except OSError:
            if time.monotonic() > deadline: raise
            time.sleep(.01)
server.serve_forever()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_launcher_starts_peers_before_wait_and_cleans_owned_processes(tmp_path, monkeypatch):
    ports = [free_port(), free_port(), free_port()]
    value = pd_case("explicit", "explicit")
    value["deployment"]["services"] = {
        "p": service("explicit", [0], ports[0], 4),
        "d": service("explicit", [0], ports[1], 5),
    }
    value["deployment"]["routing"]["proxy"].update(
        command=utils.shlex.join([sys.executable, "-c", HTTP_SCRIPT, str(ports[2])]),
        endpoint={"host": "127.0.0.1", "port": ports[2]},
    )
    local = replace(cluster(0, 1), node_ips=("127.0.0.1",))
    plan = build_plan(parse(tmp_path, value)[0].deployment, local, {})
    original = utils._ManagedProcess
    launched = []

    def process(command, *args, **kwargs):
        if command[:2] == ["vllm", "serve"]:
            port = int(command[command.index("--port") + 1])
            peer = ports[1] if port == ports[0] else ports[0]
            command = [sys.executable, "-c", HTTP_SCRIPT, str(port), str(peer)]
        result = original(command, *args, **kwargs)
        launched.append(result)
        return result

    monkeypatch.setattr(utils, "_ManagedProcess", process)
    monkeypatch.setattr(
        utils,
        "_parse_serve_args",
        lambda args: argparse.Namespace(host="127.0.0.1", port=int(args[args.index("--port") + 1]), uds=None),
    )
    monkeypatch.setattr(utils, "_vllm_shutdown_timeout", lambda args: 0.2)
    with runtime.Launcher(tmp_path / "logs", local.current_device_pool, 5).run(plan) as running:
        assert len(running.processes) == 3
        running.assert_healthy()
        assert all(p.log_file.is_file() for p in running.processes.values())
    assert all(p.proc.poll() is not None for p in launched)
    assert (tmp_path / "logs/launch_effective.json").is_file()


def test_coordinator_identity_failure_and_missing_node(tmp_path):
    first = runtime.Coordinator(tmp_path, "run", "case", 0, 2, "digest", timeout=0.01)
    second = runtime.Coordinator(tmp_path, "run", "case", 1, 2, "digest", timeout=0.01)
    second.write("node-1.result", status="failed", error="failed to spawn")
    with pytest.raises(RuntimeError, match="failed to spawn"):
        first.wait(["node-1.ready"])
    other_case = runtime.Coordinator(tmp_path, "run", "second-case", 0, 2, "digest", timeout=0.01)
    with pytest.raises(TimeoutError):
        other_case.wait(["node-1.result"])
    mismatch = runtime.Coordinator(tmp_path, "run", "case", 0, 2, "different", timeout=0.01)
    with pytest.raises(RuntimeError, match="identity"):
        mismatch.read("node-1.result")


@pytest.mark.parametrize("failed_node", [None, 0, 1])
def test_shared_case_waits_for_all_cleanup_before_final(tmp_path, monkeypatch, failed_node):
    value = pd_case("explicit", "explicit")
    value["deployment"] = {"services": {"main": service("explicit", [0, 1], 18000, 4)}}
    case = parse(tmp_path, value, ResourceSpec(2, 4))[0]
    monkeypatch.setattr(runtime, "preflight", lambda *args: (4, 5, 6, 7))
    monkeypatch.setattr(runtime, "run_test_process", lambda *args: {"status": "passed", "benchmarks": {}})
    cleanup = []

    class FakeLauncher:
        def __init__(self, *args, **kwargs):
            pass

        @contextmanager
        def run(self, plan):
            try:
                yield runtime.RunningDeployment({})
            finally:
                cleanup.append(plan.node.index)
                if plan.node.index == failed_node:
                    raise RuntimeError("cleanup failed")

    monkeypatch.setattr(runtime, "Launcher", FakeLauncher)
    errors = []

    def run(index):
        try:
            runtime.run_case(
                case, cluster(index, 2), tmp_path, tmp_path / "logs", "run", "case", "digest", tmp_path / "shared"
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert set(cleanup) == {0, 1}
    final = json.loads((tmp_path / "shared/run/cases/case/final.json").read_text())
    assert final["status"] == ("passed" if failed_node is None else "failed")
    assert len(errors) == (0 if failed_node is None else 2)


def test_resolver_uses_actual_visible_count(monkeypatch):
    monkeypatch.setattr("psutil.net_if_addrs", lambda: {})
    resources, resolved = runtime.resolve_cluster({}, (4, 6))
    assert resources == ResourceSpec(1, 2)
    assert resolved.current_device_pool == (4, 6)
    with pytest.raises(ValueError, match="differs"):
        runtime.resolve_cluster({"NPU_PER_NODE": "16"}, (4, 6))
    with pytest.raises(ValueError, match="disagree"):
        runtime.resolve_cluster({"NUM_NODES": "2", "LWS_GROUP_SIZE": "3"}, (4, 6))


@pytest.mark.parametrize("index", [0, 1])
def test_interrupted_startup_does_not_wait_for_absent_peer(tmp_path, monkeypatch, index):
    value = pd_case("explicit", "explicit")
    value["deployment"] = {"services": {"main": service("explicit", [0, 1], 18000, 4)}}
    case = parse(tmp_path, value, ResourceSpec(2, 4))[0]
    monkeypatch.setattr(runtime, "preflight", lambda *args: (4, 5, 6, 7))
    cleaned = []

    @contextmanager
    def interrupted_start(self, plan):
        try:
            raise SystemExit(143)
            yield  # pragma: no cover - contextmanager protocol
        finally:
            cleaned.append(plan.node.index)

    monkeypatch.setattr(runtime.Launcher, "run", interrupted_start)
    wait = Mock(side_effect=AssertionError("Failed node must not wait for an absent peer"))
    monkeypatch.setattr(runtime.Coordinator, "wait", wait)
    with pytest.raises(SystemExit) as exc:
        runtime.run_case(case, cluster(index, 2), tmp_path, tmp_path / "logs", "run", "case", "digest", tmp_path)
    assert exc.value.code == 143
    assert cleaned == [index]
    wait.assert_not_called()
    result = json.loads((tmp_path / f"run/cases/case/node-{index}.result.json").read_text())
    assert result["status"] == "failed"
    if index == 0:
        assert json.loads((tmp_path / "run/cases/case/final.json").read_text())["status"] == "failed"


def test_healthy_rejects_clean_exit_during_tests():
    with pytest.raises(RuntimeError, match="exit_code=0"):
        runtime.RunningDeployment({"server": Mock(poll=lambda: 0)}).assert_healthy()


def test_redaction():
    value = runtime.redact(
        {"argv": ["vllm", "serve", "m", "--api-key", "secret"], "env_overrides": {"HF_TOKEN": "token", "MODEL": "m"}}
    )
    assert value["argv"][-1] == "***"
    assert value["env_overrides"] == {"HF_TOKEN": "***", "MODEL": "m"}


@pytest.mark.parametrize("legacy", [False, True])
def test_common_kv_pool_inputs_preserve_legacy_config(tmp_path, legacy):
    from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
    from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import create_kv_pool_manager

    config = MooncakeKVPoolConfig(config={"local_hostname": "${LOCAL_IP}"}, master_port=51001, metrics_port=51002)
    ips = ["10.0.0.1", "10.0.0.2"]
    values = (
        {"config": argparse.Namespace(kv_pool=config, cluster_ips=ips)}
        if legacy
        else {"kv_pool": config, "cluster_ips": ips}
    )
    manager = create_kv_pool_manager(**values, current_node_index=1, log_root=tmp_path)
    manager._write_config()
    stored = json.loads(manager.config_path.read_text())
    assert stored["local_hostname"] == "10.0.0.2"
    assert stored["master_server_address"] == "10.0.0.1:51001"
    assert manager.server_envs["MOONCAKE_CONFIG_PATH"] == str(manager.config_path)
    manager.process = Mock(poll=lambda: 0, returncode=0)
    with pytest.raises(RuntimeError, match="code 0"):
        manager._wait_ready(timeout=1)


def test_bounded_test_process_failure_cleans_its_group(tmp_path, monkeypatch):
    from tests.ut.schedule.test_config import minimal_case

    case = parse(tmp_path, minimal_case())[0]
    plan = build_plan(case.deployment, cluster(0, 1), {})
    real_process = utils._ManagedProcess
    processes = []

    def process(command, *args, **kwargs):
        child = real_process([sys.executable, "-c", "import time; time.sleep(20)"], *args, **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(utils, "_ManagedProcess", process)
    with pytest.raises(TimeoutError, match="TEST_TIMEOUT"):
        runtime.run_test_process(case, plan, runtime.RunningDeployment({}), tmp_path, tmp_path, timeout=0.05)
    assert processes[0].poll() is not None
    assert not (tmp_path / "test_input.json").exists()


def test_published_stop_precedes_local_exit_probe(tmp_path):
    coord = runtime.Coordinator(tmp_path, "run", "case", 1, 2, "digest", timeout=0.01)
    coord.write("stop", status="passed")
    unhealthy = runtime.RunningDeployment({"worker": Mock(poll=lambda: 0)})
    assert coord.wait(["stop"], unhealthy.assert_healthy)[0]["status"] == "passed"


def test_test_phase_observes_remote_node_failure(tmp_path):
    coord = runtime.Coordinator(tmp_path, "run", "case", 0, 2, "digest")
    coord.write("node-1.result", status="failed", error="worker crashed")
    running = runtime.RunningDeployment({}, check_peers=coord.check_failures)
    with pytest.raises(RuntimeError, match="worker crashed"):
        running.assert_healthy()


def test_pd_request_tokenize_and_metrics_use_separate_endpoints(tmp_path):
    case = parse(tmp_path, pd_case(), ResourceSpec(3, 4))[0]
    plan = build_plan(case.deployment, cluster(0), {})
    payload = runtime._test_payload(case, plan)
    assert payload["completion"]["port"] == 19000
    assert payload["tokenize"]["port"] == 18000
    assert payload["metrics"]["port"] == 18100
