# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest
import requests

from tests.e2e import utils


@pytest.fixture
def http_server():
    servers = []

    def create(status=200):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(self.server.status)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.status = status
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server, f"http://127.0.0.1:{server.server_port}/health"

    yield create
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_503_is_not_ready(http_server):
    _, url = http_server(503)
    with pytest.raises(TimeoutError, match=url):
        utils.wait_for_http_ready(url, 0.1, poll_interval=0.01)


def test_same_host_different_ports_are_checked_separately(http_server):
    _, first = http_server()
    second_server, second = http_server(503)
    with pytest.raises(TimeoutError) as exc:
        utils.wait_for_http_targets([first, second], 0.1, poll_interval=0.01)
    assert second in str(exc.value)
    assert first not in str(exc.value)
    second_server.status = 200
    utils.wait_for_http_targets([first, second], 1)


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.Timeout])
def test_http_connection_errors_timeout_with_all_urls(monkeypatch, error):
    get = Mock(side_effect=error)
    monkeypatch.setattr(utils.requests, "get", get)
    urls = ["http://localhost:1/health", "http://localhost:2/health"]
    with pytest.raises(TimeoutError) as exc:
        utils.wait_for_http_targets(urls, 0.05, poll_interval=0.01)
    assert all(url in str(exc.value) for url in urls)
    assert all(call.kwargs["timeout"] > 0 for call in get.call_args_list)


def test_always_check_rechecks_previously_ready_urls(monkeypatch):
    responses = iter([200, 503, 503, 200])
    process = Mock()
    process.poll.return_value = None

    def get(url, **kwargs):
        try:
            status = next(responses)
        except StopIteration:
            process.poll.return_value = 7
            status = 503
        return Mock(status_code=status)

    monkeypatch.setattr(utils.requests, "get", get)
    with pytest.raises(RuntimeError, match="code 7"):
        utils.wait_for_http_targets(
            ["first", "second"], 1, poll_processes=[process], always_check=True, poll_interval=0
        )


@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("exit_code", [0, 7, -signal.SIGTERM, -signal.SIGKILL])
def test_readiness_detects_early_process_exit_even_on_http_response(http_server, status, exit_code):
    _, url = http_server(status)
    script = f"raise SystemExit({exit_code})" if exit_code >= 0 else f"import os; os.kill(os.getpid(), {-exit_code})"
    process = utils._ManagedProcess([sys.executable, "-c", script])
    process.start()
    try:
        process.proc.wait(timeout=5)
        with pytest.raises(RuntimeError, match=f"before readiness with code {exit_code}"):
            utils.wait_for_http_ready(url, 0.2, process=process)
    finally:
        process.shutdown()


def test_wait_for_unready(http_server):
    server, url = http_server()
    with pytest.raises(TimeoutError, match=url):
        utils.wait_for_http_unready(url, 0.05, poll_interval=0.01)
    server.status = 503
    utils.wait_for_http_unready(url, 1)


@pytest.mark.parametrize("exit_code", [0, 7, -signal.SIGTERM, -signal.SIGKILL])
def test_unready_handles_local_exit_before_http(monkeypatch, exit_code):
    process = Mock()
    process.poll.return_value = exit_code
    get = Mock(side_effect=AssertionError("An exited local worker should not wait for HTTP"))
    monkeypatch.setattr(utils.requests, "get", get)
    if exit_code == 0:
        utils.wait_for_http_unready("http://127.0.0.1:1/health", 60, process=process)
    else:
        with pytest.raises(RuntimeError, match=f"code {exit_code}"):
            utils.wait_for_http_unready("http://127.0.0.1:1/health", 60, process=process)
    get.assert_not_called()


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("status", [200, 503])
def test_unready_checks_local_exit_during_http(monkeypatch, exit_code, status):
    process = Mock()
    process.poll.side_effect = [None, exit_code]
    get = Mock(return_value=Mock(status_code=status))
    monkeypatch.setattr(utils.requests, "get", get)
    if exit_code == 0:
        utils.wait_for_http_unready("http://127.0.0.1:1/health", 60, process=process)
    else:
        with pytest.raises(RuntimeError, match=f"code {exit_code}"):
            utils.wait_for_http_unready("http://127.0.0.1:1/health", 60, process=process)
    get.assert_called_once()


def test_process_shutdown_is_idempotent_and_kills_group(tmp_path):
    child_file = tmp_path / "child"
    script = """
import pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    process = utils._ManagedProcess([sys.executable, "-c", script, str(child_file)], shutdown_timeout=0.05)
    process.start()
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_file.read_text())
        assert os.getpgid(process.proc.pid) == process.proc.pid
        assert os.getpgid(child_pid) == process.proc.pid
        process.shutdown()
        process.shutdown()
        assert process.proc.poll() == -signal.SIGKILL
        assert utils._find_pgid_members(process.pgid) == []
    finally:
        process.shutdown()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="survivor scanning uses Linux /proc")
def test_cleanup_survives_root_exit(tmp_path):
    child_file = tmp_path / "child"
    script = """
import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
"""
    process = utils._ManagedProcess([sys.executable, "-c", script, str(child_file)])
    process.start()
    try:
        process.proc.wait(timeout=5)
        child_pid = int(child_file.read_text())
        assert child_pid in utils._find_pgid_members(process.pgid)
        process.shutdown()
        assert utils._find_pgid_members(process.pgid) == []
    finally:
        process.shutdown()


@pytest.mark.parametrize("failure", ["exit", "spawn", "timeout"])
def test_group_failure_cleans_up_siblings(failure):
    first = utils._ManagedProcess([sys.executable, "-c", "import time; time.sleep(60)"], shutdown_timeout=0.1)
    command = ["/nonexistent/e2e-test-command"] if failure == "spawn" else [sys.executable, "-c", "raise SystemExit(9)"]
    second = utils._ManagedProcess(command)
    try:
        with pytest.raises((RuntimeError, FileNotFoundError, TimeoutError)):
            utils.RemoteServerGroup(
                [first, second], ["http://127.0.0.1:1/health"], timeout=0 if failure == "timeout" else 5
            )
        assert first.proc.poll() is not None
        if second.proc is not None:
            assert second.proc.poll() is not None
    finally:
        first.shutdown()
        second.shutdown()


def test_group_shutdown_starts_all_siblings_before_waiting():
    barrier = threading.Barrier(2)
    processes = [Mock(), Mock()]
    for process in processes:
        process.shutdown.side_effect = lambda: barrier.wait(timeout=2)
    group = object.__new__(utils.RemoteServerGroup)
    group.processes = processes
    group._shutdown()


def test_command_redaction_does_not_change_execution(caplog):
    args = [sys.executable, "-c", "pass", "--api-key=secret1", "secret2", "--hf_token", "secret3"]
    process = utils._ManagedProcess(args)
    with caplog.at_level("INFO"):
        process.start()
        process.proc.wait(timeout=5)
        process.shutdown()
    assert process.proc.args == args
    assert "***" in caplog.text
    assert all(secret not in caplog.text for secret in ("secret1", "secret2", "secret3"))


@pytest.fixture
def lightweight_cli(monkeypatch):
    # Lifecycle tests do not need to import a model engine or initialize a device.
    def parse(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("model")
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", "-p", type=int, default=8000)
        parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
        parser.add_argument("--prefill-context-parallel-size", "-pcp", type=int, default=1)
        parser.add_argument("--uds")
        return parser.parse_known_args(args)[0]

    monkeypatch.setattr(utils, "_parse_serve_args", parse)
    monkeypatch.setattr(utils, "_vllm_shutdown_timeout", lambda args: 0.1)


@pytest.mark.parametrize(
    "args", [["--port=12345", "--host=127.0.0.2"], "vllm serve model --port 12345 --host 127.0.0.2"]
)
def test_server_endpoint_comes_from_executed_cli(monkeypatch, lightweight_cli, args):
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    with utils.RemoteOpenAIServer("model", args, server_port=9999, wait_for_ready=False) as server:
        assert server.host == "127.0.0.2"
        assert server.port == 12345
        assert server.url_for("health") == "http://127.0.0.2:12345/health"
        with server.get_client() as client:
            assert str(client.base_url) == "http://127.0.0.2:12345/v1/"


def test_legacy_port_is_written_to_command(monkeypatch, lightweight_cli):
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    with utils.RemoteOpenAIServer("model", [], auto_port=False, server_port=12345, wait_for_ready=False) as server:
        assert server.port == 12345
        assert server._process.command[-2:] == ["--port", "12345"]


@pytest.mark.parametrize("failure", ["spawn", "readiness"])
def test_server_startup_failure_shuts_down(monkeypatch, lightweight_cli, failure):
    process = utils._ManagedProcess([sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(utils.RemoteOpenAIServer, "_make_process", lambda *args: process)
    if failure == "spawn":
        start = process.start

        def fail_after_spawn():
            start()
            raise RuntimeError("launch failed after spawn")

        monkeypatch.setattr(process, "start", fail_after_spawn)
    try:
        with pytest.raises((RuntimeError, TimeoutError)):
            utils.RemoteOpenAIServer("model", ["--port", "12345"], max_wait_seconds=0)
        assert process.proc.poll() is not None
    finally:
        process.shutdown()


@pytest.mark.parametrize("server_class", [utils.RemotePDServer, utils.RemoteEPDServer])
def test_device_allocation_respects_visible_devices(monkeypatch, lightweight_cli, server_class):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8,9,12,13")
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    monkeypatch.setattr(utils.RemoteServerGroup, "wait_all_ready", lambda *args: None)
    args = [["model", "--port", str(port), "-tp", "2"] for port in (12345, 12346)]
    with server_class(args) as server:
        assert [p.env["ASCEND_RT_VISIBLE_DEVICES"] for p in server.processes] == ["8,9", "12,13"]
        assert len(set(server.health_urls)) == 2
    with pytest.raises(ValueError, match="only 1 remain"):
        server_class(args, env_dict={"ASCEND_RT_VISIBLE_DEVICES": "12,13,14"})


@pytest.mark.parametrize(
    "args, expected",
    [
        (["-tp", "2", "-dp", "4"], 8),
        (["--tensor-parallel-size=2", "--data-parallel-size=8", "--data-parallel-size-local=2"], 4),
        (["--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--data-parallel-size", "1"], 4),
        (
            [
                "--tensor-parallel-size=2",
                "--pipeline-parallel-size=2",
                "--data-parallel-size=8",
                "--data-parallel-size-local=2",
            ],
            8,
        ),
        (["-tp", "2", "-pp", "2", "-dp", "8", "-dpl", "2"], 8),
        (["-tp=2", "-pp=2", "-dp=8", "-dpl=2"], 8),
        (["--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--prefill-context-parallel-size", "2"], 8),
        (
            [
                "--tensor-parallel-size=2",
                "--pipeline-parallel-size=2",
                "--prefill-context-parallel-size=2",
                "--data-parallel-size=8",
                "--data-parallel-size-local=2",
            ],
            16,
        ),
        (["-tp", "2", "-pp", "2", "-pcp", "2", "-dp", "8", "-dpl", "2"], 16),
        (["-tp=2", "-pp=2", "-pcp=2", "-dp=8", "-dpl=2"], 16),
    ],
)
def test_pd_required_devices(args, expected):
    assert utils._get_pd_server_required_devices(args) == expected


def test_assignment_reserves_later_explicit_devices():
    args = [["m", "-tp", "2"], ["m", "-tp", "2"]]
    assert utils._allocate_server_devices(args, [{}, {"ASCEND_RT_VISIBLE_DEVICES": "4,6"}], "4,6,8,10") == [
        "8,10",
        "4,6",
    ]


@pytest.mark.parametrize("assignment", ["", "4,4", "4,12"])
def test_assignment_rejects_invalid_explicit_devices(assignment):
    with pytest.raises(ValueError):
        utils._allocate_server_devices([["m"]], [{"ASCEND_RT_VISIBLE_DEVICES": assignment}], "4,6,8,10")


def test_per_server_env_and_deferred_group_readiness(monkeypatch, lightweight_cli, tmp_path):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,6,8,10")
    started = Mock()
    wait = Mock()
    monkeypatch.setattr(utils._ManagedProcess, "start", started)
    monkeypatch.setattr(utils.RemoteServerGroup, "wait_all_ready", wait)
    args = [["model", "--port", "12345"], ["model", "--headless"]]
    with utils.RemotePDServer(
        args,
        env_dict={"COMMON": "yes"},
        per_server_envs=[
            {"ASCEND_RT_VISIBLE_DEVICES": "8", "VLLM_WORKER_MULTIPROC_METHOD": "fork"},
            {"ASCEND_RT_VISIBLE_DEVICES": "4,6"},
        ],
        working_dirs=[str(tmp_path), None],
        log_files=[tmp_path / "first.log", None],
        health_urls=["http://remote:12345/health"],
        wait_for_ready=False,
    ) as server:
        assert started.call_count == 2
        wait.assert_not_called()
        assert server.processes[0].env["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
        assert server.processes[1].env["COMMON"] == "yes"
        assert server.processes[0].cwd == str(tmp_path)
        assert server.health_urls == ["http://remote:12345/health"]


def test_log_file_and_cwd(tmp_path):
    logfile = tmp_path / "process.log"
    process = utils._ManagedProcess(
        [sys.executable, "-c", "import os; print(os.getcwd()); print('done')"], cwd=str(tmp_path), log_file=logfile
    )
    process.start()
    process.proc.wait(timeout=5)
    process.shutdown()
    assert str(tmp_path) in logfile.read_text()
    assert "done" in logfile.read_text()


def test_cleanup_preserves_original_error_and_attempts_all_siblings():
    processes = [Mock(), Mock()]
    processes[0].shutdown.side_effect = RuntimeError("cleanup failed")
    with (
        pytest.raises(ValueError, match="test failed") as error,
        utils.RemoteServerGroup(processes, [], timeout=1, wait_for_ready=False),
    ):
        raise ValueError("test failed")
    assert "cleanup" in error.value.__notes__[0]
    assert all(process.shutdown.called for process in processes)


@pytest.mark.parametrize("pcp_size", [0, -1])
def test_pd_required_devices_rejects_nonpositive_pcp(pcp_size):
    with pytest.raises(ValueError, match="--prefill-context-parallel-size must be positive"):
        utils._get_pd_server_required_devices(["-pcp", str(pcp_size)])


@pytest.mark.parametrize("parallel_option", ["-pp", "-pcp"])
@pytest.mark.parametrize("server_class", [utils.RemotePDServer, utils.RemoteEPDServer])
def test_parallel_device_allocation(monkeypatch, lightweight_cli, server_class, parallel_option):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11,12,13,14,15")
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    monkeypatch.setattr(utils.RemoteServerGroup, "wait_all_ready", lambda *args: None)
    args = [["model", "--port", str(port), "-tp", "2", parallel_option, "2"] for port in (12345, 12346)]
    with server_class(args) as server:
        assert [process.env["ASCEND_RT_VISIBLE_DEVICES"] for process in server.processes] == [
            "8,9,10,11",
            "12,13,14,15",
        ]


def test_proxy_client_compatibility(monkeypatch):
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    monkeypatch.setattr(utils, "wait_for_http_ready", lambda *args, **kwargs: None)
    with utils.DisaggEpdProxy("--host=127.0.0.2 --port=12345") as proxy:
        assert proxy.url_root == "http://127.0.0.2:12345"
        with proxy.get_client() as client:
            assert str(client.base_url) == proxy.url_for("v1") + "/"
        assert str(proxy.get_async_client().base_url) == proxy.url_for("v1") + "/"
    assert not issubclass(utils.DisaggEpdProxy, utils.RemoteEPDServer)
    assert not issubclass(utils.DisaggPDProxy, utils.RemotePDServer)


def test_production_cli_parser_and_shutdown_timeout(monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    with utils.RemoteOpenAIServer("model", ["--port=12345", "--shutdown-timeout", "7"], wait_for_ready=False) as server:
        assert server.port == 12345
        assert server._process.shutdown_timeout >= 7 + utils._PROCESS_SHUTDOWN_TIMEOUT


def test_proxy_startup_failure_cleans_up(monkeypatch):
    processes = []
    original = utils._ManagedProcess.start

    def record_start(process):
        processes.append(process)
        original(process)

    monkeypatch.setattr(utils._ManagedProcess, "start", record_start)
    with pytest.raises(TimeoutError):
        utils.RemoteProxy(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            host="127.0.0.1",
            port=12345,
            health_path="health",
            timeout=0,
        )
    assert len(processes) == 1
    assert processes[0].proc.poll() is not None


def test_production_cli_auto_port_and_uds(monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.setattr(utils._ManagedProcess, "start", lambda self: None)
    with utils.RemoteOpenAIServer("model", [], wait_for_ready=False) as server:
        assert server._process.command[-2:] == ["--port", str(server.port)]
        assert server.url_for("health") == f"http://127.0.0.1:{server.port}/health"
    with utils.RemoteOpenAIServer("model", ["--uds=/tmp/e2e-vllm.sock"], wait_for_ready=False) as server:
        assert server.uds == "/tmp/e2e-vllm.sock"
        assert "--port" not in server._process.command
        with server.get_client() as client:
            assert str(client.base_url) == "http://localhost/v1/"


@pytest.mark.parametrize("request_timeout, expected", [(0, 60), (80, 95)])
def test_shutdown_timeout_supports_older_vllm(monkeypatch, request_timeout, expected):
    engine = Mock(utils=object())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine)
    assert utils._vllm_shutdown_timeout(argparse.Namespace(shutdown_timeout=request_timeout)) == expected


def test_shutdown_timeout_uses_upstream_engine_budget(monkeypatch):
    get_timeout = Mock(return_value=42)
    engine = Mock(utils=Mock(get_engine_process_shutdown_timeout=get_timeout))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine)
    assert utils._vllm_shutdown_timeout(argparse.Namespace(shutdown_timeout=7)) == 57
    get_timeout.assert_called_once_with(7, 7)


def test_explicit_assignment_must_fit_parallel_workers():
    with pytest.raises(ValueError, match="requires 4 devices"):
        utils._allocate_server_devices(
            [["model", "-tp", "2", "-pp", "2"]], [{"ASCEND_RT_VISIBLE_DEVICES": "4,5"}], "4,5,6,7"
        )
