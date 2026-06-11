"""Unit tests for the external_dp start/stop synchronization logic.

These cover the readiness/liveness debouncing and the fixed timeout budgets used
by the external DP framework. They mock the HTTP probes, so they need no
NPU/cluster.
"""

import itertools
from pathlib import Path

import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts import runtime, utils
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    RankInfo,
)


def _rank(local_rank: int = 0, port: int = 8000) -> RankInfo:
    return RankInfo(
        node_index=0,
        role="worker",
        local_rank=local_rank,
        dp_rank=local_rank,
        host="127.0.0.1",
        port=port,
        visible_devices="0",
        dp_size=1,
        dp_size_local=1,
        tp_size=1,
        cp_size=1,
        sp_size=1,
        pp_size=1,
        dp_address="127.0.0.1",
        dp_rpc_port=12321,
        port_start=8000,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(runtime.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(utils.time, "sleep", lambda *a, **k: None)


# --- wait_http_unready debounce (worker waiting for the master to stop) ---


def test_wait_http_unready_returns_after_threshold(monkeypatch):
    calls = {"n": 0}

    def fake(url, timeout=None):
        calls["n"] += 1
        return False

    monkeypatch.setattr(utils, "is_http_ready", fake)
    utils.wait_http_unready("http://x", timeout=100, failure_threshold=3)
    assert calls["n"] == 3


def test_wait_http_unready_ignores_transient_blip(monkeypatch):
    # Alternating down/up never reaches 3 consecutive failures, so a transient
    # blip must NOT be read as "stopped" -> the wait should time out instead.
    states = itertools.cycle([False, True])
    monkeypatch.setattr(utils, "is_http_ready", lambda url, timeout=None: next(states))
    with pytest.raises(TimeoutError):
        utils.wait_http_unready("http://x", timeout=0.2, failure_threshold=3)


# --- wait_ranks_ready (startup readiness + post-ready liveness) ---


def test_wait_ranks_ready_returns_when_all_ready(monkeypatch):
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: True)
    runtime.wait_ranks_ready([_rank()], timeout=5)


def test_wait_ranks_ready_times_out(monkeypatch):
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(TimeoutError):
        runtime.wait_ranks_ready([_rank()], timeout=0)


def test_wait_ranks_ready_tolerates_single_flap(monkeypatch):
    a = _rank(local_rank=0, port=8000)
    b = _rank(local_rank=1, port=8001)
    counts = {"a": 0, "b": 0}

    def fake(url, timeout=None):
        if url.endswith(":8000/health"):
            counts["a"] += 1
            return counts["a"] != 2  # ready, one flap on the 2nd probe, ready again
        counts["b"] += 1
        return counts["b"] >= 3  # b only becomes ready on its 3rd probe

    monkeypatch.setattr(runtime, "is_http_ready", fake)
    runtime.wait_ranks_ready([a, b], timeout=100)  # must return, not raise


def test_wait_ranks_ready_raises_after_sustained_unhealthy(monkeypatch):
    a = _rank(local_rank=0, port=8000)
    b = _rank(local_rank=1, port=8001)
    counts = {"a": 0}

    def fake(url, timeout=None):
        if url.endswith(":8000/health"):
            counts["a"] += 1
            return counts["a"] == 1  # ready once, then permanently down
        return False  # b never ready -> keeps the loop running

    monkeypatch.setattr(runtime, "is_http_ready", fake)
    with pytest.raises(RuntimeError, match="unhealthy after ready"):
        runtime.wait_ranks_ready([a, b], timeout=100)


class _FakeProc:
    def __init__(self, returncode=None, pid=4321):
        self.pid = pid
        self._returncode = returncode

    def poll(self):
        return self._returncode


def test_wait_ranks_ready_detects_local_process_exit(monkeypatch):
    a = _rank()
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(RuntimeError, match="exited before ready"):
        runtime.wait_ranks_ready([a], timeout=100, rank_processes=[(_FakeProc(returncode=1), a, Path("rank.log"))])


# --- failure messages carry pinpointing detail ---


def test_rank_process_exit_message_includes_log_tail(monkeypatch, tmp_path):
    a = _rank()
    log = tmp_path / "rank.log"
    log.write_text("RuntimeError: HCCL init failed\n")
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(RuntimeError, match="HCCL init failed"):
        runtime.wait_ranks_ready([a], timeout=100, rank_processes=[(_FakeProc(returncode=1), a, log)])


def test_ready_timeout_hints_remote_node_archive(monkeypatch):
    # Without a local process entry the rank is remote: the error must point at
    # that node's log archive instead of a nonexistent local log file.
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(TimeoutError, match="node_0_external_dp_logs"):
        runtime.wait_ranks_ready([_rank()], timeout=0)


def test_sustained_unhealthy_message_includes_local_log_tail(monkeypatch, tmp_path):
    a = _rank(local_rank=0, port=8000)
    b = _rank(local_rank=1, port=8001)
    log = tmp_path / "rank.log"
    log.write_text("npu out of memory\n")
    counts = {"a": 0}

    def fake(url, timeout=None):
        if url.endswith(":8000/health"):
            counts["a"] += 1
            return counts["a"] == 1  # ready once, then permanently down
        return False  # b never ready -> keeps the loop running

    monkeypatch.setattr(runtime, "is_http_ready", fake)
    with pytest.raises(RuntimeError, match="npu out of memory"):
        runtime.wait_ranks_ready([a, b], timeout=100, rank_processes=[(_FakeProc(), a, log)])


# --- proxy readiness (process-aware wait) ---


def test_wait_proxy_ready_returns_when_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: True)
    runtime.wait_proxy_ready(_FakeProc(), "http://x/healthcheck", tmp_path / "proxy.log", timeout=5)


def test_wait_proxy_ready_raises_on_process_exit(monkeypatch, tmp_path):
    log = tmp_path / "proxy.log"
    log.write_text("ModuleNotFoundError: No module named 'fastapi'\n")
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(RuntimeError, match=r"(?s)proxy exited before ready.*fastapi"):
        runtime.wait_proxy_ready(_FakeProc(returncode=2), "http://x/healthcheck", log, timeout=100)


def test_wait_proxy_ready_times_out_with_log_tail(monkeypatch, tmp_path):
    log = tmp_path / "proxy.log"
    log.write_text("still starting\n")
    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(TimeoutError, match="still starting"):
        runtime.wait_proxy_ready(_FakeProc(), "http://x/healthcheck", log, timeout=0)


# --- worker liveness wait (master/worker budget coordination) ---


def test_wait_master_rank_stopped_uses_liveness_debounce(monkeypatch):
    captured = {}
    monkeypatch.setattr(runtime, "wait_http_ready", lambda url, timeout: None)

    def fake_unready(url, timeout, interval, failure_threshold, probe_timeout):
        captured.update(
            timeout=timeout,
            interval=interval,
            failure_threshold=failure_threshold,
            probe_timeout=probe_timeout,
        )

    monkeypatch.setattr(runtime, "wait_http_unready", fake_unready)
    runtime.wait_master_rank_stopped([_rank()])
    # The worker deadline is a backstop: it must fire strictly after the master's
    # own global deadline, never before.
    assert captured["timeout"] > runtime.GLOBAL_TIMEOUT_S
    assert captured["failure_threshold"] == utils.LIVENESS_FAILURE_THRESHOLD
    assert captured["probe_timeout"] == utils.LIVENESS_PROBE_TIMEOUT_S
    assert captured["interval"] == utils.LIVENESS_POLL_INTERVAL_S


def test_wait_master_rank_stopped_timeout_points_at_master_node(monkeypatch):
    monkeypatch.setattr(runtime, "wait_http_ready", lambda url, timeout: None)

    def fake_unready(*args, **kwargs):
        raise TimeoutError("unready timeout")

    monkeypatch.setattr(runtime, "wait_http_unready", fake_unready)
    with pytest.raises(TimeoutError, match="master node"):
        runtime.wait_master_rank_stopped([_rank()])


# --- tail_text ---


def test_tail_text_missing_file():
    assert "log unavailable" in utils.tail_text(Path("/nonexistent/file.log"))


def test_tail_text_keeps_only_last_lines(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line-{i}" for i in range(100)) + "\n")
    assert utils.tail_text(log, max_lines=5).splitlines() == [f"line-{i}" for i in range(95, 100)]


# --- special dependency install ---


def test_install_special_dependencies_raises_on_pip_failure(monkeypatch):
    class _Result:
        returncode = 1

    monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: _Result())
    with pytest.raises(RuntimeError, match=r"transformers==5\.2\.0"):
        utils.install_special_dependencies({"transformers": "5.2.0"})


def test_install_special_dependencies_ok(monkeypatch):
    class _Result:
        returncode = 0

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    utils.install_special_dependencies({"transformers": "5.2.0"})
    assert any("transformers==5.2.0" in part for cmd in calls for part in cmd)


# --- run_with_timeout (master-side global budget watchdog) ---


def test_run_with_timeout_returns_result():
    assert utils.run_with_timeout(lambda: 42, timeout_s=5, task_name="t") == 42


def test_run_with_timeout_propagates_error():
    def boom():
        raise ValueError("inner failure")

    with pytest.raises(ValueError, match="inner failure"):
        utils.run_with_timeout(boom, timeout_s=5, task_name="t")


def test_run_with_timeout_raises_on_deadline():
    import threading

    release = threading.Event()
    try:
        with pytest.raises(TimeoutError, match="did not finish"):
            utils.run_with_timeout(release.wait, timeout_s=0.05, task_name="hang")
    finally:
        release.set()


# --- fixed timeout budgets ---


def test_external_dp_timeout_budgets():
    assert runtime.GLOBAL_TIMEOUT_S == 7200
    assert runtime.SERVICE_STARTUP_TIMEOUT_S == 3600
    assert runtime.WORKER_BACKSTOP_GRACE_S > 0
    # The liveness debounce (worker watching the loaded master) must be materially
    # more tolerant than the startup readiness debounce.
    liveness_window = utils.LIVENESS_FAILURE_THRESHOLD * utils.LIVENESS_POLL_INTERVAL_S
    readiness_window = utils.UNHEALTHY_FAILURE_THRESHOLD * utils.HEALTH_POLL_INTERVAL_S
    assert liveness_window >= 2 * readiness_window
