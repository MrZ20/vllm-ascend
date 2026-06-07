"""Unit tests for the external_dp start/stop synchronization logic.

These cover the readiness/liveness debouncing and the two-timeout resolution that
caused the nightly run to time out half-way and tear ranks down on a transient
network blip. They mock the HTTP probes, so they need no NPU/cluster.
"""

import itertools
from pathlib import Path

import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts import runtime, utils
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ROUTING_GENERIC_DP,
    ExternalDPConfig,
    NodeInfo,
    NodeTemplate,
    RankInfo,
    RoutingConfig,
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


def _config(engine_timeout: str | None = "7200") -> ExternalDPConfig:
    envs = {} if engine_timeout is None else {"VLLM_ENGINE_READY_TIMEOUT_S": engine_timeout}
    template = NodeTemplate(envs=envs, server_cmd_template=["--host", "0.0.0.0"])
    routing = RoutingConfig(
        type=ROUTING_GENERIC_DP,
        proxy_node_index=0,
        proxy_host="127.0.0.1",
        proxy_port=1999,
        proxy_script="proxy.py",
        groups={"worker": [0]},
    )
    node = NodeInfo(
        ip="127.0.0.1",
        port_start=7100,
        dp_rpc_port=12321,
        dp_size=1,
        dp_size_local=1,
        dp_rank_start=0,
        tp_size=1,
        dp_address="127.0.0.1",
    )
    return ExternalDPConfig(
        test_name="t",
        model="m",
        num_nodes=1,
        npu_per_node=8,
        cluster_hosts=None,
        cluster_ips=["127.0.0.1"],
        routing=routing,
        nodes=[node],
        launch_templates=[template],
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


def test_wait_ranks_ready_detects_local_process_exit(monkeypatch):
    a = _rank()

    class _Proc:
        pid = 4321

        def poll(self):
            return 1

    monkeypatch.setattr(runtime, "is_http_ready", lambda url, timeout=None: False)
    with pytest.raises(RuntimeError, match="exited before ready"):
        runtime.wait_ranks_ready([a], timeout=100, rank_processes=[(_Proc(), a, Path("rank.log"))])


# --- two-timeout resolution ---


def test_engine_ready_timeout_from_templates():
    assert _config("7200").engine_ready_timeout_s == 7200


def test_engine_ready_timeout_env_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "999")
    assert _config(engine_timeout=None).engine_ready_timeout_s == 999


def test_resolve_startup_timeout_derives(monkeypatch):
    monkeypatch.delenv("EXTERNAL_DP_STARTUP_TIMEOUT_S", raising=False)
    assert runtime.resolve_startup_timeout(_config("7200")) == 7200 + runtime.STARTUP_MARGIN_S


def test_resolve_startup_timeout_env_override(monkeypatch):
    monkeypatch.setenv("EXTERNAL_DP_STARTUP_TIMEOUT_S", "1234")
    assert runtime.resolve_startup_timeout(_config("7200")) == 1234


def test_resolve_total_timeout_derives(monkeypatch):
    monkeypatch.delenv("EXTERNAL_DP_TOTAL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("EXTERNAL_DP_STARTUP_TIMEOUT_S", raising=False)
    expected = 7200 + runtime.STARTUP_MARGIN_S + runtime.BENCHMARK_ALLOWANCE_S
    assert runtime.resolve_total_timeout(_config("7200")) == expected


def test_resolve_total_timeout_env_override(monkeypatch):
    monkeypatch.setenv("EXTERNAL_DP_TOTAL_TIMEOUT_S", "555")
    assert runtime.resolve_total_timeout(_config("7200")) == 555
