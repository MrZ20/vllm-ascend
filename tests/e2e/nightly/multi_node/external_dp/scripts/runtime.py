import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ROUTING_DISAGGREGATED_PREFILL,
    ROUTING_GENERIC_DP,
    ExternalDPConfig,
    NodeTemplate,
    RankInfo,
    replace_cluster_placeholders,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    HEALTH_POLL_INTERVAL_S,
    HEALTH_PROBE_TIMEOUT_S,
    LIVENESS_FAILURE_THRESHOLD,
    LIVENESS_POLL_INTERVAL_S,
    LIVENESS_PROBE_TIMEOUT_S,
    UNHEALTHY_FAILURE_THRESHOLD,
    format_server_cmd,
    is_http_ready,
    start_logged_process,
    tail_text,
    terminate_process_tree,
    wait_http_ready,
    wait_http_unready,
)
from tests.e2e.nightly.multi_node.scripts.utils import get_net_interface

logger = logging.getLogger(__name__)

TEMPLATE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
ENV_VAR_RE = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)")

# External DP keeps one global run budget. Service readiness uses a shorter fixed
# budget so startup failures surface quickly without stretching the whole run.
# The master enforces GLOBAL_TIMEOUT_S on the benchmark itself (see
# test_external_dp.py); workers add WORKER_BACKSTOP_GRACE_S on top so their own
# deadline fires strictly after the master's -- a worker must never tear its ranks
# down while the master is still benchmarking.
GLOBAL_TIMEOUT_S = 7200
SERVICE_STARTUP_TIMEOUT_S = 3600
WORKER_BACKSTOP_GRACE_S = 600


@dataclass(frozen=True)
class ServerCommand:
    """Rendered command, env, and printable command line."""

    cmd: list[str]
    env: dict[str, str]
    display_cmd: str


RankProcess = tuple[subprocess.Popen, RankInfo, Path]


class ServerCommandBuilder:
    """Render rank templates into vLLM serve commands."""

    def __init__(self, config: ExternalDPConfig):
        self.config = config

    def build(self, rank: RankInfo, template: NodeTemplate) -> ServerCommand:
        variables = self._build_variables(rank)
        rendered_env = self._render_envs(template.envs, rank, variables)
        rendered_args = [
            self._render_string(
                arg,
                rank=rank,
                braced_variables=variables,
                unbraced_variables=rendered_env,
                allow_missing_unbraced=False,
            )
            for arg in template.server_cmd_template
        ]
        cmd = ["vllm", "serve", self.config.model, *rendered_args]

        env = {key: str(value) for key, value in rendered_env.items()}
        display_cmd = format_server_cmd(cmd, env)
        logger.info(
            "External DP server command node=%s rank=%s: %s",
            rank.node_index,
            rank.local_rank,
            display_cmd,
        )
        return ServerCommand(cmd=cmd, env=env, display_cmd=display_cmd)

    def build_all(self, ranks: list[RankInfo]) -> list[ServerCommand]:
        return [self.build(rank, self.config.launch_templates[rank.node_index]) for rank in ranks]

    def _build_variables(self, rank: RankInfo) -> dict[str, str]:
        return {
            "MODEL": self.config.model,
            "PORT_START": str(rank.port_start),
            "PORT": str(rank.port),
            "DP_SIZE": str(rank.dp_size),
            "DP_SIZE_LOCAL": str(rank.dp_size_local),
            "DP_RANK_START": str(rank.dp_rank - rank.local_rank),
            "DP_RANK": str(rank.dp_rank),
            "LOCAL_RANK": str(rank.local_rank),
            "TP_SIZE": str(rank.tp_size),
            "CP_SIZE": str(rank.cp_size),
            "SP_SIZE": str(rank.sp_size),
            "PP_SIZE": str(rank.pp_size),
            "DP_ADDRESS": rank.dp_address,
            "DP_RPC_PORT": str(rank.dp_rpc_port),
            "VISIBLE_DEVICES": rank.visible_devices,
            "NODE_INDEX": str(rank.node_index),
            "CONFIG_INDEX": str(rank.node_index),
        }

    def _render_envs(
        self,
        envs: dict[str, Any],
        rank: RankInfo,
        variables: dict[str, str],
    ) -> dict[str, str]:
        rendered_envs: dict[str, str] = {}
        for key, value in envs.items():
            if isinstance(value, str):
                value = self._render_string(
                    value,
                    rank=rank,
                    braced_variables=variables,
                    unbraced_variables={**os.environ, **rendered_envs},
                    allow_missing_unbraced=True,
                )
            rendered_envs[str(key)] = str(value)
        return rendered_envs

    def _render_string(
        self,
        value: str,
        *,
        rank: RankInfo,
        braced_variables: dict[str, str],
        unbraced_variables: dict[str, str],
        allow_missing_unbraced: bool,
    ) -> str:
        value = replace_cluster_placeholders(
            value,
            cluster_ips=self.config.cluster_ips,
            local_ip=rank.host,
            current_node_index=rank.node_index,
        )
        value = self._render_variables(
            value,
            braced_variables,
            pattern=TEMPLATE_VAR_RE,
            allow_missing=False,
        )
        return self._render_variables(
            value,
            unbraced_variables,
            pattern=ENV_VAR_RE,
            allow_missing=allow_missing_unbraced,
        )

    @staticmethod
    def _render_variables(
        value: str,
        variables: dict[str, str],
        *,
        pattern: re.Pattern[str],
        allow_missing: bool,
    ) -> str:
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                if allow_missing:
                    return ""
                raise KeyError(f"Unknown external DP template variable: {key}")
            return variables[key]

        return pattern.sub(repl, value)


class ExternalDPServerManager:
    """Start and stop the external DP ranks owned by the current node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        ranks: list[RankInfo],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.ranks = ranks
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.command_builder = ServerCommandBuilder(config)
        self.dist_envs = build_dist_envs(
            config.cluster_ips[current_node_index],
            config.cluster_ips[0],
        )
        self.rank_processes: list[RankProcess] = []

    def start_current_node(self) -> None:
        local_ranks = [rank for rank in self.ranks if rank.node_index == self.current_node_index]
        logger.info("Starting %d external DP ranks on node %d", len(local_ranks), self.current_node_index)
        # Logs are opened in append mode under a fixed root; wipe this node's dir
        # first so the archived tar never mixes in a previous run's output.
        shutil.rmtree(self.log_root / f"node-{self.current_node_index}", ignore_errors=True)
        try:
            for rank in local_ranks:
                template = self.config.launch_templates[rank.node_index]
                template = type(template)(
                    envs={**template.envs, **self.dist_envs},
                    server_cmd_template=template.server_cmd_template,
                )
                server_cmd = self.command_builder.build(rank, template)
                log_file = self._rank_log_file(rank)
                process = start_logged_process(server_cmd.cmd, server_cmd.env, log_file)
                self.rank_processes.append((process, rank, log_file))

            wait_ranks_ready(
                local_ranks,
                timeout=SERVICE_STARTUP_TIMEOUT_S,
                rank_processes=self.rank_processes,
            )
        except Exception:
            self.cleanup()
            raise

    def __enter__(self):
        self.start_current_node()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        for process, rank, _log_file in reversed(self.rank_processes):
            logger.info(
                "Stopping external DP rank node=%d rank=%d pid=%d",
                rank.node_index,
                rank.local_rank,
                process.pid,
            )
            terminate_process_tree(process.pid)
        self.rank_processes.clear()

    def _rank_log_file(self, rank: RankInfo) -> Path:
        return self.log_root / f"node-{rank.node_index}" / f"rank-{rank.local_rank}.log"


class ExternalDPProxyLauncher:
    """Launch the external DP proxy on the configured proxy node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        ranks: list[RankInfo],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.ranks = ranks
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.process: subprocess.Popen | None = None
        self.log_file = log_root / f"node-{current_node_index}" / "proxy.log"

    def start(self) -> None:
        if self.current_node_index != self.config.routing.proxy_node_index:
            logger.info("Current node is not proxy node, skip launching external DP proxy")
            return

        cmd = build_proxy_server_cmd(self.config, self.ranks)
        self.process = start_logged_process(cmd, {}, self.log_file)
        logger.info("External DP proxy launched: %s", proxy_server_health_url(self.config))

    def wait_ready(self) -> None:
        url = proxy_server_health_url(self.config)
        if self.process is None:
            # The proxy runs on another node; readiness is only observable over HTTP.
            wait_http_ready(url, timeout=SERVICE_STARTUP_TIMEOUT_S)
        else:
            wait_proxy_ready(self.process, url, self.log_file, timeout=SERVICE_STARTUP_TIMEOUT_S)
        logger.info("External DP proxy ready: %s", url)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        if self.process is None:
            return
        logger.info("Stopping external DP proxy pid=%d", self.process.pid)
        terminate_process_tree(self.process.pid)
        self.process = None


def build_all_server_commands(config: ExternalDPConfig, ranks: list[RankInfo]) -> list[ServerCommand]:
    return ServerCommandBuilder(config).build_all(ranks)


def build_dist_envs(cur_ip: str, master_ip: str) -> dict[str, str]:
    nic_name = get_net_interface(cur_ip)
    return {
        "HCCL_IF_IP": cur_ip,
        "HCCL_SOCKET_IFNAME": nic_name,
        "GLOO_SOCKET_IFNAME": nic_name,
        "TP_SOCKET_IFNAME": nic_name,
        "LOCAL_IP": cur_ip,
        "NIC_NAME": nic_name,
        "MASTER_IP": master_ip,
    }


def build_proxy_server_cmd(config: ExternalDPConfig, ranks: list[RankInfo]) -> list[str]:
    routing = config.routing
    cmd = [sys.executable, routing.proxy_script, "--host", routing.proxy_host, "--port", str(routing.proxy_port)]

    if routing.type == ROUTING_GENERIC_DP:
        worker_ranks = [rank for rank in ranks if rank.role == "worker"]
        if not worker_ranks:
            raise ValueError("generic_dp proxy requires worker ranks")
        cmd.extend(["--dp-hosts", *[rank.host for rank in worker_ranks]])
        cmd.extend(["--dp-ports", *[str(rank.port) for rank in worker_ranks]])
        return cmd

    if routing.type == ROUTING_DISAGGREGATED_PREFILL:
        prefiller_ranks = [rank for rank in ranks if rank.role == "prefiller"]
        decoder_ranks = [rank for rank in ranks if rank.role == "decoder"]
        if not prefiller_ranks or not decoder_ranks:
            raise ValueError("disaggregated_prefill proxy requires prefiller and decoder ranks")
        cmd.extend(["--prefiller-hosts", *[rank.host for rank in prefiller_ranks]])
        cmd.extend(["--prefiller-ports", *[str(rank.port) for rank in prefiller_ranks]])
        cmd.extend(["--decoder-hosts", *[rank.host for rank in decoder_ranks]])
        cmd.extend(["--decoder-ports", *[str(rank.port) for rank in decoder_ranks]])
        return cmd

    raise ValueError(f"Unsupported routing.type: {routing.type}")


def proxy_server_health_url(config: ExternalDPConfig) -> str:
    return f"http://{config.routing.proxy_host}:{config.routing.proxy_port}/healthcheck"


def wait_proxy_ready(process: subprocess.Popen, url: str, log_file: Path, timeout: int) -> None:
    """Wait for the local proxy's healthcheck, failing fast if its process dies.

    Without the process check, a proxy that crashes at launch (missing dependency,
    bad argument) would silently burn the whole startup budget before surfacing a
    generic HTTP timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"External DP proxy exited before ready: pid={process.pid} "
                f"returncode={returncode} log={log_file}\n"
                f"--- tail of {log_file} ---\n{tail_text(log_file)}"
            )
        if is_http_ready(url, timeout=HEALTH_PROBE_TIMEOUT_S):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for external DP proxy ready: {url}\n"
                f"--- tail of {log_file} ---\n{tail_text(log_file)}"
            )
        time.sleep(HEALTH_POLL_INTERVAL_S)


def rank_health_url(rank: RankInfo) -> str:
    return f"http://{rank.host}:{rank.port}/health"


def master_rank_health_url(ranks: list[RankInfo]) -> str:
    for rank in ranks:
        if rank.node_index == 0 and rank.local_rank == 0:
            return rank_health_url(rank)
    raise RuntimeError("External DP master rank was not found")


def rank_label(rank: RankInfo) -> str:
    return f"node={rank.node_index} rank={rank.local_rank} role={rank.role} url={rank_health_url(rank)}"


def format_http_status(label: str, url: str) -> str:
    # Heartbeat-only status. A loaded server can legitimately miss the probe
    # window, so report "no-response" rather than "waiting" -- this line must not
    # read as "the service is down" to someone scanning the logs.
    ready = is_http_ready(url, timeout=HEALTH_PROBE_TIMEOUT_S)
    status = "ready" if ready else f"no-response(>{HEALTH_PROBE_TIMEOUT_S:g}s)"
    return f"{label}={status} url={url}"


def _format_rank_statuses(
    ranks: list[RankInfo],
    rank_ready: dict[RankInfo, bool],
) -> str:
    parts = []
    for rank in ranks:
        status = "ready" if rank_ready[rank] else "waiting"
        parts.append(f"  {rank_label(rank)} status={status}")
    return "\n".join(parts)


def _rank_log_path(rank: RankInfo, rank_processes: list[RankProcess] | None) -> Path | None:
    for _process, proc_rank, log_file in rank_processes or []:
        if proc_rank == rank:
            return log_file
    return None


def _describe_rank_failure(rank: RankInfo, rank_processes: list[RankProcess] | None) -> str:
    """Label a failed rank with everything needed to pinpoint the root cause.

    Local ranks get their log tail inlined so the master's pytest output is
    self-contained; remote ranks get an explicit pointer to that node's archive,
    because a rank stuck here is usually the victim of a failure on its DP peer.
    """
    log_file = _rank_log_path(rank, rank_processes)
    if log_file is None:
        return (
            f"{rank_label(rank)} (runs on node {rank.node_index}; root cause may be on that node -- "
            f"see node_{rank.node_index}_external_dp_logs.tar.gz in the job artifacts)"
        )
    return f"{rank_label(rank)} log={log_file}\n--- tail of {log_file} ---\n{tail_text(log_file)}"


def _raise_if_rank_process_exited(rank_processes: list[RankProcess] | None) -> None:
    if not rank_processes:
        return

    exited = []
    for process, rank, log_file in rank_processes:
        returncode = process.poll()
        if returncode is not None:
            exited.append(
                f"{rank_label(rank)} pid={process.pid} returncode={returncode} log={log_file}\n"
                f"--- tail of {log_file} ---\n{tail_text(log_file)}"
            )

    if exited:
        raise RuntimeError("External DP rank process exited before ready:\n" + "\n".join(exited))


def _probe_rank_ready(rank: RankInfo) -> bool:
    return is_http_ready(rank_health_url(rank), timeout=HEALTH_PROBE_TIMEOUT_S)


def wait_ranks_ready(
    ranks: Iterable[RankInfo],
    timeout: int,
    rank_processes: list[RankProcess] | None = None,
) -> None:
    ranks = list(ranks)
    if not ranks:
        return
    rank_ready = {rank: False for rank in ranks}
    consecutive_unhealthy = {rank: 0 for rank in ranks}
    deadline = time.monotonic() + timeout
    last_log_time = 0.0

    # Probe concurrently: serial probes of N ranks take up to N * probe-timeout per
    # iteration, which both delays failure detection and distorts the
    # consecutive-failure debounce below.
    with ThreadPoolExecutor(max_workers=min(len(ranks), 16)) as probe_pool:
        while True:
            _raise_if_rank_process_exited(rank_processes)

            probe_results = dict(zip(ranks, probe_pool.map(_probe_rank_ready, ranks)))
            for rank, is_ready in probe_results.items():
                if is_ready:
                    if not rank_ready[rank]:
                        logger.info("[READY] External DP rank %s", rank_label(rank))
                    rank_ready[rank] = True
                    consecutive_unhealthy[rank] = 0
                elif rank_ready[rank]:
                    consecutive_unhealthy[rank] += 1

            # Only treat an already-ready rank as crashed once it has failed several
            # probes in a row; a single transient failure (slow /health under load, a
            # network blip) must not abort an otherwise-healthy run.
            flapping = [
                rank
                for rank in ranks
                if rank_ready[rank] and consecutive_unhealthy[rank] >= UNHEALTHY_FAILURE_THRESHOLD
            ]
            if flapping:
                failed = "\n".join(_describe_rank_failure(rank, rank_processes) for rank in flapping)
                raise RuntimeError(f"External DP rank became unhealthy after ready:\n{failed}")

            if all(probe_results.values()):
                return

            now = time.monotonic()
            if now - last_log_time >= 30:
                logger.info(
                    "Polling external DP ranks: ready=%d/%d\n%s",
                    sum(rank_ready.values()),
                    len(ranks),
                    _format_rank_statuses(ranks, rank_ready),
                )
                last_log_time = now

            if now >= deadline:
                pending = [rank for rank in ranks if not rank_ready[rank]]
                details = "\n".join(_describe_rank_failure(rank, rank_processes) for rank in pending)
                raise TimeoutError(f"Timed out waiting for external DP ranks ready:\n{details}")

            time.sleep(HEALTH_POLL_INTERVAL_S)


def wait_master_rank_stopped(ranks: list[RankInfo]) -> None:
    """Block a worker until the master rank has come up and then stopped.

    Phase 1 waits for the master to become ready, bounded by the service startup
    budget. Phase 2 hangs until the master's health drops, using the tolerant
    liveness debounce (the master is under benchmark load, and a false "stopped"
    verdict would tear this worker's ranks down mid-run). Its deadline is the
    global budget plus a grace margin: the master enforces the global budget on
    the benchmark itself, so this timeout is a backstop that must only fire if
    the master failed to act on its own deadline.
    """
    url = master_rank_health_url(ranks)
    start = time.monotonic()
    wait_http_ready(url, timeout=SERVICE_STARTUP_TIMEOUT_S)
    logger.info("Hanging until master external DP rank stops: %s", url)
    remaining = max(int(GLOBAL_TIMEOUT_S - (time.monotonic() - start)), 0) + WORKER_BACKSTOP_GRACE_S
    try:
        wait_http_unready(
            url,
            timeout=remaining,
            interval=LIVENESS_POLL_INTERVAL_S,
            failure_threshold=LIVENESS_FAILURE_THRESHOLD,
            probe_timeout=LIVENESS_PROBE_TIMEOUT_S,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            "Master external DP rank never stopped within the global budget; the root "
            f"cause is on the master node (node 0), not this worker -- check the master "
            f"node's pytest output and node_0_external_dp_logs.tar.gz. url={url}"
        ) from exc
