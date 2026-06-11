import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ExternalDPConfig,
    ExternalDPConfigLoader,
    RankResolver,
    resolve_current_node_index,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import (
    GLOBAL_TIMEOUT_S,
    SERVICE_STARTUP_TIMEOUT_S,
    ExternalDPProxyLauncher,
    ExternalDPServerManager,
    build_all_server_commands,
    format_http_status,
    master_rank_health_url,
    proxy_server_health_url,
    wait_master_rank_stopped,
    wait_ranks_ready,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    collect_logs,
    install_special_dependencies,
    run_with_timeout,
    write_benchmark_results_json,
)
from tools.aisbench import run_aisbench_cases

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LOG_ROOT = Path("/tmp/external_dp_logs")


@contextmanager
def _heartbeat(
    task_name: str,
    *,
    interval: int = 30,
    status_fn: Callable[[], str] | None = None,
):
    start_time = time.monotonic()
    stop_event = threading.Event()

    def report_progress() -> None:
        while not stop_event.wait(interval):
            elapsed = int(time.monotonic() - start_time)
            status = ""
            if status_fn is not None:
                try:
                    status = f" {status_fn()}"
                except Exception as exc:  # pragma: no cover - diagnostic only
                    status = f" status_error={exc!r}"
            logger.info("%s still running: elapsed=%ds%s", task_name, elapsed, status)

    logger.info("%s started", task_name)
    thread = threading.Thread(target=report_progress, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)
        elapsed = int(time.monotonic() - start_time)
        logger.info("%s finished: elapsed=%ds", task_name, elapsed)


def _format_benchmark_cases(config: ExternalDPConfig) -> str:
    names = [str(case.get("case_name", "<unnamed>")) for case in config.benchmark_cases]
    return ", ".join(names) if names else "<none>"


def _archive_rank_logs(log_root: Path, current_node_index: int) -> None:
    log_prefix = os.environ.get("LOG_PREFIX")
    if not log_prefix:
        return
    node_log_dir = log_root / f"node-{current_node_index}"
    output_tar = Path(log_prefix) / f"node_{current_node_index}_external_dp_logs.tar.gz"
    collect_logs(node_log_dir, output_tar)


def test_external_dp() -> None:
    run_start = time.monotonic()
    config = ExternalDPConfigLoader.from_yaml()
    install_special_dependencies(config.special_dependencies)
    ranks = RankResolver(config).resolve()
    current_node_index = resolve_current_node_index(config)
    log_root = Path(os.environ.get("EXTERNAL_DP_LOG_DIR", str(DEFAULT_LOG_ROOT)))
    logger.info(
        "External DP timeouts: global=%ds service_startup=%ds",
        GLOBAL_TIMEOUT_S,
        SERVICE_STARTUP_TIMEOUT_S,
    )
    is_master = current_node_index == 0

    server_manager = ExternalDPServerManager(
        config=config,
        ranks=ranks,
        current_node_index=current_node_index,
        log_root=log_root,
    )
    proxy_launcher = ExternalDPProxyLauncher(
        config=config,
        ranks=ranks,
        current_node_index=current_node_index,
        log_root=log_root,
    )

    try:
        with server_manager, proxy_launcher:
            if is_master:
                wait_ranks_ready(
                    ranks,
                    timeout=SERVICE_STARTUP_TIMEOUT_S,
                    rank_processes=server_manager.rank_processes,
                )
                proxy_launcher.wait_ready()
                target = f"http://{config.routing.proxy_host}:{config.routing.proxy_port}"
                logger.info(
                    "Running AISBench cases: model=%s target=%s cases=[%s]",
                    config.model,
                    target,
                    _format_benchmark_cases(config),
                )
                # Enforce the global budget here: without a master-side deadline a
                # stuck benchmark hangs forever while the workers' backstop fires
                # first and tears the DP group down mid-run.
                remaining_budget = GLOBAL_TIMEOUT_S - (time.monotonic() - run_start)
                with _heartbeat(
                    "Running AISBench",
                    status_fn=lambda: format_http_status("proxy", proxy_server_health_url(config)),
                ):
                    results = run_with_timeout(
                        lambda: run_aisbench_cases(
                            model=config.model,
                            port=config.routing.proxy_port,
                            aisbench_cases=config.benchmark_cases,
                            host_ip=config.routing.proxy_host,
                        ),
                        timeout_s=remaining_budget,
                        task_name=(
                            "AISBench (the serving stack was healthy when the benchmark started; "
                            "check proxy.log and the rank logs in the node archives for stuck requests)"
                        ),
                    )
                logger.info("AISBench completed: results=%d", len(results or []))
                all_commands = build_all_server_commands(config, ranks)
                write_benchmark_results_json(
                    config=config,
                    ranks=ranks,
                    commands=all_commands,
                    results=results,
                )
            else:
                master_url = master_rank_health_url(ranks)
                with _heartbeat(
                    "Waiting for master external DP rank to stop",
                    status_fn=lambda: format_http_status("master", master_url),
                ):
                    wait_master_rank_stopped(ranks)
    finally:
        _archive_rank_logs(log_root, current_node_index)
