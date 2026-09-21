# SPDX-License-Identifier: Apache-2.0
"""Environment resolution, Remote* launch adapter and scheduled test runner."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import operator
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tests.e2e.schedule.config import (
    NPU_QUERY_TIMEOUT_SECONDS,
    START_TIMEOUT,
    TEST_TIMEOUT,
    CaseSpec,
    ClusterContext,
    Endpoint,
    LaunchPlan,
    ResourceSpec,
    TestSpec,
    parse_cases,
    positive_int,
)
from tests.e2e.schedule.deployment import build_plan

logger = logging.getLogger(__name__)


def resolve_cluster(env: dict[str, str], device_pool: tuple[int, ...]) -> tuple[ResourceSpec, ClusterContext]:
    """Resolve an allocation without initializing the pytest process's NPU state."""
    hosts = tuple(host.strip() for host in env.get("CLUSTER_HOSTS", "").split(",") if host.strip())
    sizes = [positive_int(env[name], name) for name in ("NUM_NODES", "LWS_GROUP_SIZE") if env.get(name)]
    if hosts:
        sizes.append(len(hosts))
    if len(set(sizes)) > 1:
        raise ValueError("NUM_NODES, LWS_GROUP_SIZE and CLUSTER_HOSTS disagree")
    count = sizes[0] if sizes else 1
    if count > 1 and "LWS_WORKER_INDEX" not in env:
        raise ValueError("Multinode local runs require LWS_WORKER_INDEX")
    index = int(env.get("LWS_WORKER_INDEX", "0"))
    if not 0 <= index < count:
        raise ValueError("LWS_WORKER_INDEX is outside the allocation")
    if not hosts:
        if count == 1:
            hosts = ("127.0.0.1",)
        else:
            leader = env.get("LWS_LEADER_ADDRESS")
            if not leader:
                raise ValueError("Multinode runs require CLUSTER_HOSTS or LWS_LEADER_ADDRESS")
            name, separator, domain = leader.partition(".")
            hosts = tuple(
                socket.gethostbyname(leader if i == 0 else f"{name}-{i}{separator}{domain}") for i in range(count)
            )
    npu_count = positive_int(env.get("NPU_PER_NODE") or str(len(device_pool)), "NPU_PER_NODE")
    if npu_count != len(device_pool):
        raise ValueError(
            "NPU_PER_NODE differs from the visible allocation; use ASCEND_RT_VISIBLE_DEVICES to select cards"
        )
    import psutil

    local_ip = socket.gethostbyname(hosts[index])
    interfaces = [
        name
        for name, addresses in psutil.net_if_addrs().items()
        if any(a.family == socket.AF_INET and a.address == local_ip for a in addresses)
    ]
    if len(interfaces) > 1 or (count > 1 and not interfaces):
        raise ValueError(f"Node address {local_ip} does not identify one local NIC: {interfaces}")
    return ResourceSpec(count, npu_count), ClusterContext(
        index, hosts, device_pool, interfaces[0] if interfaces else None
    )


def redact(value):
    """Redact declared secrets; never serialize the whole inherited environment."""
    from tests.e2e.utils import _redact_sensitive_cli_args

    if isinstance(value, dict):
        return {
            k: "***"
            if re.search(r"(?:password|secret|api[_-]?key|(?:^|_)token(?:$|_))", k, re.I)
            else _redact_sensitive_cli_args(v)
            if k in ("argv", "command") and isinstance(v, (list, tuple))
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-") or "case"


@dataclass
class RunningDeployment:
    processes: dict
    check_peers: Callable[[], None] | None = None

    def assert_healthy(self) -> None:
        for name, process in self.processes.items():
            result = process.poll()
            if result is not None:
                raise RuntimeError(f"{name} exited during deployment: exit_code={result}")
        if self.check_peers is not None:
            self.check_peers()

    def wait_for_stop(self, url: str, timeout: float = TEST_TIMEOUT):
        import requests

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            results = [p.poll() for p in self.processes.values()]
            if any(result not in (None, 0) for result in results):
                raise RuntimeError(f"Local worker failed while waiting for stop: {results}")
            if results and all(result == 0 for result in results):
                return
            try:
                with requests.get(url, timeout=5) as response:
                    if response.status_code != 200:
                        return
            except requests.RequestException:
                return
            time.sleep(0.5)
        raise TimeoutError("Primary endpoint did not stop within the local wait budget")


class Launcher:
    """Adapt a local plan to the shared runtime, with one owner per process."""

    def __init__(
        self,
        log_dir: Path,
        device_pool: tuple[int, ...],
        timeout: float = START_TIMEOUT,
        auxiliary_processes: dict | None = None,
    ):
        self.log_dir, self.device_pool, self.timeout = log_dir, device_pool, timeout
        self.auxiliary_processes = auxiliary_processes or {}

    @contextmanager
    def run(self, plan: LaunchPlan):
        from tests.e2e.utils import RemoteProxy, RemoteServerGroup, _prepare_server_group, wait_for_http_targets

        owners, handles = [], {}
        deadline = time.monotonic() + self.timeout
        original_error = None

        def remaining():
            return max(0, deadline - time.monotonic())

        def save_effective():
            write_json(
                self.log_dir / "launch_effective.json",
                redact(
                    {
                        "processes": [
                            {
                                "id": name,
                                "argv": p.command,
                                "env_overrides": p.env_overrides,
                                "cwd": p.cwd,
                                "pid": p.proc.pid if p.proc else None,
                                "pgid": p.pgid,
                                "log_file": str(p.log_file),
                            }
                            for name, p in handles.items()
                        ]
                    }
                ),
            )

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.log_dir / "launch_plan.json", redact(asdict(plan)))
            servers = [p for p in plan.processes if p.kind == "server"]
            if servers:
                # One case-wide preparation reserves later explicit allocations
                # before any automatic assignment or Popen happens.
                prepared, _, _, _ = _prepare_server_group(
                    [list(p.argv[2:]) for p in servers],
                    "0.0.0.0",
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, self.device_pool))},
                    "SCHEDULE",
                    per_server_envs=[p.env_overrides for p in servers],
                    working_dirs=[p.cwd for p in servers],
                    log_files=[self.log_dir / "processes" / f"{safe_name(p.id)}.log" for p in servers],
                    health_urls=[],
                )
                handles.update(zip((p.id for p in servers), prepared))
                owner = RemoteServerGroup(prepared, [], timeout=remaining(), wait_for_ready=False)
                owners.append(owner)
                save_effective()
            # Peers are all started. Headless peers use their Service's serving
            # endpoints as anchors, without a before-start dependency on them.
            targets = []
            for process in servers:
                endpoints = (process.endpoint,) if process.endpoint else plan.service_endpoints[process.owner.name]
                if not endpoints:
                    raise ValueError(f"Service {process.owner.name} has no readiness anchor")
                targets.extend(e.url + "/health" for e in endpoints)
            for process in plan.processes:
                if process.kind != "router":
                    continue
                dependencies = [e for dep in plan.dependencies if dep.process_id == process.id for e in dep.endpoints]
                wait_for_http_targets(
                    [e.url + "/health" for e in dependencies],
                    remaining(),
                    poll_processes=[*handles.values(), *self.auxiliary_processes.values()],
                    always_check=True,
                )
                health_path = "healthcheck" if process.owner.name == "disaggregated_prefill" else "health"
                owner = RemoteProxy(
                    process.argv,
                    host=process.endpoint.host,
                    port=process.endpoint.port,
                    health_path=health_path,
                    env_dict=process.env_overrides,
                    timeout=remaining(),
                    cwd=process.cwd,
                    log_file=self.log_dir / "processes" / f"{safe_name(process.id)}.log",
                    wait_for_ready=False,
                )
                owners.append(owner)
                handles[process.id] = owner._process
                targets.append(process.endpoint.url + "/" + health_path)
                save_effective()
            if targets:
                wait_for_http_targets(
                    targets,
                    remaining(),
                    poll_processes=[*handles.values(), *self.auxiliary_processes.values()],
                    always_check=True,
                )
            save_effective()
            yield RunningDeployment({**self.auxiliary_processes, **handles})
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            errors = []
            for owner in reversed(owners):
                try:
                    owner._shutdown()
                except Exception as exc:
                    errors.append(str(exc))
            try:
                write_json(
                    self.log_dir / "cleanup.json",
                    {
                        "status": "failed" if errors else "passed",
                        "errors": errors,
                        "exit_codes": {name: process.poll() for name, process in handles.items()},
                    },
                )
            except Exception as exc:
                errors.append(f"Writing cleanup result failed: {exc}")
            if errors:
                message = f"Deployment cleanup failed: {errors}"
                if original_error is not None:
                    original_error.add_note(message)
                else:
                    raise RuntimeError(message)


class Coordinator:
    """Run/case-isolated result exchange over an explicitly shared directory."""

    def __init__(
        self, root: Path, run_id: str, case_id: str, index: int, count: int, digest: str, timeout: float = TEST_TIMEOUT
    ):
        self.root = root / run_id / "cases" / case_id
        self.identity = {"run_id": run_id, "case_id": case_id, "node_index": index, "config_digest": digest}
        self.index, self.count, self.timeout = index, count, timeout

    def write(self, name: str, **values):
        write_json(self.root / (name + ".json"), {**self.identity, **values})

    def read(self, name: str):
        path = self.root / (name + ".json")
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if any(value.get(k) != self.identity[k] for k in ("run_id", "case_id", "config_digest")):
            raise RuntimeError(f"Inconsistent run/case/config identity: {path}")
        return value

    def check_failures(self):
        for index in range(self.count):
            value = self.read(f"node-{index}.result")
            if value is not None and value["status"] == "failed":
                raise RuntimeError(f"Node {index} failed: {value.get('error')}")

    def wait(self, names: list[str], healthy: Callable[[], None] = lambda: None, *, detect_failure: bool = True):
        deadline = time.monotonic() + self.timeout
        while True:
            if detect_failure:
                self.check_failures()
            values = [self.read(name) for name in names]
            if all(value is not None for value in values):
                return values
            # A published stop permits peers to exit. Read it before probing
            # local processes, which may already be following the primary out.
            healthy()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Waiting for coordination files: {[n for n, v in zip(names, values) if v is None]}")
            time.sleep(0.2)


def preflight(repo_root: Path, output: Path) -> tuple[int, ...]:
    command = [sys.executable, str(repo_root / "tests/e2e/schedule/scripts/check_npu.py"), "--output", str(output)]
    subprocess.run(command, check=True, timeout=NPU_QUERY_TIMEOUT_SECONDS)
    return tuple(json.loads(output.read_text())["device_pool"])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _test_payload(case: CaseSpec, plan: LaunchPlan) -> dict:
    routing = case.deployment.routing
    tokenize = metrics = plan.client_endpoint
    if routing:
        groups = routing.groups
        p_group, d_group = ("prefiller", "decoder") if routing.type == "disaggregated_prefill" else ("decode", "decode")
        tokenize = plan.service_endpoints[groups[p_group][0]][0]
        metrics = plan.service_endpoints[groups[d_group][0]][0]
    server = next((p for p in plan.processes if p.kind == "server"), None)
    return {
        "test": asdict(case.test),
        "completion": asdict(plan.client_endpoint),
        "tokenize": asdict(tokenize),
        "metrics": asdict(metrics),
        "server_argv": list(server.argv) if server else [],
    }


def run_test_process(
    case: CaseSpec,
    plan: LaunchPlan,
    running: RunningDeployment,
    log_dir: Path,
    repo_root: Path,
    timeout: float = TEST_TIMEOUT,
):
    from tests.e2e.utils import _ManagedProcess

    payload_path, result_path = log_dir / "test_input.json", log_dir / "benchmark_results.json"
    # Request content is needed by the child; keep this non-artifact file private.
    write_json(payload_path, _test_payload(case, plan))
    payload_path.chmod(0o600)
    env = {"PYTHONPATH": os.pathsep.join(filter(None, (str(repo_root), os.environ.get("PYTHONPATH"))))}
    if case.test.benchmarks:
        # tools.aisbench writes *_custom.py. Keep those writes local to this run.
        benchmark_home = Path(os.environ.get("BENCHMARK_HOME", str(repo_root / "benchmark")))
        local_home = log_dir / "benchmark"
        shutil.copytree(
            benchmark_home / "ais_bench" / "benchmark" / "configs", local_home / "ais_bench" / "benchmark" / "configs"
        )
        (local_home / "ais_bench" / "datasets").mkdir(parents=True)
        env["BENCHMARK_HOME"] = str(local_home)
        env["AIS_BENCH_DATASETS_CACHE"] = str(local_home)
        # AISBench's installed package can read explicit config paths; its helper
        # names below are resolved against this per-run config copy by the child.
    process = _ManagedProcess(
        [
            sys.executable,
            "-m",
            "tests.e2e.schedule.runtime",
            "--test-input",
            str(payload_path),
            "--test-output",
            str(result_path),
        ],
        env,
        cwd=str(log_dir),
        log_file=log_dir / "tests.log",
    )
    deadline = time.monotonic() + timeout
    try:
        process.start()
        while process.poll() is None:
            running.assert_healthy()
            if time.monotonic() >= deadline:
                raise TimeoutError("Model tests/benchmark exceeded TEST_TIMEOUT")
            time.sleep(0.2)
        if process.poll() != 0:
            raise RuntimeError(f"Model test process exited with {process.poll()}; see {log_dir / 'tests.log'}")
        running.assert_healthy()
        if not result_path.is_file():
            raise RuntimeError("Model test process produced no result artifact")
        return json.loads(result_path.read_text())
    finally:
        process.shutdown()
        payload_path.unlink(missing_ok=True)


def _exception_text(exc: BaseException) -> str:
    return "\n".join([f"{type(exc).__name__}: {exc}", *getattr(exc, "__notes__", [])])


def run_case(
    case: CaseSpec,
    cluster: ClusterContext,
    repo_root: Path,
    run_dir: Path,
    run_id: str,
    case_id: str,
    digest: str,
    coord_root: Path | None,
):
    """Keep tests, stop, local cleanup and final cluster success separate."""
    index, count = cluster.current_node_index, case.deployment.resources.num_nodes
    log_dir = run_dir / f"node-{index}" / "cases" / case_id
    coord = Coordinator(coord_root, run_id, case_id, index, count, digest) if coord_root else None
    error = None
    tests_ok = False
    phase = "preflight"
    try:
        if preflight(repo_root, log_dir / "npu_preflight.json") != cluster.current_device_pool:
            raise RuntimeError("Visible NPU allocation changed between cases")
        for package, version in case.special_dependencies.items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f"Prepare special dependency {package}=={version} before running")
        with ExitStack() as stack:
            pool_env, auxiliary = {}, {}
            if case.deployment.infrastructure is not None:
                from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import create_kv_pool_manager

                phase = "infrastructure"
                pool = stack.enter_context(
                    create_kv_pool_manager(
                        kv_pool=case.deployment.infrastructure,
                        cluster_ips=list(cluster.node_ips),
                        current_node_index=index,
                        log_root=run_dir / "infrastructure" / case_id,
                    )
                )
                pool_env = pool.server_envs
                if pool.process is not None:
                    auxiliary["infrastructure/kv_pool"] = pool.process
            phase = "planning"
            plan = build_plan(case.deployment, cluster, {**os.environ, **pool_env}, port_provider=_free_port)
            if pool_env:
                plan = replace(
                    plan,
                    processes=tuple(
                        replace(p, env_overrides={**pool_env, **p.env_overrides}) if p.kind == "server" else p
                        for p in plan.processes
                    ),
                )
            phase = "startup"
            launcher = Launcher(log_dir, cluster.current_device_pool, auxiliary_processes=auxiliary)
            running = stack.enter_context(launcher.run(plan))
            if coord:
                coord.write(f"node-{index}.ready", status="ready")
                running.check_peers = coord.check_failures
            try:
                phase = "tests" if index == 0 else "wait_for_primary"
                if index == 0:
                    if coord:
                        coord.wait([f"node-{i}.ready" for i in range(count)], running.assert_healthy)
                    result = run_test_process(case, plan, running, log_dir, repo_root)
                    write_json(
                        run_dir / "benchmark_results" / f"{case_id}.json",
                        {"run_id": run_id, "case_id": case_id, "config_digest": digest, **result},
                    )
                    tests_ok = True
                elif coord:
                    stop = coord.wait(["stop"], running.assert_healthy)[0]
                    if stop["status"] != "passed":
                        raise RuntimeError(f"Primary tests failed: {stop.get('error')}")
                else:
                    from tests.e2e.utils import wait_for_http_targets

                    path = (
                        "healthcheck"
                        if case.deployment.routing and case.deployment.routing.type == "disaggregated_prefill"
                        else "health"
                    )
                    url = plan.client_endpoint.url + "/" + path
                    wait_for_http_targets([url], START_TIMEOUT, poll_processes=list(running.processes.values()))
                    # Only local-stop semantics are available without shared state.
                    running.wait_for_stop(url)
            except BaseException as exc:
                error = exc
                raise
            finally:
                if index == 0 and coord:
                    coord.write(
                        "stop",
                        status="passed" if tests_ok else "failed",
                        error=_exception_text(error) if error else None,
                    )
                if error is None:
                    phase = "shutdown"
    except BaseException as exc:
        error = exc
    local_result = {
        "status": "failed" if error else "passed" if index == 0 or coord else "local_stopped",
        "error": _exception_text(error) if error else None,
        "tests_passed": tests_ok,
        "node_index": index,
        "run_id": run_id,
        "case_id": case_id,
        "config_digest": digest,
        "phase": phase if error else "complete",
    }
    write_json(log_dir / "result.json", local_result)
    if coord:
        coord.write(f"node-{index}.result", **{k: v for k, v in local_result.items() if k not in coord.identity})
        if index == 0:
            # A failed/interrupted node has already cleaned up its own processes.
            # Do not delay failure waiting for peers that may never have started.
            if error is None:
                try:
                    results = coord.wait([f"node-{i}.result" for i in range(count)], detect_failure=False)
                    if any(r["status"] != "passed" for r in results):
                        error = RuntimeError(f"Node cleanup/run failed: {results}")
                except BaseException as exc:
                    error = exc
            coord.write(
                "final",
                status="passed" if error is None and tests_ok else "failed",
                error=_exception_text(error) if error else None,
            )
        elif error is None:
            final = coord.wait(["final"], detect_failure=False)[0]
            if final["status"] != "passed":
                error = RuntimeError(f"Cluster case failed: {final.get('error')}")
    if error is not None:
        raise error


def run_from_environment() -> None:
    env = dict(os.environ)
    repo = Path(env["REPO_ROOT"]).resolve()
    index = int(env.get("LWS_WORKER_INDEX", "0"))
    run_id = env.get("RUN_ID") or uuid.uuid4().hex
    if safe_name(run_id) != run_id:
        raise ValueError("RUN_ID must be a simple path component")
    run_dir = Path(env.get("LOG_PREFIX") or tempfile.mkdtemp(prefix="schedule-v2-")) / run_id
    coord_root = Path(env["COORD_DIR"]) if env.get("COORD_DIR") else None
    if coord_root and not env.get("RUN_ID"):
        raise ValueError("Shared coordination requires the same explicit RUN_ID on every node")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    source_commit = git.stdout.strip() if git.returncode == 0 else None
    if env.get("EXPECTED_SOURCE_SHA"):
        if source_commit != env["EXPECTED_SOURCE_SHA"]:
            raise RuntimeError("Prepared image source does not match the requested workflow commit")
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=repo, check=True, capture_output=True)
    pool = preflight(repo, run_dir / f"node-{index}" / "npu_preflight.json")
    resources, cluster = resolve_cluster(env, pool)
    cases = parse_cases(env["CONFIG_YAML_PATH"], resources, repo)
    if resources.num_nodes > 1 and len(cases) > 1 and coord_root is None:
        raise ValueError("Multinode multi-case runs require COORD_DIR and RUN_ID")
    digest = hashlib.sha256(Path(cases[0].config_path).read_bytes()).hexdigest()
    versions = {package: importlib.metadata.version(package) for package in ("vllm", "vllm-ascend")}
    metadata = {
        "run_id": run_id,
        "node_index": index,
        "resources": asdict(resources),
        "config_path": cases[0].config_path,
        "config_digest": digest,
        "case_names": [case.name for case in cases],
        "versions": versions,
        "source_commit": source_commit,
        "image": env.get("SCHEDULE_IMAGE"),
        "framework_digest": hashlib.sha256(
            b"".join(path.read_bytes() for path in sorted((repo / "tests/e2e/schedule").glob("*.py")))
        ).hexdigest(),
        "started_at": time.time(),
    }
    write_json(run_dir / f"node-{index}" / "run_metadata.json", metadata)
    for ordinal, case in enumerate(cases):
        run_case(case, cluster, repo, run_dir, run_id, f"{ordinal:03d}-{safe_name(case.name)}", digest, coord_root)


@dataclass(frozen=True)
class _TestServer:
    endpoint: Endpoint

    def url_for(self, *parts):
        return self.endpoint.url + "/" + "/".join(parts)


def _command_value(argv, flag):
    for index, value in enumerate(argv):
        if value == flag:
            return argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def execute_tests(payload: dict) -> dict:
    """Reuse request/benchmark tools inside a bounded, owned subprocess."""
    import requests

    from tools.send_request import resolve_prompt, send_v1_chat_completions, send_v1_completions

    test = TestSpec(**payload["test"])
    completion, tokenize, metrics = (_TestServer(Endpoint(**payload[k])) for k in ("completion", "tokenize", "metrics"))
    argv = payload["server_argv"]
    max_len = _command_value(argv, "--max-model-len")
    baseline = None
    num_spec = None
    if "spec_decode_acceptance" in test.test_content:
        from tools.spec_decode_metrics import capture_baseline

        speculative = json.loads(_command_value(argv, "--speculative-config") or "{}")
        num_spec = test.acceptance_rate.get("num_speculative_tokens", speculative.get("num_speculative_tokens"))
        if num_spec is None or "baseline" not in test.acceptance_rate:
            raise ValueError("Spec acceptance requires baseline and num_speculative_tokens in config/command")

        def warmup():
            response = requests.post(
                completion.url_for("v1", "chat", "completions"),
                json={"model": test.model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16},
                timeout=120,
            )
            response.raise_for_status()

        baseline = capture_baseline(metrics, num_spec, warmup)
    for content in test.test_content:
        if content in ("completion", "chat_completion"):
            prompts = test.prompts or ["Hello!"]
            arguments = (
                test.api_keyword_args
                if isinstance(test.api_keyword_args, list)
                else [test.api_keyword_args] * len(prompts)
            )
            expected = test.expected_response.get("per_prompt", [test.expected_response] * len(prompts))
            if len(arguments) != len(prompts) or len(expected) != len(prompts):
                raise ValueError("Per-prompt arguments and expected responses must match prompts")
            for raw, args, exp in zip(prompts, arguments, expected):
                prompt, tokens = resolve_prompt(tokenize, raw, use_chat=content == "chat_completion")
                exp = dict(exp)
                if tokens is not None:
                    exp.setdefault("prompt_tokens", tokens)
                send = send_v1_chat_completions if content == "chat_completion" else send_v1_completions
                send(prompt, test.model, completion, args, exp, int(max_len) if max_len else None)
        elif content == "image":
            from tools.send_mm_request import send_image_request

            send_image_request(test, completion)
    results = []
    if test.benchmarks:
        from tools.aisbench import BENCHMARK_HOME, DATASET_CONF_DIR, REQUEST_CONF_DIR, AisbenchRunner

        class ScheduledAisbenchRunner(AisbenchRunner):
            # Keep the existing config generation and result validation, but use
            # a per-run combined config and a foreground child in our PGID.
            def _run_aisbench_task(self):
                from ais_bench.benchmark.cli.utils import recur_convert_config_type
                from mmengine.config import Config

                suffix = "_custom" if self.task_type == "performance" else ""
                model = Config.fromfile(
                    str(Path(REQUEST_CONF_DIR) / f"{self.request_conf}_custom.py"), format_python_code=False
                )
                dataset = Config.fromfile(
                    str(Path(DATASET_CONF_DIR) / f"{self.dataset_conf}{suffix}.py"), format_python_code=False
                )
                datasets = [entry for key, entries in dataset.items() if key.endswith("_datasets") for entry in entries]
                if not datasets:
                    raise ValueError(f"No *_datasets list in {self.dataset_conf}")
                if self.num_prompts:
                    for entry in datasets:
                        # AISBench otherwise ignores --num-prompts when a copied
                        # dataset config already declares a test_range.
                        entry.get("reader_cfg", {}).pop("test_range", None)
                summarizer_path = "perf/default_perf.py" if self.task_type == "performance" else "example.py"
                summarizer = Config.fromfile(
                    str(Path(BENCHMARK_HOME) / "ais_bench/benchmark/configs/summarizers" / summarizer_path),
                    format_python_code=False,
                )
                combined = Path.cwd() / "aisbench_config.py"
                config = Config(
                    dict(models=model.models, datasets=datasets, summarizer=summarizer.summarizer),
                    format_python_code=False,
                )
                recur_convert_config_type(config)
                config.dump(str(combined))
                cmd = ["ais_bench", str(combined), "--debug"]
                if self.task_type == "performance":
                    cmd += ["--mode", "perf"]
                if self.num_prompts:
                    cmd += ["--num-prompts", str(self.num_prompts)]
                self.stdout_file = f"output_{self.task_type}.txt"
                with open(self.stdout_file, "w") as stream:
                    self.proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)

            def _check_runtime_stdout(self):
                time.sleep(0.5)
                text = Path(self.stdout_file).read_text(errors="replace")
                if self.proc.poll() not in (None, 0):
                    raise RuntimeError(f"AISBench exited with {self.proc.returncode}: {text[-4000:]}")
                if self.proc.poll() == 0 and self.RESULT_MSG[self.task_type] not in text:
                    raise RuntimeError("AISBench exited without a result marker")
                return text

        test_dir = Path.cwd()
        for ordinal, (name, benchmark) in enumerate(test.benchmarks.items()):
            output_dir = test_dir / "benchmarks" / f"{ordinal:03d}-{safe_name(name)}"
            output_dir.mkdir(parents=True)
            try:
                os.chdir(output_dir)
                with ScheduledAisbenchRunner(
                    test.model,
                    completion.endpoint.port,
                    benchmark,
                    host_ip=completion.endpoint.host,
                    metrics_server=metrics,
                ) as runner:
                    results.append(runner.result)
            finally:
                os.chdir(test_dir)
        if len(results) != len(test.benchmarks):
            raise RuntimeError("Missing benchmark results")
    if baseline is not None:
        from tools.spec_decode_metrics import measure_acceptance_rate, validate_acceptance_rate

        _, rates = measure_acceptance_rate(metrics, num_spec, baseline)
        validate_acceptance_rate(
            rates[0], float(test.acceptance_rate["baseline"]), float(test.acceptance_rate.get("tolerance", 0.05))
        )
    if "benchmark_comparisons" in test.test_content:
        from tools.aisbench import get_TTFT

        ttft = dict(zip(test.benchmarks, get_TTFT(results)))
        operations = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
        if not test.benchmark_comparisons:
            raise ValueError("benchmark_comparisons test requires comparison rules")
        for comparison in test.benchmark_comparisons:
            if comparison.get("metric", "TTFT") != "TTFT" or comparison.get("operator", "<") not in operations:
                raise ValueError(f"Unsupported benchmark comparison {comparison}")
            actual = ttft[comparison["target"]]
            threshold = ttft[comparison["baseline"]] * comparison.get("ratio", 1)
            if not operations[comparison.get("operator", "<")](actual, threshold):
                raise AssertionError(f"Benchmark comparison failed: {comparison}; {actual} vs {threshold}")
    # Keep the tools' metric structure; DataFrames are converted only for JSON.
    serializable = [
        [value[0].to_dict(), value[1]]
        if isinstance(value, list) and len(value) == 2 and hasattr(value[0], "to_dict")
        else value
        for value in results
    ]
    return {
        "status": "passed",
        "test_content": list(test.test_content),
        "benchmarks": dict(zip(test.benchmarks, serializable)),
        "test_endpoints": {k: payload[k] for k in ("completion", "tokenize", "metrics")},
    }


def cleanup_isolated_container():
    """Opt-in startup cleanup; callers must use a dedicated PID namespace."""
    import psutil

    if Path("/proc/1/comm").read_text().strip() in ("systemd", "init"):
        raise RuntimeError("Startup cleanup refused in a host-like PID namespace")
    excluded = {1, os.getpid(), *(parent.pid for parent in psutil.Process().parents())}
    candidates = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        args = process.info["cmdline"] or []
        if process.pid not in excluded and (
            process.info["name"].startswith("VLLM")
            or (any(Path(arg).name == "vllm" for arg in args) and "serve" in args)
        ):
            candidates.append(process)
    for process in candidates:
        with suppress(psutil.NoSuchProcess):
            process.terminate()
    _, alive = psutil.wait_procs(candidates, timeout=10)
    for process in alive:
        with suppress(psutil.NoSuchProcess):
            process.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup_isolated_container()
    elif args.test_input:
        write_json(args.test_output, execute_tests(json.loads(args.test_input.read_text())))
    else:
        run_from_environment()


if __name__ == "__main__":
    main()
