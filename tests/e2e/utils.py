#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/utils.py
# Copyright 2023 The vLLM team.
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
#

import argparse
import contextlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import openai
import requests
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULT_START_TIMEOUT = 2800.0
_PROCESS_SHUTDOWN_TIMEOUT = 15.0
_LEGACY_PROCESS_SHUTDOWN_TIMEOUT = 60.0
_PROCESS_KILL_TIMEOUT = 10.0


# =============================================================================
# Embedding Utilities
# =============================================================================


def check_embeddings_close(
    *,
    embeddings_0_lst: Sequence[list[float]],
    embeddings_1_lst: Sequence[list[float]],
    name_0: str,
    name_1: str,
    tol: float = 1e-3,
) -> None:
    assert len(embeddings_0_lst) == len(embeddings_1_lst)

    for prompt_idx, (embeddings_0, embeddings_1) in enumerate(zip(embeddings_0_lst, embeddings_1_lst)):
        assert len(embeddings_0) == len(embeddings_1), f"Length mismatch: {len(embeddings_0)} vs. {len(embeddings_1)}"

        sim = F.cosine_similarity(torch.tensor(embeddings_0), torch.tensor(embeddings_1), dim=0)

        fail_msg = (
            f"Test{prompt_idx}:"
            f"\nCosine similarity: \t{sim:.4f}"
            f"\n{name_0}:\t{embeddings_0[:16]!r}"
            f"\n{name_1}:\t{embeddings_1[:16]!r}"
        )

        assert sim >= 1 - tol, fail_msg


# =============================================================================
# HTTP / Endpoint Utilities
# =============================================================================


def _check_processes(processes, *, fail_on_clean_exit: bool) -> bool:
    """Return whether a process exited cleanly; reject exits invalid for this phase."""
    clean_exit = False
    for process in processes:
        result = process.poll()
        if result is None:
            continue
        if fail_on_clean_exit:
            raise RuntimeError(f"Server process exited before readiness with code {result}.")
        if result != 0:
            raise RuntimeError(f"Server process exited unexpectedly with code {result}.")
        clean_exit = True
    return clean_exit


def wait_for_http_targets(
    urls: Sequence[str],
    timeout: float,
    *,
    poll_processes=None,
    always_check: bool = False,
    poll_interval: float = 1.0,
    request_timeout: float = 5.0,
    client=None,
) -> None:
    """Wait for HTTP 200 from every URL, without taking ownership of processes."""
    deadline = time.monotonic() + timeout
    ready = dict.fromkeys(urls, False)
    processes = () if poll_processes is None else poll_processes
    client = requests if client is None else client
    while True:
        _check_processes(processes, fail_on_clean_exit=True)
        if always_check:
            ready = dict.fromkeys(ready, False)
        for url in ready:
            if ready[url] and not always_check:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = client.get(url, timeout=min(request_timeout, remaining))
                ready[url] = response.status_code == 200
                response.close()
            except (requests.RequestException, httpx.RequestError):
                ready[url] = False
            _check_processes(processes, fail_on_clean_exit=True)
        if all(ready.values()):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending = [url for url, is_ready in ready.items() if not is_ready]
            raise TimeoutError(f"Timed out after {timeout}s waiting for HTTP readiness: {pending}")
        time.sleep(min(poll_interval, remaining))


def wait_for_http_ready(url: str, timeout: float, *, process=None, **kwargs) -> None:
    wait_for_http_targets(urls=[url], timeout=timeout, poll_processes=() if process is None else [process], **kwargs)


def wait_for_http_unready(
    url: str,
    timeout: float | None = None,
    *,
    process=None,
    poll_interval: float = 5.0,
    request_timeout: float = 5.0,
) -> None:
    """Wait for the master to stop or the local worker to exit cleanly."""
    deadline = None if timeout is None else time.monotonic() + timeout
    processes = () if process is None else [process]
    while True:
        if _check_processes(processes, fail_on_clean_exit=False):
            return
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"Timed out waiting for HTTP endpoint to stop: {url}")
        try:
            response = requests.get(
                url, timeout=request_timeout if remaining is None else min(request_timeout, remaining)
            )
            ready = response.status_code == 200
            response.close()
        except requests.RequestException:
            ready = False
        if _check_processes(processes, fail_on_clean_exit=False) or not ready:
            return
        time.sleep(poll_interval if remaining is None else min(poll_interval, max(0, deadline - time.monotonic())))


def _http_root(host: str, port: int) -> str:
    # Wildcard bind addresses are not destinations, especially with HTTP proxies.
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


class _OpenAIEndpoint:
    """Shared URL/client API for servers and proxies."""

    DUMMY_API_KEY = "token-abc123"
    host: str
    port: int
    uds: str | None = None

    @property
    def url_root(self) -> str:
        return "http://localhost" if self.uds else _http_root(self.host, self.port)

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs) -> openai.OpenAI:
        kwargs.setdefault("timeout", 600)
        kwargs.setdefault("api_key", self.DUMMY_API_KEY)
        kwargs.setdefault("max_retries", 0)
        if self.uds and "http_client" not in kwargs:
            kwargs["http_client"] = httpx.Client(transport=httpx.HTTPTransport(uds=self.uds))
        return openai.OpenAI(base_url=self.url_for("v1"), **kwargs)

    def get_async_client(self, **kwargs) -> openai.AsyncOpenAI:
        kwargs.setdefault("timeout", 600)
        kwargs.setdefault("api_key", self.DUMMY_API_KEY)
        kwargs.setdefault("max_retries", 0)
        if self.uds and "http_client" not in kwargs:
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=self.uds))
        return openai.AsyncOpenAI(base_url=self.url_for("v1"), **kwargs)


# =============================================================================
# Process Lifecycle Utilities
# =============================================================================


def _redact_sensitive_cli_args(args: Sequence[str]) -> list[str]:
    redacted = list(args)
    index = 0
    while index < len(args):
        name, separator, _ = args[index].partition("=")
        name = name.replace("_", "-")
        if name not in ("--api-key", "--hf-token"):
            index += 1
            continue
        if separator:
            redacted[index] = args[index].split("=", 1)[0] + "=***"
        index += 1
        if not separator or name == "--api-key":
            while index < len(args) and not args[index].startswith("-"):
                redacted[index] = "***"
                index += 1
                if name == "--hf-token":
                    break
    return redacted


def _find_pgid_members(pgid: int) -> list[int]:
    """Find living Linux group members even after the root has been reaped."""
    proc_path = Path("/proc")
    if not proc_path.is_dir():
        return []
    members = []
    for entry in proc_path.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses. Fields after it start at state.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                members.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return members


class _ManagedProcess:
    """Own one dedicated process group, including orphaned workers."""

    def __init__(
        self,
        command: Sequence[str],
        env: dict[str, str] | None = None,
        *,
        log_prefix: str | None = None,
        shutdown_timeout: float = _PROCESS_SHUTDOWN_TIMEOUT,
        cwd: str | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        self.command = list(command)
        self.env_overrides = dict(env or {})
        self.env = os.environ.copy()
        if env is not None:
            self.env.update(env)
        self.log_prefix = log_prefix
        self.shutdown_timeout = shutdown_timeout
        self.cwd = cwd
        self.log_file = Path(log_file) if log_file is not None else None
        self.proc: subprocess.Popen | None = None
        self.pgid: int | None = None
        self._threads: list[threading.Thread] = []
        self._shutdown_complete = False

    def start(self) -> None:
        if self.proc is not None or self._shutdown_complete:
            raise RuntimeError("A managed process can only be started once")
        logger.info("Starting process: %s", shlex.join(_redact_sensitive_cli_args(self.command)))
        with contextlib.ExitStack() as stack:
            output = None
            if self.log_file is not None:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                output = stack.enter_context(self.log_file.open("w", encoding="utf-8"))
                output.write(shlex.join(_redact_sensitive_cli_args(self.command)) + "\n")
                output.flush()
            self.proc = subprocess.Popen(
                self.command,
                env=self.env,
                cwd=self.cwd,
                start_new_session=True,
                stdout=output if output else subprocess.PIPE if self.log_prefix else None,
                stderr=subprocess.STDOUT if output else subprocess.PIPE if self.log_prefix else None,
                text=True,
            )
        # Save before the root can exit; querying its PGID during cleanup is too late.
        self.pgid = self.proc.pid
        if self.log_prefix and self.log_file is None:
            for pipe in (self.proc.stdout, self.proc.stderr):
                thread = threading.Thread(target=self._read_output, args=(pipe,), daemon=True)
                self._threads.append(thread)
                thread.start()

    def _read_output(self, pipe) -> None:
        with pipe:
            for line in pipe:
                print(f"{self.log_prefix}{line}", end="", flush=True)

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def _kill_group(self) -> None:
        if self.pgid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.pgid, signal.SIGKILL)

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        if self.proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self._kill_group()
                self.proc.wait(timeout=_PROCESS_KILL_TIMEOUT)
            self._kill_group()
            deadline = time.monotonic() + _PROCESS_KILL_TIMEOUT
            while self.pgid is not None and (survivors := _find_pgid_members(self.pgid)):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Process group {self.pgid} still has live members: {survivors}")
                self._kill_group()
                time.sleep(0.1)
            for thread in self._threads:
                thread.join(timeout=1)
        self._shutdown_complete = True


# =============================================================================
# Remote vLLM Server
# =============================================================================


def _has_cli_option(args: Sequence[str], *names: str) -> bool:
    return any(arg.split("=", 1)[0].replace("_", "-") in names for arg in args)


def _parse_serve_args(args: list[str]) -> argparse.Namespace:
    # Lazy imports keep embedding and process-only utilities independent of vLLM startup.
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="Ascend E2E server")
    subparsers = parser.add_subparsers(dest="subparser")
    serve_parser = ServeSubcommand().subparser_init(subparsers)
    return serve_parser.parse_args(args)


def _vllm_shutdown_timeout(args: argparse.Namespace) -> float:
    from vllm.v1.engine import utils as engine_utils

    timeout = float(getattr(args, "shutdown_timeout", 0))
    get_engine_timeout = getattr(engine_utils, "get_engine_process_shutdown_timeout", None)
    if get_engine_timeout is None:
        # Older supported vLLM checkouts do not expose the engine helper yet.
        # Preserve Ascend's previous grace period while allowing request draining.
        return max(_LEGACY_PROCESS_SHUTDOWN_TIMEOUT, timeout + _PROCESS_SHUTDOWN_TIMEOUT)
    engine_timeout = get_engine_timeout(timeout, timeout)
    assert engine_timeout is not None
    return engine_timeout + _PROCESS_SHUTDOWN_TIMEOUT


class RemoteVLLMServer(_OpenAIEndpoint):
    """Common lifecycle of ``vllm serve`` processes used by Ascend E2E tests.

    Cluster orchestration belongs to the caller.
    """

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str] | str,
        *,
        server_host: str = "0.0.0.0",
        server_port: int = 8080,
        env_dict: dict[str, str] | None = None,
        seed: int | None = None,
        auto_port: bool = True,
        max_wait_seconds: float | None = None,
        override_hf_configs: dict[str, Any] | None = None,
        wait_for_ready: bool = True,
        cwd: str | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        if isinstance(vllm_serve_args, str):
            command = shlex.split(vllm_serve_args)
            if command[:2] != ["vllm", "serve"]:
                raise ValueError("String server arguments must be a complete 'vllm serve MODEL ...' command")
            args = command[2:]
        else:
            args = [model, *map(str, vllm_serve_args)]
        if not _has_cli_option(args, "--host"):
            args += ["--host", server_host]
        if not _has_cli_option(args, "--port", "-p", "--uds"):
            if auto_port:
                from vllm.utils.network_utils import get_open_port

                server_port = get_open_port()
            args += ["--port", str(server_port)]
        if seed is not None:
            if _has_cli_option(args, "--seed"):
                raise ValueError(f"You have manually specified the seed when seed={seed}.")
            args += ["--seed", str(seed)]
        if override_hf_configs is not None:
            args += ["--hf-overrides", json.dumps(override_hf_configs)]
        parsed = _parse_serve_args(args)
        self.host = str(parsed.host or "127.0.0.1")
        self.port = int(parsed.port)
        self.uds = parsed.uds
        self._process = self._make_process(args, parsed, env_dict)
        self._process.cwd = cwd
        self._process.log_file = Path(log_file) if log_file is not None else None
        try:
            self._start_server()
            if wait_for_ready:
                self.wait_ready(_DEFAULT_START_TIMEOUT if max_wait_seconds is None else max_wait_seconds)
        except BaseException as exc:
            try:
                self._shutdown()
            except BaseException as cleanup_error:
                exc.add_note(f"Startup cleanup failed: {cleanup_error}")
            raise

    def _make_process(
        self, args: list[str], parsed: argparse.Namespace, env_dict: dict[str, str] | None
    ) -> _ManagedProcess:
        raise NotImplementedError

    def _start_server(self) -> None:
        self._process.start()

    @property
    def proc(self) -> subprocess.Popen | None:
        return self._process.proc

    def poll(self) -> int | None:
        return self._process.poll()

    def wait_ready(self, timeout: float = _DEFAULT_START_TIMEOUT) -> None:
        if self.uds:
            with httpx.Client(transport=httpx.HTTPTransport(uds=self.uds)) as client:
                wait_for_http_ready(self.url_for("health"), timeout, process=self, client=client)
        else:
            wait_for_http_ready(self.url_for("health"), timeout, process=self)

    def _shutdown(self) -> None:
        self._process.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._shutdown()


class RemoteOpenAIServer(RemoteVLLMServer):
    """Launch ``vllm serve`` with spawn-safe workers and OpenAI clients."""

    def _make_process(
        self, args: list[str], parsed: argparse.Namespace, env_dict: dict[str, str] | None
    ) -> _ManagedProcess:
        env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", **(env_dict or {})}
        return _ManagedProcess(["vllm", "serve", *args], env, shutdown_timeout=_vllm_shutdown_timeout(parsed))


# =============================================================================
# Remote Server Groups
# =============================================================================


def _get_pd_server_required_devices(vllm_serve_args: list[str]) -> int:
    def get_size(*names: str) -> int:
        value = 1
        for index, arg in enumerate(vllm_serve_args):
            name, separator, inline = arg.partition("=")
            if name.replace("_", "-") in names:
                value = int(inline if separator else vllm_serve_args[index + 1])
        if value <= 0:
            raise ValueError(f"{names[0]} must be positive, got {value}.")
        return value

    dp_names = ("--data-parallel-size-local", "-dpl")
    if not _has_cli_option(vllm_serve_args, *dp_names):
        dp_names = ("--data-parallel-size", "-dp")
    tp_size = get_size("--tensor-parallel-size", "-tp")
    pp_size = get_size("--pipeline-parallel-size", "-pp")
    pcp_size = get_size("--prefill-context-parallel-size", "-pcp")
    dp_size_local = get_size(*dp_names)
    return tp_size * pp_size * pcp_size * dp_size_local


class RemoteServerGroup(_OpenAIEndpoint):
    """Start all siblings before readiness; clean up every sibling on failure."""

    def __init__(
        self,
        processes: Sequence[_ManagedProcess],
        health_urls: Sequence[str],
        *,
        timeout: float,
        wait_for_ready: bool = True,
    ) -> None:
        self.processes = list(processes)
        self.health_urls = list(health_urls)
        try:
            for process in self.processes:
                process.start()
            if wait_for_ready:
                self.wait_all_ready(timeout)
        except BaseException as exc:
            try:
                self._shutdown()
            except Exception as cleanup_error:
                exc.add_note(f"Server group cleanup also failed: {cleanup_error}")
            raise

    def poll_all(self) -> list[int | None]:
        return [process.poll() for process in self.processes]

    def wait_all_ready(self, timeout: float) -> None:
        wait_for_http_targets(self.health_urls, timeout, poll_processes=self.processes, always_check=True)

    def _shutdown(self) -> None:
        if not self.processes:
            return
        with ThreadPoolExecutor(max_workers=len(self.processes)) as executor:
            futures = [executor.submit(process.shutdown) for process in self.processes]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError(f"Server group cleanup failed: {errors}") from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._shutdown()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Server group cleanup also failed: {cleanup_error}")


def _allocate_server_devices(
    args_list: Sequence[Sequence[str]], per_server_envs: Sequence[dict[str, str]], visible: str | None
) -> list[str]:
    """Reserve all explicit assignments before filling automatic assignments."""

    def parse(value):
        ids = [str(int(item.strip())) for item in value.split(",") if item.strip()]
        if not ids or any(int(i) < 0 for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES: {value!r}")
        return ids

    pool = parse(visible) if visible is not None else None
    reserved, assignments = set(), {}
    for index, env in enumerate(per_server_envs):
        if "ASCEND_RT_VISIBLE_DEVICES" not in env:
            continue
        ids = parse(env["ASCEND_RT_VISIBLE_DEVICES"])
        required = _get_pd_server_required_devices(list(args_list[index]))
        if len(ids) < required:
            raise ValueError(f"Server {index}: requires {required} devices but assignment has {len(ids)}")
        if reserved.intersection(ids) or (pool is not None and not set(ids) <= set(pool)):
            raise ValueError(f"Server {index}: overlapping or out-of-pool visible devices {ids}")
        reserved.update(ids)
        assignments[index] = ids
    for index, args in enumerate(args_list):
        if index in assignments:
            continue
        required = _get_pd_server_required_devices(list(args))
        available = (str(i) for i in range(len(reserved) + required)) if pool is None else iter(pool)
        selected = [i for i in available if i not in reserved][:required]
        if len(selected) != required:
            raise ValueError(
                f"Server {index} needs {required} devices, but only {len(selected)} remain in ASCEND_RT_VISIBLE_DEVICES"
            )
        reserved.update(selected)
        assignments[index] = selected
    return [
        per_server_envs[index].get("ASCEND_RT_VISIBLE_DEVICES", ",".join(assignments[index]))
        for index in range(len(args_list))
    ]


def _prepare_server_group(
    vllm_serve_args: list[str] | list[list[str]],
    server_host: str,
    env_dict: dict[str, str] | None,
    prefix: str,
    *,
    per_server_envs: Sequence[dict[str, str]] | None = None,
    working_dirs: Sequence[str | None] | None = None,
    log_files: Sequence[str | Path | None] | None = None,
    health_urls: Sequence[str] | None = None,
) -> tuple[list[_ManagedProcess], list[str], str, int]:
    if not isinstance(vllm_serve_args, list) or not vllm_serve_args:
        raise ValueError("vllm_serve_args must be a nonempty list")
    args_list = vllm_serve_args if isinstance(vllm_serve_args[0], list) else [vllm_serve_args]
    env = dict(
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
        PYTORCH_NPU_ALLOC_CONF=os.environ.get("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"),
        VLLM_WORKER_MULTIPROC_METHOD=os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
    )
    env.update(env_dict or {})
    per_server_envs = list(per_server_envs) if per_server_envs is not None else [{} for _ in args_list]
    working_dirs = list(working_dirs) if working_dirs is not None else [None] * len(args_list)
    log_files = list(log_files) if log_files is not None else [None] * len(args_list)
    if any(len(values) != len(args_list) for values in (per_server_envs, working_dirs, log_files)):
        raise ValueError("Per-server env/cwd/log counts must match server commands")
    visible = env.get("ASCEND_RT_VISIBLE_DEVICES", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    assignments = _allocate_server_devices(args_list, per_server_envs, visible)
    processes, urls = [], []
    for index, original_args in enumerate(args_list):
        args = list(map(str, original_args))
        if health_urls is None and not _has_cli_option(args, "--port", "-p"):
            raise ValueError("You have to manually specify the port")
        if not _has_cli_option(args, "--host"):
            args += ["--host", server_host]
        parsed = _parse_serve_args(args)
        if parsed.uds:
            raise ValueError("Server groups require TCP endpoints")
        server_env = {**env, **per_server_envs[index], "ASCEND_RT_VISIBLE_DEVICES": assignments[index]}
        processes.append(
            _ManagedProcess(
                ["vllm", "serve", *args],
                server_env,
                log_prefix=f"[{prefix}_{index}] ",
                shutdown_timeout=_vllm_shutdown_timeout(parsed),
                cwd=working_dirs[index],
                log_file=log_files[index],
            )
        )
        urls.append(_http_root(parsed.host, parsed.port) + "/health")
    return processes, list(health_urls) if health_urls is not None else urls, parsed.host, parsed.port


class RemotePDServer(RemoteServerGroup):
    def __init__(
        self,
        vllm_serve_args: list[str] | list[list[str]],
        server_host: str = "127.0.0.1",
        env_dict: dict[str, str] | None = None,
        max_wait_seconds: float | None = 600,
        *,
        per_server_envs: Sequence[dict[str, str]] | None = None,
        working_dirs: Sequence[str | None] | None = None,
        log_files: Sequence[str | Path | None] | None = None,
        health_urls: Sequence[str] | None = None,
        wait_for_ready: bool = True,
    ) -> None:
        processes, urls, self.host, self.port = _prepare_server_group(
            vllm_serve_args,
            server_host,
            env_dict,
            "PD",
            per_server_envs=per_server_envs,
            working_dirs=working_dirs,
            log_files=log_files,
            health_urls=health_urls,
        )
        super().__init__(
            processes,
            urls,
            timeout=_DEFAULT_START_TIMEOUT if max_wait_seconds is None else max_wait_seconds,
            wait_for_ready=wait_for_ready,
        )


class RemoteEPDServer(RemoteServerGroup):
    def __init__(
        self,
        vllm_serve_args: list[str] | list[list[str]],
        server_host: str = "0.0.0.0",
        env_dict: dict[str, str] | None = None,
        max_wait_seconds: float | None = 2800,
        *,
        per_server_envs: Sequence[dict[str, str]] | None = None,
        working_dirs: Sequence[str | None] | None = None,
        log_files: Sequence[str | Path | None] | None = None,
        health_urls: Sequence[str] | None = None,
        wait_for_ready: bool = True,
    ) -> None:
        processes, urls, self.host, self.port = _prepare_server_group(
            vllm_serve_args,
            server_host,
            env_dict,
            "VLLM",
            per_server_envs=per_server_envs,
            working_dirs=working_dirs,
            log_files=log_files,
            health_urls=health_urls,
        )
        super().__init__(
            processes,
            urls,
            timeout=_DEFAULT_START_TIMEOUT if max_wait_seconds is None else max_wait_seconds,
            wait_for_ready=wait_for_ready,
        )


# =============================================================================
# Disaggregated Proxy
# =============================================================================

DISAGG_PD_PROXY_SCRIPT = (
    Path(__file__).parents[2] / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
)
DISAGG_EPD_PROXY_SCRIPT = Path(__file__).parents[2] / "examples/disaggregated_encoder/disagg_epd_proxy.py"


class RemoteProxy(_OpenAIEndpoint):
    def __init__(
        self,
        command: Sequence[str],
        *,
        host: str,
        port: int,
        health_path: str,
        env_dict: dict[str, str] | None = None,
        timeout: float = 600,
        log_prefix: str | None = None,
        cwd: str | None = None,
        log_file: str | Path | None = None,
        wait_for_ready: bool = True,
    ) -> None:
        self.host, self.port = host, int(port)
        self._process = _ManagedProcess(command, env_dict, log_prefix=log_prefix, cwd=cwd, log_file=log_file)
        try:
            self._process.start()
            if wait_for_ready:
                wait_for_http_ready(self.url_for(health_path), timeout, process=self._process)
        except BaseException as exc:
            try:
                self._shutdown()
            except BaseException as cleanup_error:
                exc.add_note(f"Startup cleanup failed: {cleanup_error}")
            raise

    def _shutdown(self) -> None:
        self._process.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._shutdown()


class DisaggPDProxy(RemoteProxy):
    def __init__(
        self,
        port: int,
        prefiller_ports: list[int],
        decoder_ports: list[int],
        host: str = "127.0.0.1",
        env_dict: dict[str, str] | None = None,
        max_wait_seconds: float | None = 600,
    ) -> None:
        args = [
            "--host",
            host,
            "--port",
            str(port),
            "--prefiller-hosts",
            *[host] * len(prefiller_ports),
            "--prefiller-ports",
            *map(str, prefiller_ports),
            "--decoder-hosts",
            *[host] * len(decoder_ports),
            "--decoder-ports",
            *map(str, decoder_ports),
        ]
        super().__init__(
            [sys.executable, str(DISAGG_PD_PROXY_SCRIPT), *args],
            host=host,
            port=port,
            health_path="healthcheck",
            env_dict=env_dict,
            timeout=600 if max_wait_seconds is None else max_wait_seconds,
            log_prefix="[PD_PROXY] ",
        )


class DisaggEpdProxy(RemoteProxy):
    def __init__(
        self,
        proxy_args: list[str] | str | None = None,
        env_dict: dict[str, str] | None = None,
        server_host: str = "0.0.0.0",
        max_wait_seconds: float | None = 2800,
    ) -> None:
        args = shlex.split(proxy_args) if isinstance(proxy_args, str) else list(proxy_args or [])
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--host", default=server_host)
        parser.add_argument("--port", type=int, required=True)
        parsed, _ = parser.parse_known_args(args)
        if not _has_cli_option(args, "--host"):
            args += ["--host", server_host]
        super().__init__(
            [sys.executable, str(DISAGG_EPD_PROXY_SCRIPT), *args],
            host=parsed.host,
            port=parsed.port,
            health_path="health",
            env_dict=env_dict,
            timeout=_DEFAULT_START_TIMEOUT if max_wait_seconds is None else max_wait_seconds,
            log_prefix="[PROXY] ",
        )
