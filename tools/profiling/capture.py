"""Capture runtime profiles and publish raw artifacts from benchmark jobs."""

import argparse
import json
import logging
import os
import shlex
import shutil
import tarfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from tools.profiling.storage import (
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_REGION,
    GZIP_COMPRESSION_LEVEL,
    create_obs_client,
    create_transfer_config,
    directory_size_bytes,
    upload_archive,
    upload_json,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and serving topology
# =============================================================================

ServeRole = Literal["unified", "prefill", "decode"]
ProfileScope = Literal["representative", "all"]
ProfileOutput = Literal["parsed", "raw"]


def _env_bool(name: str, default: bool) -> bool:
    """Read a strict, case-insensitive ``true``/``false`` environment value."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}; expected true or false")


@dataclass(frozen=True)
class ProfileSpec:
    """User-selectable settings for one scheduled profiling session.

    ``start_after`` and ``duration`` are seconds relative to the first AISBench
    request. ``scope`` controls target selection, while ``cases`` optionally
    restricts profiling to named performance cases.
    """

    enabled: bool = False
    start_after: int = 15
    duration: int = 8
    with_stack: bool = False
    scope: ProfileScope = "representative"
    output: ProfileOutput = "parsed"
    cases: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ProfileSpec":
        """Build and validate a specification from ``ASCEND_PROFILE_*`` values."""
        defaults = cls()
        enabled = _env_bool("ASCEND_PROFILE_ENABLED", defaults.enabled)
        if not enabled:
            return defaults
        cases = os.getenv("ASCEND_PROFILE_CASES")
        scope = os.getenv("ASCEND_PROFILE_SCOPE", defaults.scope)
        output = os.getenv("ASCEND_PROFILE_OUTPUT", defaults.output)
        if scope not in ("representative", "all") or output not in ("parsed", "raw"):
            raise ValueError("Invalid profile scope or output mode")
        spec = cls(
            enabled=True,
            start_after=int(os.getenv("ASCEND_PROFILE_START_AFTER", defaults.start_after)),
            duration=int(os.getenv("ASCEND_PROFILE_DURATION", defaults.duration)),
            with_stack=_env_bool("ASCEND_PROFILE_WITH_STACK", defaults.with_stack),
            scope=cast(ProfileScope, scope),
            output=cast(ProfileOutput, output),
            cases=defaults.cases if cases is None else tuple(case.strip() for case in cases.split(",") if case.strip()),
        )
        if spec.start_after < 0 or spec.duration <= 0:
            raise ValueError("Profile times must be positive (start-after may be zero)")
        return spec

    def includes_case(self, case_name: str, case_type: str) -> bool:
        """Return whether a benchmark case should be profiled."""
        return self.enabled and case_type == "performance" and (not self.cases or case_name in self.cases)


@dataclass(frozen=True)
class ServeInstance:
    """Address and artifact location for one independently profiled server."""

    name: str
    endpoint: str
    raw_trace_dir: str
    role: ServeRole = "unified"
    dp_rank: int = 0
    node_index: int = 0

    @classmethod
    def from_endpoint(
        cls,
        name: str,
        endpoint: str,
        role: ServeRole = "unified",
        dp_rank: int = 0,
        node_index: int = 0,
        *,
        root: Path | None = None,
    ) -> "ServeInstance":
        """Create an instance with a filesystem-safe raw trace directory."""
        import regex as re

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        artifact_root = profile_root() if root is None else root.resolve()
        return cls(name, endpoint, str(artifact_root / "raw" / safe_name), role, dp_rank, node_index)


@dataclass(frozen=True)
class ServeManifest:
    """Serializable inventory of server instances started by a benchmark job."""

    instances: tuple[ServeInstance, ...]

    def write(self, path: Path) -> None:
        """Write the versioned server inventory to ``path`` as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "instances": [asdict(i) for i in self.instances]}, indent=2))

    @classmethod
    def read(cls, path: Path) -> "ServeManifest":
        """Load a server inventory previously written by :meth:`write`."""
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError(f"Unsupported serve manifest version: {data.get('version')!r}")
        return cls(tuple(ServeInstance(**item) for item in data["instances"]))

    def select_targets(self, scope: ProfileScope) -> tuple[ServeInstance, ...]:
        """Resolve ``scope`` into a deterministic tuple of profiling targets."""
        instances = self.instances
        if scope == "all":
            return instances
        roles = {item.role for item in instances}
        if "prefill" in roles or "decode" in roles:
            return tuple(
                min((i for i in instances if i.role == role), key=lambda i: i.dp_rank)
                for role in ("prefill", "decode")
                if role in roles
            )
        return (min(instances, key=lambda i: i.dp_rank),) if instances else ()


def profile_root() -> Path:
    """Return the absolute root used for manifests, traces, and archives."""
    return Path(os.getenv("ASCEND_PROFILE_ROOT", "profile_artifact")).resolve()


def initialize_profile_manifests(root: Path, instances: list[ServeInstance], spec: ProfileSpec) -> None:
    """Start a profiling run with a fresh server and result manifest."""
    ServeManifest(tuple(instances)).write(root / "serve_manifest.json")
    manifest_path = root / "profile_manifest.json"
    manifest_path.write_text(json.dumps({"requested": asdict(spec), "cases": [], "status": "skipped"}))


def configure_profiler_command(command: list[str] | str, instance: ServeInstance, spec: ProfileSpec) -> list[str] | str:
    """Merge only profiler-owned options into the existing server command."""
    command_args = shlex.split(command) if isinstance(command, str) else list(command)
    profiler_flag = "--profiler-config"
    existing = json.loads(command_args[command_args.index(profiler_flag) + 1]) if profiler_flag in command_args else {}
    existing.update(
        profiler="torch",
        torch_profiler_dir=instance.raw_trace_dir,
        torch_profiler_with_stack=spec.with_stack,
        ignore_frontend=True,
        max_iterations=0,
    )
    if profiler_flag in command_args:
        command_args[command_args.index(profiler_flag) + 1] = json.dumps(existing)
    else:
        command_args.extend((profiler_flag, json.dumps(existing)))
    return shlex.join(command_args) if isinstance(command, str) else command_args


# =============================================================================
# Runtime profiling control
# =============================================================================

STOP_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 1
MAX_PROFILE_SIZE_BYTES = 50 * 1024**3


def record_first_request(marker: Path) -> None:
    """Atomically record the first request's monotonic timestamp once."""
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as file:
        file.write(str(time.monotonic()))


class _ProfileClient:
    """HTTP client for vLLM's runtime profiling control endpoints."""

    def start_profile(self, endpoint: str) -> None:
        import requests

        requests.post(f"{endpoint.rstrip('/')}/start_profile", timeout=30).raise_for_status()

    def stop_profile(self, endpoint: str) -> None:
        import requests

        requests.post(f"{endpoint.rstrip('/')}/stop_profile", timeout=STOP_TIMEOUT_SECONDS).raise_for_status()


class ProfileController:
    """Own one benchmark case's start/stop lifecycle without failing the benchmark."""

    def __init__(
        self, spec: ProfileSpec, targets: tuple[ServeInstance, ...], marker: Path, client: _ProfileClient | None = None
    ):
        self.spec, self.targets, self.marker = spec, targets, marker
        self.client = client or _ProfileClient()
        self.finished = threading.Event()
        self.result: dict = {"status": "skipped", "targets": {}}
        self.thread = threading.Thread(target=self._run_profile_window, daemon=True)

    def start(self) -> None:
        """Start monitoring for the first benchmark request."""
        self.thread.start()

    def finish(self) -> dict:
        """Signal benchmark completion, wait for trace shutdown, and return status."""
        self.finished.set()
        self.thread.join(STOP_TIMEOUT_SECONDS + 60)
        if self.thread.is_alive():
            self.result = {"status": "partial", "reason": "controller_timeout", "targets": {}}
        return self.result

    def _call_profile_endpoints(
        self, operation: Callable[[str], None], targets: tuple[ServeInstance, ...]
    ) -> dict[str, str | None]:
        """Call an operation once per unique endpoint and map errors to targets."""
        endpoints: dict[str, list[str]] = {}
        for target in targets:
            endpoints.setdefault(target.endpoint, []).append(target.name)
        with ThreadPoolExecutor(max_workers=len(endpoints) or 1) as pool:
            futures = {pool.submit(operation, endpoint): names for endpoint, names in endpoints.items()}
            errors_by_target = {}
            for future in as_completed(futures):
                try:
                    future.result()
                    error = None
                except Exception as exc:
                    error = str(exc)
                errors_by_target.update(dict.fromkeys(futures[future], error))
            return errors_by_target

    def _run_profile_window(self) -> None:
        """Execute the timed start/monitor/stop lifecycle in the daemon thread."""
        try:
            while not self.finished.wait(POLL_INTERVAL_SECONDS):
                if self.marker.exists() and self.marker.stat().st_size:
                    first_request = float(self.marker.read_text())
                    break
            else:
                self.result = {"status": "skipped", "reason": "no_request_before_benchmark_end", "targets": {}}
                return
            if self.finished.wait(max(0, first_request + self.spec.start_after - time.monotonic())):
                self.result = {"status": "skipped", "reason": "benchmark_ended_before_start", "targets": {}}
                return
            print(f"[Profiling] start_profile -> {', '.join(t.name for t in self.targets)}", flush=True)
            start_errors = self._call_profile_endpoints(self.client.start_profile, self.targets)
            started_at = time.monotonic()
            active_targets = tuple(target for target in self.targets if start_errors[target.name] is None)
            reason = "duration_reached"
            try:
                while active_targets and not self.finished.wait(POLL_INTERVAL_SECONDS):
                    if time.monotonic() - started_at >= self.spec.duration:
                        break
                    if any(
                        directory_size_bytes(Path(target.raw_trace_dir)) >= MAX_PROFILE_SIZE_BYTES
                        for target in active_targets
                    ):
                        reason = "size_limit_exceeded"
                        break
                if self.finished.is_set() and time.monotonic() - started_at < self.spec.duration:
                    reason = "benchmark_ended"
            finally:
                actual_duration = time.monotonic() - started_at
                print(f"[Profiling] stop_profile ({reason}) -> {', '.join(t.name for t in self.targets)}", flush=True)
                # A failed start response may still have started some workers.
                stop_errors = self._call_profile_endpoints(self.client.stop_profile, self.targets)
            self.result = {
                "status": "success"
                if len(active_targets) == len(self.targets)
                and not any(stop_errors.values())
                and reason == "duration_reached"
                and actual_duration >= self.spec.duration
                else "partial",
                "reason": reason,
                "actual_duration": actual_duration,
                "targets": {
                    target.name: {
                        "start_error": start_errors[target.name],
                        "stop_error": stop_errors.get(target.name),
                    }
                    for target in self.targets
                },
            }
        except Exception as exc:
            logger.exception("Profiling controller failed; benchmark result is unchanged")
            self.result = {"status": "partial", "reason": str(exc), "targets": {}}


# =============================================================================
# Raw artifact collection and upload
# =============================================================================

RAW_COMPRESSION_CONCURRENCY = 4
RAW_UPLOAD_CONCURRENCY = 2


def collect_raw_artifacts(
    root: Path,
    case_name: str,
    targets: tuple[ServeInstance, ...],
    profile_result: dict,
    spec: ProfileSpec,
) -> None:
    """Archive raw traces and append one case to the local result manifest.

    Successful target directories are cleared only after their gzip archives
    have closed. Collection failures are recorded instead of escaping into the
    benchmark process.
    """
    import regex as re

    safe_case = re.sub(r"[^A-Za-z0-9_.-]", "_", case_name)
    case_dir = root / safe_case
    case_dir.mkdir(parents=True, exist_ok=True)

    def compress_target(target: ServeInstance) -> dict:
        raw_trace_dir = Path(target.raw_trace_dir)
        archive_path = case_dir / f"{target.name}.tar.gz"
        target_state = dict(profile_result.get("targets", {}).get(target.name, {}))
        try:
            if not raw_trace_dir.exists() or not any(raw_trace_dir.iterdir()):
                raise FileNotFoundError(f"No profiler output in {raw_trace_dir}")
            raw_size_bytes = directory_size_bytes(raw_trace_dir)
            print(f"[Profiling] Compress raw {case_name}/{target.name}: {raw_size_bytes} bytes before", flush=True)
            with tarfile.open(archive_path, "w:gz", compresslevel=GZIP_COMPRESSION_LEVEL) as tar:
                tar.add(raw_trace_dir, arcname=target.name)
            target_state.update(
                archive=str(archive_path.relative_to(root)),
                size_bytes=archive_path.stat().st_size,
                output="raw",
            )
            print(
                f"[Profiling] Compressed raw {case_name}/{target.name}: {archive_path.stat().st_size} bytes after",
                flush=True,
            )
            for child in raw_trace_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        except Exception as exc:
            target_state["artifact_error"] = str(exc)
        return {
            "name": target.name,
            "endpoint": target.endpoint,
            "node_index": target.node_index,
            **target_state,
        }

    with ThreadPoolExecutor(max_workers=RAW_COMPRESSION_CONCURRENCY) as pool:
        target_records = list(pool.map(compress_target, targets))
    case_record = {
        "case": case_name,
        "actual_duration": profile_result.get("actual_duration", 0),
        "status": "partial"
        if profile_result.get("status") != "success" or any(record.get("artifact_error") for record in target_records)
        else "success",
        "reason": profile_result.get("reason"),
        "targets": target_records,
    }
    manifest_path = root / "profile_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"requested": asdict(spec), "cases": []}
    )
    manifest["cases"].append(case_record)
    manifest["status"] = "partial" if any(case["status"] != "success" for case in manifest["cases"]) else "success"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def upload_raw_artifacts(root: Path, prefix: str, bucket: str, endpoint: str, region: str) -> dict:
    """Upload raw archives concurrently and publish the raw manifest last."""
    manifest_path = root / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    client = create_obs_client(endpoint, region)
    transfer_config = create_transfer_config()
    raw_prefix = f"{prefix.strip('/')}/raw"

    def upload_target(target_record: dict) -> None:
        if "archive" not in target_record:
            return
        try:
            upload_archive(client, transfer_config, root, raw_prefix, bucket, target_record, "raw")
        except Exception as exc:
            target_record["upload_error"] = str(exc)

    target_records = [target for case in manifest["cases"] for target in case["targets"]]
    with ThreadPoolExecutor(max_workers=RAW_UPLOAD_CONCURRENCY) as pool:
        list(pool.map(upload_target, target_records))
    for target_record in target_records:
        if target_record.get("obs_url"):
            print(f"Profiling artifact uploaded: {target_record['obs_url']}", flush=True)
        elif target_record.get("upload_error"):
            print(
                f"Profiling artifact upload failed: {target_record['archive']}: {target_record['upload_error']}",
                flush=True,
            )
    if any(target_record.get("upload_error") for target_record in target_records):
        manifest["status"] = "partial"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest_key = f"{raw_prefix}/profile_manifest.json"
    upload_json(client, manifest_path, bucket, manifest_key)
    logger.info("Profiling manifest: obs://%s/%s", bucket, manifest_key)
    print(f"Profiling manifest uploaded: obs://{bucket}/{manifest_key} (status={manifest['status']})", flush=True)
    return manifest


# =============================================================================
# Command-line entry point
# =============================================================================


def main() -> None:
    """Upload raw profiling artifacts from a benchmark job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["upload"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()
    result = upload_raw_artifacts(args.root, args.prefix, args.bucket, args.endpoint, args.region)
    if result["status"] == "partial":
        raise RuntimeError("Profiling raw artifact/upload incomplete; see manifest and per-target errors above")


if __name__ == "__main__":
    main()
