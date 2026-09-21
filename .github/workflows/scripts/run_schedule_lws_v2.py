# SPDX-License-Identifier: Apache-2.0
"""Render/run one V2 LWS; require terminated containers and shared case results."""

import argparse
import json
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from jinja2 import Environment, StrictUndefined


def positive(value):
    if not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError(f"Expected a positive integer: {value}")
    return int(value)


def render(
    repo: Path,
    *,
    name,
    namespace,
    image,
    num_nodes,
    npu_per_node,
    config_path,
    sha,
    log_root,
    pvc,
    cpu="16",
    memory="128Gi",
    npu_resource_key="huawei.com/ascend-1980",
) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", name):
        raise ValueError("Use an LWS name of at most 31 lowercase alphanumeric/dash characters")
    env = {
        "CONFIG_YAML_PATH": config_path,
        "REPO_ROOT": "/vllm-workspace/vllm-ascend",
        "NUM_NODES": positive(num_nodes),
        "NPU_PER_NODE": positive(npu_per_node),
        "RUN_ID": name,
        "LOG_PREFIX": str(log_root),
        "COORD_DIR": str(log_root),
        "EXPECTED_SOURCE_SHA": sha,
        "SCHEDULE_IMAGE": image,
        "SCHEDULE_CLEANUP_PROCESSES": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "VLLM_USE_MODELSCOPE": "True",
        "BENCHMARK_JOB_NAME": name,
    }
    template = repo / "tests/e2e/schedule/templates/lws.yaml.jinja2"
    return (
        Environment(undefined=StrictUndefined, autoescape=False)
        .from_string(template.read_text())
        .render(
            name=name,
            namespace=namespace,
            image=image,
            num_nodes=positive(num_nodes),
            npu_per_node=positive(npu_per_node),
            pvc=pvc,
            env=env,
            cpu=cpu,
            memory=memory,
            npu_resource_key=npu_resource_key,
        )
    )


def pod_completion(pods: list[dict], count: int) -> bool:
    """Accept fast termination, reject failures/restarts and require unique hosts."""
    if len(pods) > count:
        raise RuntimeError("Unexpected number of Pods in this run")
    hosts, completed, indices = set(), 0, set()
    for pod in pods:
        status, spec = pod.get("status", {}), pod["spec"]
        index = pod["metadata"]["labels"].get("leaderworkerset.sigs.k8s.io/worker-index")
        if index in indices or (index is not None and int(index) not in range(count)):
            raise RuntimeError("Unexpected or repeated LWS worker index")
        indices.add(index)
        if spec.get("nodeName"):
            if spec["nodeName"] in hosts:
                raise RuntimeError("Multinode Pods were placed on the same host")
            hosts.add(spec["nodeName"])
        if status.get("phase") == "Failed":
            raise RuntimeError(f"Pod {pod['metadata']['name']} failed: {status.get('reason')}")
        for container in status.get("initContainerStatuses", []) + status.get("containerStatuses", []):
            state = container.get("state", {})
            terminated = state.get("terminated")
            if container.get("restartCount", 0) or (terminated and terminated["exitCode"] != 0):
                raise RuntimeError(f"Container failure: {container}")
            if state.get("waiting", {}).get("reason") in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"):
                raise RuntimeError(f"Container preparation failed: {state}")
            if container["name"] == "schedule" and terminated:
                completed += 1
    return completed == count and len(hosts) == count and indices == {str(i) for i in range(count)}


def verify_results(run_dir: Path, count: int, sha: str):
    metadata = [json.loads((run_dir / f"node-{i}" / "run_metadata.json").read_text()) for i in range(count)]
    primary = metadata[0]
    for item in metadata:
        if item["source_commit"] != sha or any(
            item[k] != primary[k] for k in ("config_digest", "case_names", "resources")
        ):
            raise RuntimeError("Node metadata disagrees with the requested run")
    finals = sorted((run_dir / "cases").glob("*/final.json"))
    if len(finals) != len(primary["case_names"]):
        raise RuntimeError("Missing final case results")
    for path in finals:
        final = json.loads(path.read_text())
        if final["status"] != "passed" or final["config_digest"] != primary["config_digest"]:
            raise RuntimeError(f"Case failed or has inconsistent config: {path}")
        for index in range(count):
            result = json.loads((path.parent / f"node-{index}.result.json").read_text())
            if result["status"] != "passed" or result["run_id"] != run_dir.name:
                raise RuntimeError(f"Node {index} result failed: {result}")
        result_file = run_dir / "benchmark_results" / f"{path.parent.name}.json"
        if json.loads(result_file.read_text())["status"] != "passed":
            raise RuntimeError(f"Missing/failed test artifact: {result_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("name", "image", "config-path", "sha"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--namespace", default="vllm-project")
    parser.add_argument("--num-nodes", type=positive, required=True)
    parser.add_argument("--npu-per-node", type=positive, required=True)
    parser.add_argument("--log-root", type=Path, default=Path("/root/.cache/schedule-v2"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--pvc", default="nv-action-vllm-benchmarks-v2")
    parser.add_argument("--cpu", default="16")
    parser.add_argument("--memory", default="128Gi")
    parser.add_argument("--npu-resource-key", default="huawei.com/ascend-1980")
    parser.add_argument("--timeout", type=positive, default=7200)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.artifact_dir / "lws.yaml"
    manifest.write_text(
        render(
            args.repo,
            **{k: v for k, v in vars(args).items() if k not in {"repo", "artifact_dir", "timeout", "render_only"}},
        )
    )
    if args.render_only:
        return
    run_dir = args.log_root / args.name
    if run_dir.exists():
        raise RuntimeError("Run directory already exists; use a fresh name")
    run_dir.mkdir(parents=True)

    def kubectl(*command, check=True):
        return subprocess.run(
            ["kubectl", "--request-timeout=30s", "-n", args.namespace, *command],
            capture_output=True,
            text=True,
            check=check,
            timeout=90,
        )

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    error, created = None, False
    selector = f"schedule-v2-run={args.name}"
    try:
        claim = json.loads(kubectl("get", "pvc", args.pvc, "-o", "json").stdout)
        if claim.get("status", {}).get("phase") != "Bound":
            raise RuntimeError("The shared result PVC is not bound")
        # create (not apply) prevents taking ownership of a pre-existing run.
        kubectl("create", "-f", str(manifest))
        created = True
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            document = json.loads(kubectl("get", "pods", "-l", selector, "-o", "json").stdout)
            (args.artifact_dir / "pods.json").write_text(json.dumps(document, indent=2))
            if pod_completion(document["items"], args.num_nodes):
                verify_results(run_dir, args.num_nodes, args.sha)
                break
            time.sleep(10)
        else:
            raise TimeoutError("LWS did not finish within its execution budget")
    except BaseException as exc:
        error = exc
    finally:
        if created:
            try:
                pods = json.loads(kubectl("get", "pods", "-l", selector, "-o", "json").stdout)["items"]
                for pod in pods:
                    name = pod["metadata"]["name"]
                    log = kubectl("logs", name, "--all-containers=true", check=False)
                    (args.artifact_dir / f"{name}.log").write_text(log.stdout + log.stderr)
                describe = kubectl("describe", "lws", args.name, check=False)
                (args.artifact_dir / "describe.txt").write_text(describe.stdout + describe.stderr)
            except Exception as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"Collecting diagnostics failed: {exc}")
            try:
                kubectl("delete", "lws", args.name, "--wait=false")
                kubectl("wait", "--for=delete", "pod", "-l", selector, "--timeout=60s")
            except Exception as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"LWS cleanup failed: {exc}")
        try:
            if run_dir.exists():
                shutil.copytree(run_dir, args.artifact_dir / "run", dirs_exist_ok=True)
        except Exception as exc:
            if error is None:
                error = exc
            else:
                error.add_note(f"Copying run artifacts failed: {exc}")
        finally:
            signal.signal(signal.SIGTERM, previous)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
