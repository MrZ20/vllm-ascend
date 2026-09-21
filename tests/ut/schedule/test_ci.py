# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from tests.e2e.schedule.config import ClusterContext, ResourceSpec, parse_cases
from tests.e2e.schedule.deployment import build_plan

ROOT = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / ".github/workflows/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_samples_compile_on_every_node():
    resolver = load_script("resolve_schedule_tests_v2")
    matrix = resolver.resolve(ROOT / ".github/workflows/configs/nightly_config_v2.yaml", ROOT, "a3")
    for entries in matrix.values():
        for entry in entries:
            resources = ResourceSpec(entry["num_nodes"], int(entry["npu_per_node"]))
            cases = parse_cases(entry["config_path"], resources, ROOT)
            for case in cases:
                plans = []
                for index in range(resources.num_nodes):
                    context = ClusterContext(
                        index,
                        tuple(f"10.0.0.{i + 1}" for i in range(resources.num_nodes)),
                        tuple(range(resources.npu_per_node)),
                        "eth0",
                    )
                    ports = iter(range(19000, 19020))
                    plans.append(
                        build_plan(
                            case.deployment,
                            context,
                            {"REPO_ROOT": str(ROOT), "RUN_ID": "ut"},
                            port_provider=ports.__next__,
                        )
                    )
                assert all(plan.service_endpoints == plans[0].service_endpoints for plan in plans)


@pytest.mark.parametrize("value", [0, -1, True, "0", "two"])
def test_matrix_rejects_invalid_resource(value, tmp_path):
    resolver = load_script("resolve_schedule_tests_v2")
    (tmp_path / "model.yaml").write_text("schema_version: 2")
    matrix = {
        "a3": {
            "single_node": {
                "test_config": [
                    dict(name="sample", config_path="model.yaml", num_nodes=1, npu_per_node=value, os="runner")
                ]
            }
        }
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(matrix))
    with pytest.raises(ValueError, match="positive integer"):
        resolver.resolve(path, tmp_path, "a3")


def test_matrix_selection_and_omission(tmp_path):
    resolver = load_script("resolve_schedule_tests_v2")
    (tmp_path / "model.yaml").write_text("schema_version: 2")
    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "a3": {
                    "single_node": {
                        "test_config": [dict(name="sample", config_path="model.yaml", num_nodes=1, os="runner")]
                    }
                }
            }
        )
    )
    assert resolver.resolve(path, tmp_path, "a3", "sample")["single_node"][0]["npu_per_node"] == ""
    with pytest.raises(ValueError, match="Unknown test names"):
        resolver.resolve(path, tmp_path, "a3", "missing")


def test_lws_resources_placement_and_environment():
    controller = load_script("run_schedule_lws_v2")
    output = yaml.safe_load(
        controller.render(
            ROOT,
            name="s2-unit",
            namespace="vllm-project",
            image="registry/image@sha256:" + "a" * 64,
            num_nodes=2,
            npu_per_node=2,
            config_path="model.yaml",
            sha="b" * 40,
            log_root="/root/.cache/schedule-v2",
            pvc="shared-pvc",
        )
    )
    group = output["spec"]["leaderWorkerTemplate"]
    assert group["size"] == 2 and output["spec"]["replicas"] == 1
    for role in ("leaderTemplate", "workerTemplate"):
        spec = group[role]["spec"]
        assert spec["hostPID"] is False and spec["shareProcessNamespace"] is False
        container = spec["containers"][0]
        assert container["resources"]["requests"]["huawei.com/ascend-1980"] == 2
        assert container["resources"]["limits"]["huawei.com/ascend-1980"] == 2
        assert spec["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
        env = {e["name"]: e["value"] for e in container["env"]}
        assert env["SCHEDULE_CLEANUP_PROCESSES"] == "0"
        assert env["NPU_PER_NODE"] == "2" and env["NUM_NODES"] == "2"
        assert not {"LWS_WORKER_INDEX", "LWS_GROUP_SIZE", "CONFIG_BASE_PATH"} & env.keys()
        assert spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "shared-pvc"


def pod(index, code=0):
    return {
        "metadata": {"name": f"pod-{index}", "labels": {"leaderworkerset.sigs.k8s.io/worker-index": str(index)}},
        "spec": {"nodeName": f"host-{index}"},
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "schedule",
                    "restartCount": 0,
                    "state": {"terminated": {"exitCode": code, "reason": "Completed"}},
                }
            ],
        },
    }


def test_controller_handles_fast_exit_failures_and_placement():
    controller = load_script("run_schedule_lws_v2")
    assert controller.pod_completion([pod(0), pod(1)], 2)
    assert not controller.pod_completion([pod(0)], 2)
    failed = pod(1, 137)
    failed["status"]["containerStatuses"][0]["state"]["terminated"]["reason"] = "OOMKilled"
    with pytest.raises(RuntimeError, match="Container failure"):
        controller.pod_completion([pod(0), failed], 2)
    failed = pod(1)
    failed["spec"]["nodeName"] = "host-0"
    with pytest.raises(RuntimeError, match="same host"):
        controller.pod_completion([pod(0), failed], 2)


def test_controller_requires_final_and_test_artifacts(tmp_path):
    controller = load_script("run_schedule_lws_v2")
    run = tmp_path / "run-id"
    metadata = dict(source_commit="abc", config_digest="digest", case_names=["example"], resources={"num_nodes": 2})
    for i in range(2):
        (run / f"node-{i}").mkdir(parents=True)
        (run / f"node-{i}/run_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="Missing final"):
        controller.verify_results(run, 2, "abc")
    case = run / "cases/example"
    case.mkdir(parents=True)
    (case / "final.json").write_text(json.dumps(dict(status="passed", config_digest="digest")))
    for i in range(2):
        (case / f"node-{i}.result.json").write_text(json.dumps(dict(status="passed", run_id="run-id")))
    with pytest.raises(FileNotFoundError):
        controller.verify_results(run, 2, "abc")
    (run / "benchmark_results").mkdir()
    (run / "benchmark_results/example.json").write_text(json.dumps(dict(status="passed")))
    controller.verify_results(run, 2, "abc")


def test_v2_workflows_are_independent_dispatch_only():
    workflow = yaml.safe_load((ROOT / ".github/workflows/schedule_nightly_test_a3_v2.yaml").read_text())
    assert set(workflow["on"]) == {"workflow_dispatch"}
    jobs = workflow["jobs"]
    assert jobs["double_node"]["needs"] == ["matrix", "multi_node"]
    assert jobs["single_node"]["needs"] == ["matrix", "double_node"]
    assert [jobs[k]["strategy"]["max-parallel"] for k in ("multi_node", "double_node", "single_node")] == [2, 4, 9]
    assert all(job.get("uses", "").endswith("_v2.yaml") for name, job in jobs.items() if name != "matrix")
