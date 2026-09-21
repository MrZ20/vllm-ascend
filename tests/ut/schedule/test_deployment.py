# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from tests.e2e.schedule.config import ClusterContext, ResourceSpec
from tests.e2e.schedule.deployment import build_plan
from tests.ut.schedule.test_config import minimal_case, parse


def service(strategy, nodes, port, device):
    launches = {}
    for rank, node in enumerate(nodes):
        launch = {
            "server_cmd": "vllm serve model --port {{ LaunchContext.port }} --host 0.0.0.0",
            "envs": {"COORDINATOR": "{{ ServiceContext.coordinator_ip }}"},
        }
        if strategy == "external_dp_rank":
            launch["expansion"] = {
                "dp_size": len(nodes),
                "dp_size_local": 1,
                "dp_rank_start": rank,
                "port_start": port,
                "dp_rpc_port": port + 100,
                "tp_size": 1,
                "device_ids": [device],
            }
            launch["envs"]["DEVICES"] = "{{ LaunchContext.visible_devices }}"
        else:
            launch["envs"]["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
            launch["endpoint"] = {"host": "{{ NodeContext.ip }}", "port": port} if rank == 0 else None
            if rank:
                launch["server_cmd"] = "vllm serve model --headless"
        launches[f"node{node}"] = launch
    return {"launch_strategy": strategy, "launches": launches}


def pd_case(p_strategy="explicit", d_strategy="external_dp_rank"):
    return {
        "name": "pd",
        "model": "model",
        "test_content": ["completion"],
        "deployment": {
            "services": {"p": service(p_strategy, [0, 1], 18000, 4), "d": service(d_strategy, [1, 2], 18100, 5)},
            "routing": {
                "type": "disaggregated_prefill",
                "groups": {"prefiller": ["p"], "decoder": ["d"]},
                "proxy": {
                    "command": "python proxy.py --prefiller-hosts {{ PDRoutingContext.prefiller_hosts }} "
                    "--prefiller-ports {{ PDRoutingContext.prefiller_ports }} "
                    "--decoder-hosts {{ PDRoutingContext.decoder_hosts }}",
                    "endpoint": {"host": "{{ NodeContext.ip }}", "port": 19000},
                },
            },
        },
    }


def cluster(index, count=3):
    return ClusterContext(index, tuple(f"10.0.0.{i + 1}" for i in range(count)), (4, 5, 6, 7), f"eth{index}")


@pytest.mark.parametrize("p_strategy", ["explicit", "external_dp_rank"])
@pytest.mark.parametrize("d_strategy", ["explicit", "external_dp_rank"])
def test_mixed_topologies_global_endpoints_local_processes(tmp_path, p_strategy, d_strategy):
    case = parse(tmp_path, pd_case(p_strategy, d_strategy), ResourceSpec(3, 4))[0]
    plans = [build_plan(case.deployment, cluster(i), {}) for i in range(3)]
    assert all(plan.service_endpoints == plans[0].service_endpoints for plan in plans)
    assert all(plan.client_endpoint == plans[0].client_endpoint for plan in plans)
    assert all(process.node_id == plan.node.id for plan in plans for process in plan.processes)
    assert len({p.id for plan in plans for p in plan.processes}) == 5
    # D coordinator is node1, not global primary node0.
    assert next(p for p in plans[2].processes if p.owner.name == "d").env_overrides["COORDINATOR"] == "10.0.0.2"
    assert all(dependency.process_id.startswith("routing/") for plan in plans for dependency in plan.dependencies)


def test_internal_dp_and_empty_node(tmp_path):
    value = pd_case()
    value["deployment"] = {"services": {"main": service("explicit", [0, 1], 18000, 4)}}
    case = parse(tmp_path, value, ResourceSpec(3, 4))[0]
    plans = [build_plan(case.deployment, cluster(i), {}) for i in range(3)]
    assert not plans[2].processes
    assert plans[1].processes[0].endpoint is None
    assert plans[2].client_endpoint == plans[0].client_endpoint


def test_single_auto_port_shared_by_command_and_env(tmp_path):
    value = minimal_case()
    value["endpoint"]["port"] = "auto"
    value["envs"] = {"PORT": "{{ LaunchContext.port }}"}
    value["server_cmd"] = "vllm serve model --port $PORT"
    case = parse(tmp_path, value)[0]
    plan = build_plan(case.deployment, cluster(0, 1), {}, port_provider=lambda: 12345)
    assert plan.client_endpoint.port == 12345
    assert plan.processes[0].argv[-1] == "12345"
    assert plan.processes[0].env_overrides["PORT"] == "12345"


def test_epd_two_services_empty_prefill(tmp_path):
    value = pd_case("explicit", "explicit")
    value["deployment"]["services"] = {
        "encode0": service("explicit", [0], 18000, 4),
        "pd0": service("explicit", [0], 18100, 5),
    }
    value["deployment"]["routing"] = {
        "type": "epd",
        "groups": {"encode": ["encode0"], "prefill": [], "decode": ["pd0"]},
        "proxy": {
            "command": "python proxy.py --prefill-servers-urls {{ EPDRoutingContext.prefill_urls }}",
            "endpoint": {"host": "{{ NodeContext.ip }}", "port": 19000},
        },
    }
    plan = build_plan(parse(tmp_path, value, ResourceSpec(1, 4))[0].deployment, cluster(0, 1), {})
    assert len(plan.processes) == 3
    assert plan.processes[-1].argv[-1] == "disable"


def test_2p2d_respects_group_order(tmp_path):
    value = pd_case()
    value["deployment"]["services"] = {
        name: service("explicit", [0], port, device)
        for name, port, device in (("p0", 18000, 4), ("p1", 18001, 5), ("d0", 18002, 6), ("d1", 18003, 7))
    }
    value["deployment"]["routing"]["groups"] = {"prefiller": ["p1", "p0"], "decoder": ["d0", "d1"]}
    plan = build_plan(parse(tmp_path, value, ResourceSpec(1, 4))[0].deployment, cluster(0, 1), {})
    assert len(plan.processes) == 5
    assert plan.processes[-1].argv[6:8] == ("18001", "18000")


def test_external_default_devices_and_fixed_env_conflict(tmp_path):
    value = pd_case()
    for launch in value["deployment"]["services"]["d"]["launches"].values():
        del launch["expansion"]["device_ids"]
    case = parse(tmp_path, value, ResourceSpec(3, 4))[0]
    plan = build_plan(case.deployment, cluster(1), {})
    assert next(p for p in plan.processes if p.owner.name == "d").env_overrides["ASCEND_RT_VISIBLE_DEVICES"] == "5"
    value["deployment"]["services"]["d"]["launches"]["node1"]["envs"]["ASCEND_RT_VISIBLE_DEVICES"] = "4"
    with pytest.raises(ValueError, match="disagree"):
        build_plan(parse(tmp_path, value, ResourceSpec(3, 4))[0].deployment, cluster(1), {})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda v: v["deployment"]["services"]["d"]["launches"]["node2"]["expansion"].update(dp_rank_start=0),
            "overlapping",
        ),
        (
            lambda v: v["deployment"]["services"]["d"]["launches"]["node2"]["expansion"].update(tp_size=2),
            "inconsistent",
        ),
        (
            lambda v: v["deployment"]["services"]["d"]["launches"]["node1"]["expansion"].update(device_ids=[4]),
            "Overlapping",
        ),
        (lambda v: v["deployment"]["routing"]["proxy"]["endpoint"].update(port=18000), "Duplicate HTTP"),
        (lambda v: v["deployment"]["routing"]["proxy"]["endpoint"].update(host="127.0.0.1"), "reachable"),
        (lambda v: v["deployment"]["routing"]["proxy"]["endpoint"].update(port="auto"), "single-node"),
    ],
)
def test_invalid_plans(tmp_path, mutate, message):
    value = copy.deepcopy(pd_case())
    mutate(value)
    case = parse(tmp_path, value, ResourceSpec(3, 4))[0]
    with pytest.raises(ValueError, match=message):
        for i in range(3):
            build_plan(case.deployment, cluster(i), {}, port_provider=lambda: 12000)


def test_remote_endpoint_cannot_read_parent_env(tmp_path):
    value = pd_case()
    launch = value["deployment"]["services"]["p"]["launches"]["node0"]
    launch["endpoint"]["port"] = "$PORT"
    with pytest.raises(ValueError, match="Missing environment"):
        build_plan(parse(tmp_path, value, ResourceSpec(3, 4))[0].deployment, cluster(0), {"PORT": "18000"})
    launch["envs"]["PORT"] = "18000"
    assert (
        build_plan(parse(tmp_path, value, ResourceSpec(3, 4))[0].deployment, cluster(0), {}).client_endpoint.port
        == 19000
    )
