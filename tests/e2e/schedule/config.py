# SPDX-License-Identifier: Apache-2.0
"""CPU-only source schema and records for scheduled model tests.

The public Context field names below are the YAML template API. Importing this
module does not inspect the environment, import vLLM, or initialize a device.
"""

import copy
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from tests.e2e.common.kv_pool.config import KVPoolConfig, parse_kv_pool_config

PRIMARY_NODE_INDEX = 0
DEFAULT_PROXY_NODE_INDEX = PRIMARY_NODE_INDEX
DEFAULT_TEST_OWNER_INDEX = PRIMARY_NODE_INDEX
START_TIMEOUT = 600.0
TEST_TIMEOUT = 1800.0
MIN_NPU_FREE_RATIO = 0.90
NPU_RELEASE_WAIT_SECONDS = 30
NPU_QUERY_TIMEOUT_SECONDS = 60


# Resources and normalized source records.
def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ResourceSpec:
    num_nodes: int
    npu_per_node: int

    def __post_init__(self):
        for name in ("num_nodes", "npu_per_node"):
            object.__setattr__(self, name, positive_int(getattr(self, name), name))


@dataclass(frozen=True)
class NodeSpec:
    id: str
    index: int


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    scheme: str = "http"

    def __post_init__(self):
        if not self.host or self.scheme not in ("http", "https") or not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid serving endpoint: {self}")

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}"


@dataclass(frozen=True)
class EndpointTemplate:
    host: str
    port: str
    scheme: str = "http"


@dataclass(frozen=True)
class ExternalDPRankExpansionSpec:
    dp_size: int
    dp_size_local: int
    dp_rank_start: int
    port_start: int
    dp_rpc_port: int
    tp_size: int
    pp_size: int = 1
    cp_size: int = 1
    sp_size: int = 1
    device_ids: tuple[int, ...] | None = None

    @property
    def devices_per_rank(self) -> int:
        return self.tp_size * self.pp_size * self.cp_size * self.sp_size


@dataclass(frozen=True)
class LaunchSpec:
    node: str
    command_template: str
    env_template: dict[str, str]
    endpoint_template: EndpointTemplate | None
    expansion: ExternalDPRankExpansionSpec | None = None
    cwd: str | None = None


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    launch_strategy: str
    launches: tuple[LaunchSpec, ...]
    coordinator_node: str | None = None


@dataclass(frozen=True)
class ProxySpec:
    node: str
    command_template: str
    env_template: dict[str, str]
    endpoint_template: EndpointTemplate
    cwd: str | None = None


@dataclass(frozen=True)
class RoutingSpec:
    type: str
    groups: dict[str, tuple[str, ...]]
    proxy: ProxySpec


@dataclass(frozen=True)
class DeploymentSpec:
    name: str
    resources: ResourceSpec
    nodes: tuple[NodeSpec, ...]
    services: tuple[ServiceSpec, ...]
    routing: RoutingSpec | None = None
    infrastructure: KVPoolConfig | None = None


@dataclass(frozen=True)
class TestSpec:
    model: str
    test_content: tuple[str, ...]
    prompts: list[Any] = field(default_factory=list)
    api_keyword_args: dict[str, Any] | list[dict[str, Any]] = field(default_factory=dict)
    expected_response: dict[str, Any] = field(default_factory=dict)
    mm_request: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)
    acceptance_rate: dict[str, Any] = field(default_factory=dict)
    benchmark_comparisons: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    config_path: str
    deployment: DeploymentSpec
    test: TestSpec
    special_dependencies: dict[str, str] = field(default_factory=dict)


# Context API. ClusterContext is an input, never an implicit YAML namespace.
@dataclass(frozen=True)
class ClusterContext:
    current_node_index: int
    node_ips: tuple[str, ...]
    current_device_pool: tuple[int, ...]
    current_nic_name: str | None = None


@dataclass(frozen=True)
class NodeContext:
    id: str
    index: int
    ip: str
    npu_per_node: int
    nic_name: str | None = None


@dataclass(frozen=True)
class ServiceContext:
    name: str
    coordinator_node_id: str
    coordinator_ip: str


@dataclass(frozen=True)
class LaunchContext:
    port: int | None = None
    local_rank: int | None = None
    dp_rank: int | None = None
    visible_devices: str | None = None


@dataclass(frozen=True)
class PDRoutingContext:
    prefiller_endpoints: tuple[Endpoint, ...]
    decoder_endpoints: tuple[Endpoint, ...]
    proxy_endpoint: Endpoint

    @property
    def prefiller_hosts(self) -> str:
        return " ".join(e.host for e in self.prefiller_endpoints)

    @property
    def prefiller_ports(self) -> str:
        return " ".join(str(e.port) for e in self.prefiller_endpoints)

    @property
    def decoder_hosts(self) -> str:
        return " ".join(e.host for e in self.decoder_endpoints)

    @property
    def decoder_ports(self) -> str:
        return " ".join(str(e.port) for e in self.decoder_endpoints)


@dataclass(frozen=True)
class EPDRoutingContext:
    encode_endpoints: tuple[Endpoint, ...]
    prefill_endpoints: tuple[Endpoint, ...]
    decode_endpoints: tuple[Endpoint, ...]
    proxy_endpoint: Endpoint

    @property
    def encode_urls(self) -> str:
        return " ".join(e.url for e in self.encode_endpoints)

    @property
    def prefill_urls(self) -> str:
        return " ".join(e.url for e in self.prefill_endpoints) or "disable"

    @property
    def decode_urls(self) -> str:
        return " ".join(e.url for e in self.decode_endpoints)


CONTEXT_FIELDS = MappingProxyType(
    {
        cls.__name__: frozenset(f.name for f in fields(cls))
        | frozenset(name for name, value in vars(cls).items() if isinstance(value, property))
        for cls in (NodeContext, ServiceContext, LaunchContext, PDRoutingContext, EPDRoutingContext)
    }
)


# Fully materialized local launch records.
@dataclass(frozen=True)
class ProcessOwner:
    type: str
    name: str


@dataclass(frozen=True)
class ProcessSpec:
    id: str
    node_id: str
    kind: str
    owner: ProcessOwner
    argv: tuple[str, ...]
    env_overrides: dict[str, str]
    cwd: str | None
    endpoint: Endpoint | None


@dataclass(frozen=True)
class DependencySpec:
    process_id: str
    endpoints: tuple[Endpoint, ...]


@dataclass(frozen=True)
class LaunchPlan:
    node: NodeContext
    processes: tuple[ProcessSpec, ...]
    service_endpoints: dict[str, tuple[Endpoint, ...]]
    client_endpoint: Endpoint
    dependencies: tuple[DependencySpec, ...]


# YAML loading: reject duplicate explicit keys but preserve native merge overrides.
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key, _ in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                continue
            name = self.construct_object(key, deep=deep)
            if name in seen:
                raise ValueError(f"Duplicate YAML key {name!r} at line {key.start_mark.line + 1}")
            seen.add(name)
        self.flatten_mapping(node)
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _mapping(value: Any, allowed: set[str], path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected a nonempty string")
    return value


def scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"Expected an env scalar, got {value!r}")
    return str(value)


def _envs(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an environment mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{path}: invalid environment key {key!r}")
        result[key] = scalar_text(item)
    return result


def _endpoint(value: Any, path: str) -> EndpointTemplate | None:
    if value is None:
        return None
    raw = _mapping(value, {"host", "port", "scheme"}, path)
    return EndpointTemplate(_text(raw["host"], path + ".host"), scalar_text(raw["port"]), raw.get("scheme", "http"))


def _node(value: str, nodes: tuple[NodeSpec, ...]) -> str:
    if value not in {node.id for node in nodes}:
        raise ValueError(f"Unknown launch node {value!r}; allocation has {len(nodes)} nodes")
    return value


def _expansion(raw: Any, path: str) -> ExternalDPRankExpansionSpec:
    raw = dict(_mapping(raw, {f.name for f in fields(ExternalDPRankExpansionSpec)}, path))
    for name in ("dp_size", "dp_size_local", "port_start", "dp_rpc_port", "tp_size", "pp_size", "cp_size", "sp_size"):
        if name in raw:
            raw[name] = positive_int(raw[name], path + "." + name)
    rank = raw.get("dp_rank_start")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"{path}.dp_rank_start must be a nonnegative integer")
    if "device_ids" in raw:
        ids = raw["device_ids"]
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{path}.device_ids must contain distinct nonnegative integers")
        raw["device_ids"] = tuple(ids)
    return ExternalDPRankExpansionSpec(**raw)


def _launch(raw: Any, node: str, strategy: str, path: str) -> LaunchSpec:
    raw = _mapping(raw, {"server_cmd", "server_cmd_extra", "envs", "endpoint", "cwd", "expansion"}, path)
    command = _text(raw["server_cmd"], path + ".server_cmd")
    if "server_cmd_extra" in raw:
        command += " " + _text(raw["server_cmd_extra"], path + ".server_cmd_extra")
    expansion = _expansion(raw["expansion"], path + ".expansion") if "expansion" in raw else None
    if (strategy == "external_dp_rank") != (expansion is not None):
        raise ValueError(f"{path}: expansion is required only for external_dp_rank")
    return LaunchSpec(
        node, command, _envs(raw.get("envs", {}), path), _endpoint(raw.get("endpoint"), path), expansion, raw.get("cwd")
    )


def _deployment(raw: dict, name: str, resources: ResourceSpec) -> DeploymentSpec:
    nodes = tuple(NodeSpec(f"node{i}", i) for i in range(resources.num_nodes))
    if "server_cmd" in raw:
        if "deployment" in raw:
            raise ValueError("server_cmd and deployment are mutually exclusive")
        launch = _launch(
            {k: raw[k] for k in ("server_cmd", "server_cmd_extra", "envs", "endpoint", "cwd") if k in raw},
            "node0",
            "explicit",
            name,
        )
        if launch.endpoint_template is None:
            raise ValueError("Single-service shorthand requires endpoint")
        services = (ServiceSpec("main", "explicit", (launch,)),)
        routing = None
    else:
        if set(raw) & {"server_cmd_extra", "envs", "endpoint", "cwd"}:
            raise ValueError("Shorthand launch fields require server_cmd")
        deploy = _mapping(raw["deployment"], {"services", "routing"}, name + ".deployment")
        service_map = deploy["services"]
        if not isinstance(service_map, dict) or not service_map:
            raise ValueError("deployment.services must be a nonempty mapping")
        parsed = []
        for service_name, value in service_map.items():
            path = f"{name}.deployment.services.{service_name}"
            value = _mapping(value, {"launch_strategy", "coordinator_node", "launches"}, path)
            strategy = value["launch_strategy"]
            if strategy not in ("explicit", "external_dp_rank"):
                raise ValueError(f"{path}: unsupported launch_strategy {strategy!r}")
            launches = value["launches"]
            if not isinstance(launches, dict) or not launches:
                raise ValueError(f"{path}.launches must be a nonempty mapping")
            entries = tuple(_launch(v, _node(k, nodes), strategy, f"{path}.{k}") for k, v in launches.items())
            coordinator = value.get("coordinator_node")
            if coordinator is not None and coordinator not in launches:
                raise ValueError(f"{path}: coordinator must belong to this Service")
            parsed.append(ServiceSpec(_text(service_name, path), strategy, entries, coordinator))
        services = tuple(parsed)
        routing = _routing(deploy.get("routing"), services, nodes)
    return DeploymentSpec(name, resources, nodes, services, routing, parse_kv_pool_config(raw.get("kv_pool")))


def _routing(raw: Any, services: tuple[ServiceSpec, ...], nodes: tuple[NodeSpec, ...]) -> RoutingSpec | None:
    if raw is None:
        return None
    raw = _mapping(raw, {"type", "groups", "proxy"}, "routing")
    required = {"disaggregated_prefill": ("prefiller", "decoder"), "epd": ("encode", "prefill", "decode")}
    if raw["type"] not in required:
        raise ValueError(f"Unsupported routing.type {raw['type']!r}")
    groups = _mapping(raw["groups"], set(required[raw["type"]]), "routing.groups")
    seen = set()
    names = {s.name for s in services}
    parsed = {}
    for name in required[raw["type"]]:
        values = groups.get(name, [])
        if not isinstance(values, list) or (not values and name != "prefill"):
            raise ValueError(f"routing.groups.{name} requires a nonempty service list")
        for service in values:
            if service not in names or service in seen:
                raise ValueError(f"Unknown or repeated routing Service {service!r}")
            seen.add(service)
        parsed[name] = tuple(values)
    proxy = _mapping(raw["proxy"], {"node", "command", "envs", "endpoint", "cwd"}, "routing.proxy")
    endpoint = _endpoint(proxy["endpoint"], "routing.proxy.endpoint")
    if endpoint is None:
        raise ValueError("routing.proxy.endpoint is required")
    return RoutingSpec(
        raw["type"],
        parsed,
        ProxySpec(
            _node(proxy.get("node", f"node{DEFAULT_PROXY_NODE_INDEX}"), nodes),
            _text(proxy["command"], "routing.proxy.command"),
            _envs(proxy.get("envs", {}), "routing.proxy.envs"),
            endpoint,
            proxy.get("cwd"),
        ),
    )


def parse_cases(config_path: str | Path, resources: ResourceSpec, repo_root: str | Path) -> tuple[CaseSpec, ...]:
    path = Path(config_path)
    path = (Path(repo_root) / path).resolve() if not path.is_absolute() else path.resolve()
    raw = load_yaml(path)
    if "schema_version" in raw and (type(raw["schema_version"]) is not int or raw["schema_version"] != 2):
        raise ValueError(f"{path}: schema_version, when provided, must be 2")
    root = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    if "test_cases" in root:
        _mapping(root, {"schema_version", "test_cases"}, str(path))
        cases = root["test_cases"]
    else:
        cases = [{k: v for k, v in root.items() if k != "schema_version"}]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: test_cases must be a nonempty list")
    test_fields = {f.name for f in fields(TestSpec)}
    allowed = test_fields | {
        "name",
        "server_cmd",
        "server_cmd_extra",
        "envs",
        "endpoint",
        "cwd",
        "deployment",
        "kv_pool",
        "special_dependencies",
    }
    supported = {"completion", "chat_completion", "image", "spec_decode_acceptance", "benchmark_comparisons"}
    result = []
    names = set()
    for value in cases:
        try:
            value = copy.deepcopy(_mapping(value, allowed, str(path)))
            name = _text(value["name"], "case.name")
            if name in names:
                raise ValueError(f"Duplicate case name {name!r}")
            names.add(name)
            model = _text(value["model"], "case.model")
            content = value.get("test_content") or []
            if not isinstance(content, list) or set(content) - supported:
                raise ValueError(f"Unsupported test_content {content!r}")
            test = TestSpec(
                **{**{k: value[k] for k in test_fields if k in value}, "model": model, "test_content": tuple(content)}
            )
            if not content and not test.benchmarks:
                raise ValueError("Case must request test_content or benchmarks")
            if not isinstance(test.benchmarks, dict):
                raise ValueError("benchmarks must be a mapping")
            dependencies = value.get("special_dependencies", {})
            if not isinstance(dependencies, dict) or any(
                not isinstance(package, str) or not isinstance(version, str) or not version
                for package, version in dependencies.items()
            ):
                raise ValueError("special_dependencies must map package names to exact version strings")
            result.append(
                CaseSpec(
                    name,
                    str(path),
                    _deployment(value, name, resources),
                    test,
                    dependencies,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: case {value.get('name', '<unknown>') if isinstance(value, dict) else '<invalid>'}: {exc}"
            ) from exc
    return tuple(result)
