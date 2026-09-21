# SPDX-License-Identifier: Apache-2.0
"""Pure expansion, template materialization and Service-based routing."""

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from tests.e2e.schedule.config import (
    CONTEXT_FIELDS,
    ClusterContext,
    DependencySpec,
    DeploymentSpec,
    Endpoint,
    EndpointTemplate,
    EPDRoutingContext,
    LaunchContext,
    LaunchPlan,
    LaunchSpec,
    NodeContext,
    PDRoutingContext,
    ProcessOwner,
    ProcessSpec,
    ServiceContext,
)

CONTEXT_REFERENCE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _context_value(match: re.Match, contexts: dict[str, Any]) -> str:
    namespace, name = match.groups()
    if name not in CONTEXT_FIELDS.get(namespace, ()) or namespace not in contexts:
        raise ValueError(f"Unknown or unavailable Context reference {match.group()}")
    value = getattr(contexts[namespace], name)
    if value is None:
        raise ValueError(f"Context reference has no value: {match.group()}")
    if isinstance(value, Endpoint):
        return value.url
    if isinstance(value, tuple) and all(isinstance(e, Endpoint) for e in value):
        return " ".join(e.url for e in value)
    return str(value)


def _references(template: str, resolve_env: Callable[[str], str], contexts: dict[str, Any], *, command: bool):
    """Protect substitutions before shlex so substituted data never becomes syntax."""
    output, substitutions = [], {}
    quote = None
    index = 0
    while index < len(template):
        char = template[index]
        if char == "\\" and index + 1 < len(template) and quote != "'":
            if command:
                # Let shlex handle command escaping after protecting values.
                output.append(template[index : index + 2])
                index += 2
            elif template[index + 1] == "$":
                output.append("$")
                index += 2
            else:
                # Env scalars are data, including JSON backslashes.
                output.append(char)
                index += 1
            continue
        if command and char in "\"'":
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            output.append(char)
            index += 1
            continue
        match = CONTEXT_REFERENCE.match(template, index)
        if match:
            value = _context_value(match, contexts)
        elif template.startswith("{{", index):
            raise ValueError(f"Invalid Context reference near {template[index:]!r}")
        else:
            match = ENV_REFERENCE.match(template, index) if quote != "'" else None
            value = resolve_env(match.group(1) or match.group(2)) if match else None
        if match:
            marker = f"\ue000{len(substitutions)}\ue001"
            substitutions[marker] = (value, quote is not None)
            output.append(marker)
            index = match.end()
            continue
        if command and quote is None and char in "|&;<>()`":
            raise ValueError(f"Shell operator {char!r} is unsupported; provide a single argv command")
        if command and char == "`" and quote != "'":
            raise ValueError("Command substitution is unsupported")
        if command and template.startswith("$(", index) and quote != "'":
            raise ValueError("Command substitution is unsupported")
        output.append(char)
        index += 1
    return "".join(output), substitutions


def render_scalar(template: str, env: dict[str, str], contexts: dict[str, Any]) -> str:
    def lookup(name):
        if name not in env:
            raise ValueError(f"Missing environment variable ${name}")
        return env[name]

    protected, values = _references(template, lookup, contexts, command=False)
    return re.sub(r"\ue000\d+\ue001", lambda m: values[m.group()][0], protected)


def materialize_env(templates: dict[str, str], inherited: dict[str, str], contexts: dict[str, Any]) -> dict[str, str]:
    resolved, visiting = {}, set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(f"Environment reference cycle involving {name}")
        if name not in templates:
            if name not in inherited:
                raise ValueError(f"Missing environment variable ${name}")
            return inherited[name]
        visiting.add(name)

        def lookup(other):
            if other == name:
                if name not in inherited:
                    raise ValueError(f"Self-reference ${name} has no inherited value")
                return inherited[name]
            return resolve(other)

        protected, values = _references(templates[name], lookup, contexts, command=False)
        resolved[name] = re.sub(r"\ue000\d+\ue001", lambda m: values[m.group()][0], protected)
        visiting.remove(name)
        return resolved[name]

    for name in templates:
        resolve(name)
    return resolved


def materialize_command(template: str, env: dict[str, str], contexts: dict[str, Any]) -> tuple[str, ...]:
    def lookup(name):
        if name not in env:
            raise ValueError(f"Missing environment variable ${name}")
        return env[name]

    protected, values = _references(template, lookup, contexts, command=True)
    result = []
    for token in shlex.split(protected):
        if token in values and not values[token][1]:
            result.extend(values[token][0].split())
            continue

        def substitute(match):
            value, quoted = values[match.group()]
            if not quoted and len(value.split()) > 1:
                raise ValueError("Embedded unquoted multiword value; quote it or use a standalone list token")
            return value

        result.append(re.sub(r"\ue000\d+\ue001", substitute, token))
    if not result:
        raise ValueError("Empty process command")
    return tuple(result)


def parse_device_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ids or any(i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Expected distinct nonnegative visible devices, got {value!r}")
    return ids


@dataclass(frozen=True)
class _PlannedProcess:
    id: str
    service: str
    launch: LaunchSpec
    contexts: dict[str, Any]
    endpoint: Endpoint | None


def _endpoint_env(template: EndpointTemplate, envs: dict[str, str]) -> dict[str, str]:
    """Select only env dependencies of an endpoint, excluding unrelated local NICs."""
    selected = {}
    pending = [
        m.group(1) or m.group(2) for value in (template.host, template.port) for m in ENV_REFERENCE.finditer(value)
    ]
    while pending:
        name = pending.pop()
        if name in selected or name not in envs:
            continue
        selected[name] = envs[name]
        pending.extend(m.group(1) or m.group(2) for m in ENV_REFERENCE.finditer(envs[name]))
    return selected


def build_plan(
    deployment: DeploymentSpec,
    cluster: ClusterContext,
    inherited_env: dict[str, str],
    *,
    port_provider: Callable[[], int] | None = None,
) -> LaunchPlan:
    """Plan globally known endpoints, then materialize only this node's commands.

    Services expand without P/D role assumptions. Only routing consumes roles.
    Distributed peers start together; routing alone depends on backend readiness.
    """
    resources = deployment.resources
    if len(cluster.node_ips) != resources.num_nodes or not 0 <= cluster.current_node_index < resources.num_nodes:
        raise ValueError("Cluster hosts/index disagree with ResourceSpec")
    if (
        len(cluster.current_device_pool) != resources.npu_per_node
        or len(set(cluster.current_device_pool)) != resources.npu_per_node
    ):
        raise ValueError("Current device pool disagrees with ResourceSpec")
    nodes = {
        node.id: NodeContext(
            node.id,
            node.index,
            cluster.node_ips[node.index],
            resources.npu_per_node,
            cluster.current_nic_name if node.index == cluster.current_node_index else None,
        )
        for node in deployment.nodes
    }
    current = nodes[f"node{cluster.current_node_index}"]
    endpoints, planned, bindings = {}, [], set()
    auto_ports = set()

    def endpoint_for(template, envs, contexts, rank_port=None):
        if template is None:
            return None, contexts
        launch_context = contexts["LaunchContext"]
        if template.port == "auto":
            if resources.num_nodes != 1 or port_provider is None:
                raise ValueError("auto endpoint ports require a single-node allocation and port provider")
            port = port_provider()
            if port in auto_ports:
                raise ValueError(f"Port provider returned duplicate auto port {port}")
            auto_ports.add(port)
            contexts = {**contexts, "LaunchContext": replace(launch_context, port=port)}
        global_env = inherited_env if resources.num_nodes == 1 else {}
        selected = materialize_env(_endpoint_env(template, envs), global_env, contexts)
        env = {**global_env, **selected}
        port = (
            contexts["LaunchContext"].port
            if template.port == "auto"
            else int(render_scalar(template.port, env, contexts))
        )
        host = render_scalar(template.host, env, contexts)
        if resources.num_nodes > 1 and host in ("127.0.0.1", "localhost", "0.0.0.0", "::", "::1"):
            raise ValueError(f"Multinode endpoint must be reachable, got {host}")
        endpoint = Endpoint(host, port, template.scheme)
        if rank_port is not None and port != rank_port:
            raise ValueError("External endpoint conflicts with expanded rank port")
        binding = (contexts["NodeContext"].id, port)
        if binding in bindings:
            raise ValueError(f"Duplicate HTTP port on {binding[0]}: {port}")
        bindings.add(binding)
        return endpoint, {**contexts, "LaunchContext": replace(contexts["LaunchContext"], port=port)}

    for service in deployment.services:
        coordinator = service.coordinator_node or min(
            (launch.node for launch in service.launches), key=lambda name: nodes[name].index
        )
        service_context = ServiceContext(service.name, coordinator, nodes[coordinator].ip)
        rank_set, agreement = set(), None
        service_endpoints = []
        for launch in sorted(service.launches, key=lambda item: nodes[item.node].index):
            contexts = {
                "NodeContext": nodes[launch.node],
                "ServiceContext": service_context,
                "LaunchContext": LaunchContext(),
            }
            expansion = launch.expansion
            if expansion:
                values = (
                    expansion.dp_size,
                    expansion.tp_size,
                    expansion.pp_size,
                    expansion.cp_size,
                    expansion.sp_size,
                    expansion.dp_rpc_port,
                )
                if agreement is not None and values != agreement:
                    raise ValueError(f"Service {service.name}: inconsistent external DP group configuration")
                agreement = values
                if not 1 <= expansion.dp_rpc_port <= 65535:
                    raise ValueError("Invalid dp_rpc_port")
                ranks = range(expansion.dp_rank_start, expansion.dp_rank_start + expansion.dp_size_local)
                if rank_set.intersection(ranks) or any(rank >= expansion.dp_size for rank in ranks):
                    raise ValueError(f"Service {service.name}: overlapping or out-of-range DP ranks")
                rank_set.update(ranks)
                if expansion.devices_per_rank * expansion.dp_size_local > resources.npu_per_node:
                    raise ValueError(f"Service {service.name}: external ranks exceed node allocation")
                if (
                    expansion.device_ids is not None
                    and len(expansion.device_ids) != expansion.devices_per_rank * expansion.dp_size_local
                ):
                    raise ValueError(f"Service {service.name}: device_ids size disagrees with expansion")
            for local_rank in range(expansion.dp_size_local if expansion else 1):
                process_id = f"service/{service.name}/{launch.node}"
                rank_port = None
                if expansion:
                    rank = expansion.dp_rank_start + local_rank
                    rank_port = expansion.port_start + local_rank
                    process_id += f"/dp-rank-{rank}"
                    contexts = {**contexts, "LaunchContext": LaunchContext(rank_port, local_rank, rank)}
                template = launch.endpoint_template
                if expansion and template is None:
                    template = EndpointTemplate("{{ NodeContext.ip }}", "{{ LaunchContext.port }}")
                endpoint, process_contexts = endpoint_for(template, launch.env_template, contexts, rank_port)
                planned.append(_PlannedProcess(process_id, service.name, launch, process_contexts, endpoint))
                if endpoint:
                    service_endpoints.append(endpoint)
        if agreement and rank_set != set(range(agreement[0])):
            raise ValueError(f"Service {service.name}: DP ranks must cover 0..{agreement[0] - 1}")
        endpoints[service.name] = tuple(service_endpoints)

    local = [item for item in planned if item.launch.node == current.id]
    # Reserve explicit pools before allocating external launches that omit device_ids.
    reserved, external_pools = set(), {}
    for item in local:
        expansion = item.launch.expansion
        if expansion:
            if item.service in external_pools or expansion.device_ids is None:
                continue
            ids = expansion.device_ids
            external_pools[item.service] = ids
        elif "ASCEND_RT_VISIBLE_DEVICES" in item.launch.env_template:
            template = _endpoint_env(EndpointTemplate("local", "$ASCEND_RT_VISIBLE_DEVICES"), item.launch.env_template)
            ids = parse_device_ids(materialize_env(template, inherited_env, item.contexts)["ASCEND_RT_VISIBLE_DEVICES"])
        else:
            continue
        if not set(ids) <= set(cluster.current_device_pool) or reserved.intersection(ids):
            raise ValueError(f"Overlapping or out-of-allocation devices for {item.id}: {ids}")
        reserved.update(ids)
    for item in local:
        expansion = item.launch.expansion
        if expansion and item.service not in external_pools:
            count = expansion.devices_per_rank * expansion.dp_size_local
            pool = tuple(i for i in cluster.current_device_pool if i not in reserved)[:count]
            if len(pool) != count:
                raise ValueError(f"Insufficient devices for {item.id}")
            external_pools[item.service] = pool
            reserved.update(pool)
    processes = []
    for item in local:
        contexts = item.contexts
        expansion = item.launch.expansion
        assignment = None
        if expansion:
            rank = contexts["LaunchContext"].local_rank
            count = expansion.devices_per_rank
            assignment = ",".join(map(str, external_pools[item.service][rank * count : (rank + 1) * count]))
            contexts = {**contexts, "LaunchContext": replace(contexts["LaunchContext"], visible_devices=assignment)}
        env = materialize_env(item.launch.env_template, inherited_env, contexts)
        if assignment is not None:
            if "ASCEND_RT_VISIBLE_DEVICES" in env and parse_device_ids(
                env["ASCEND_RT_VISIBLE_DEVICES"]
            ) != parse_device_ids(assignment):
                raise ValueError(f"{item.id}: env devices disagree with external rank assignment")
            env["ASCEND_RT_VISIBLE_DEVICES"] = assignment
        argv = materialize_command(item.launch.command_template, {**inherited_env, **env}, contexts)
        if argv[:2] != ("vllm", "serve") or len(argv) < 3 or argv[2].startswith("-"):
            raise ValueError(f"{item.id}: expected canonical 'vllm serve MODEL ...'")
        cwd = render_scalar(item.launch.cwd, {**inherited_env, **env}, contexts) if item.launch.cwd else None
        processes.append(
            ProcessSpec(
                item.id, current.id, "server", ProcessOwner("service", item.service), argv, env, cwd, item.endpoint
            )
        )

    dependencies = []
    routing = deployment.routing
    if routing:
        groups = {}
        for name, services in routing.groups.items():
            for service in services:
                if not endpoints[service]:
                    raise ValueError(f"Routing Service {service} has no serving endpoint")
            groups[name] = tuple(endpoint for service in services for endpoint in endpoints[service])
        proxy = routing.proxy
        contexts = {"NodeContext": nodes[proxy.node], "LaunchContext": LaunchContext()}
        client, contexts = endpoint_for(proxy.endpoint_template, proxy.env_template, contexts)
        routing_context = (
            PDRoutingContext(groups["prefiller"], groups["decoder"], client)
            if routing.type == "disaggregated_prefill"
            else EPDRoutingContext(groups["encode"], groups["prefill"], groups["decode"], client)
        )
        contexts[type(routing_context).__name__] = routing_context
        process_id = f"routing/{routing.type}/{proxy.node}"
        dependencies.append(DependencySpec(process_id, tuple(e for values in groups.values() for e in values)))
        if proxy.node == current.id:
            env = materialize_env(proxy.env_template, inherited_env, contexts)
            argv = materialize_command(proxy.command_template, {**inherited_env, **env}, contexts)
            cwd = render_scalar(proxy.cwd, {**inherited_env, **env}, contexts) if proxy.cwd else None
            processes.append(
                ProcessSpec(
                    process_id, current.id, "router", ProcessOwner("routing", routing.type), argv, env, cwd, client
                )
            )
    else:
        serving = [e for values in endpoints.values() for e in values]
        if len(serving) != 1:
            raise ValueError("Without routing, exactly one global serving endpoint is required")
        client = serving[0]
    return LaunchPlan(current, tuple(processes), endpoints, client, tuple(dependencies))
