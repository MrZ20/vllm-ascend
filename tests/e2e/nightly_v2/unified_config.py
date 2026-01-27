#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""
Unified Test Configuration Module

This module provides a unified configuration system for both single-node and
multi-node vLLM inference tests. It supports:

1. Single-node tests: Running on a single machine with multiple NPUs
2. Multi-node tests: Running across multiple machines (K8s LeaderWorkerSet)

The configuration is driven by YAML files, enabling easy addition of new
model tests without modifying Python code.
"""

import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import psutil
import regex as re
import yaml

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SERVER_PORT = 8080
SINGLE_NODE_CONFIG_BASE_PATH = "tests/e2e/nightly_v2/single_node/config/"
MULTI_NODE_CONFIG_BASE_PATH = "tests/e2e/nightly_v2/multi_node/config/"


class DeploymentMode(Enum):
    """Deployment mode for the test."""
    SINGLE_NODE = "single_node"
    MULTI_NODE = "multi_node"


@dataclass(frozen=True)
class NodeInfo:
    """Information about a single node in the cluster."""
    index: int
    ip: str
    server_cmd: str
    envs: dict | None = None
    headless: bool = False

    def __post_init__(self):
        if not self.ip:
            raise ValueError("NodeInfo.ip must not be empty")

    def __str__(self) -> str:
        return ("NodeInfo(\n"
                f"  index={self.index},\n"
                f"  ip={self.ip},\n"
                f"  headless={self.headless},\n"
                ")")


@dataclass
class BenchmarkCase:
    """Configuration for a single benchmark case."""
    case_type: str  # "accuracy" or "performance"
    dataset_path: str
    request_conf: str
    dataset_conf: str
    max_out_len: int
    batch_size: int
    baseline: float = 1.0
    threshold: float = 0.97
    num_prompts: Optional[int] = None
    request_rate: float = 0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    trust_remote_code: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for aisbench runner."""
        result = {
            "case_type": self.case_type,
            "dataset_path": self.dataset_path,
            "request_conf": self.request_conf,
            "dataset_conf": self.dataset_conf,
            "max_out_len": self.max_out_len,
            "batch_size": self.batch_size,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.num_prompts is not None:
            result["num_prompts"] = self.num_prompts
        if self.request_rate:
            result["request_rate"] = self.request_rate
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.seed is not None:
            result["seed"] = self.seed
        if self.repetition_penalty is not None:
            result["repetition_penalty"] = self.repetition_penalty
        return result


@dataclass
class DisaggregatedPrefillConfig:
    """Configuration for disaggregated prefill mode."""
    enabled: bool = False
    prefiller_host_index: list[int] = field(default_factory=list)
    decoder_host_index: list[int] = field(default_factory=list)
    proxy_script: str = "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"

    def __post_init__(self):
        if self.enabled and not self.decoder_host_index:
            raise RuntimeError("decoder_host_index must be provided when disaggregated prefill is enabled")

    def validate(self, num_nodes: int):
        """Validate the configuration against the number of nodes."""
        if not self.enabled:
            return
        
        overlap = set(self.prefiller_host_index) & set(self.decoder_host_index)
        if overlap:
            raise AssertionError(f"Prefiller and decoder overlap: {overlap}")

        all_indices = self.prefiller_host_index + self.decoder_host_index
        if any(i >= num_nodes for i in all_indices):
            raise ValueError("Disaggregated prefill index out of range")

    @property
    def decode_start_index(self) -> int:
        return self.decoder_host_index[0] if self.decoder_host_index else 0

    @property
    def num_prefillers(self) -> int:
        return len(self.prefiller_host_index)

    @property
    def num_decoders(self) -> int:
        return len(self.decoder_host_index)

    def is_prefiller(self, index: int) -> bool:
        return index in self.prefiller_host_index

    def is_decoder(self, index: int) -> bool:
        return index in self.decoder_host_index

    def master_ip_for_node(self, index: int, nodes: list[NodeInfo]) -> str:
        if self.is_prefiller(index):
            return nodes[0].ip
        return nodes[self.decode_start_index].ip


@dataclass
class TestConfig:
    """
    Unified test configuration for both single-node and multi-node tests.
    
    This class encapsulates all configuration needed to run a vLLM inference test,
    regardless of whether it's running on a single node or multiple nodes.
    """
    # Basic info
    test_name: str
    model: str
    mode: DeploymentMode
    
    # Node configuration
    num_nodes: int
    npu_per_node: int
    nodes: list[NodeInfo]
    
    # Environment and server configuration
    env_common: dict[str, Any]
    server_args: list[str]
    
    # Benchmark configuration
    benchmarks: list[BenchmarkCase]
    
    # Optional configurations
    disaggregated_prefill: Optional[DisaggregatedPrefillConfig] = None
    
    # Runtime state (set after initialization)
    cur_index: int = 0
    cur_node: Optional[NodeInfo] = None
    proxy_port: int = 0
    envs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.disaggregated_prefill and self.disaggregated_prefill.enabled:
            self.disaggregated_prefill.validate(self.num_nodes)

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.npu_per_node

    @property
    def is_master(self) -> bool:
        return self.cur_index == 0

    @property
    def server_port(self) -> int:
        return int(self.env_common.get("SERVER_PORT", DEFAULT_SERVER_PORT))

    @property
    def master_ip(self) -> str:
        return self.nodes[0].ip if self.nodes else "localhost"

    @property
    def benchmark_endpoint(self) -> tuple[str, int]:
        """Endpoint used by benchmark clients."""
        if self.disaggregated_prefill and self.disaggregated_prefill.enabled:
            return self.master_ip, self.proxy_port
        return self.master_ip, self.server_port

    @property
    def server_cmd(self) -> str:
        """Get the server command for the current node."""
        if self.cur_node:
            return self.cur_node.server_cmd
        # For single node, build command from server_args
        return self._build_server_cmd()

    def _build_server_cmd(self) -> str:
        """Build server command string from server_args list."""
        if not self.server_args:
            return ""
        return " ".join(self.server_args)

    def get_aisbench_cases(self) -> list[dict]:
        """Get benchmark cases as list of dicts for aisbench runner."""
        return [b.to_dict() for b in self.benchmarks]

    @property
    def acc_cmd(self) -> Optional[dict]:
        """Get accuracy benchmark case."""
        for b in self.benchmarks:
            if b.case_type == "accuracy":
                return b.to_dict()
        return None

    @property
    def perf_cmd(self) -> Optional[dict]:
        """Get performance benchmark case."""
        for b in self.benchmarks:
            if b.case_type == "performance":
                return b.to_dict()
        return None


# =============================================================================
# Network Utilities
# =============================================================================

def get_all_ipv4() -> set[str]:
    """Get all IPv4 addresses of the current machine."""
    ips = set()
    for iface_name, iface_addrs in psutil.net_if_addrs().items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET:
                ips.add(addr.address)
    return ips


def get_net_interface(target_ip: str) -> str:
    """Get the network interface name for a given IP address."""
    for iface_name, iface_addrs in psutil.net_if_addrs().items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and addr.address == target_ip:
                return iface_name
    raise RuntimeError(f"No interface found for IP: {target_ip}")


def get_available_port(start_port: int = 6000, end_port: int = 7000) -> int:
    """Find an available port in the given range."""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


def dns_resolver(retries: int = 240, base_delay: float = 0.5):
    """Create a DNS resolver function with retry logic."""
    def resolve(dns: str) -> str:
        delay = base_delay
        for attempt in range(retries):
            try:
                return socket.gethostbyname(dns)
            except socket.gaierror:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.5, 5)
        raise RuntimeError(f"Failed to resolve DNS: {dns}")
    return resolve


def get_cluster_dns_list(world_size: int) -> list[str]:
    """Get DNS names for all nodes in a K8s LeaderWorkerSet cluster."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    leader_dns = os.getenv("LWS_LEADER_ADDRESS")
    if not leader_dns:
        raise RuntimeError("environment variable LWS_LEADER_ADDRESS is not set")

    parts = leader_dns.split(".")
    if len(parts) < 3:
        raise ValueError(f"invalid leader DNS format: {leader_dns}")

    leader_name, group_name, namespace = parts[0], parts[1], parts[2]

    worker_dns_list = [
        f"{leader_name}-{idx}.{group_name}.{namespace}"
        for idx in range(1, world_size)
    ]

    return [leader_dns, *worker_dns_list]


def get_cluster_ips(num_nodes: int = 2) -> list[str]:
    """Get IP addresses for all nodes in the cluster."""
    resolver = dns_resolver()
    return [resolver(dns) for dns in get_cluster_dns_list(num_nodes)]


def get_cur_ip(retries: int = 20, base_delay: float = 0.5) -> str:
    """Get the current machine's primary IP address with retry."""
    delay = base_delay

    for attempt in range(retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 5)
    raise RuntimeError("Failed to get current IP")


# =============================================================================
# Environment Builder
# =============================================================================

class DistEnvBuilder:
    """Build distributed environment variables for a node."""

    def __init__(
        self,
        *,
        cur_node: NodeInfo,
        master_ip: str,
        common_envs: dict,
    ):
        self.cur_ip = cur_node.ip
        self.nic_name = get_net_interface(self.cur_ip)
        self.master_ip = master_ip
        self.base_envs = {**common_envs, **(cur_node.envs or {})}

    def build(self) -> dict[str, str]:
        envs = dict(self.base_envs)
        envs.update({
            "HCCL_IF_IP": self.cur_ip,
            "HCCL_SOCKET_IFNAME": self.nic_name,
            "GLOO_SOCKET_IFNAME": self.nic_name,
            "TP_SOCKET_IFNAME": self.nic_name,
            "LOCAL_IP": self.cur_ip,
            "NIC_NAME": self.nic_name,
            "MASTER_IP": self.master_ip,
        })
        return {k: str(v) for k, v in envs.items()}


class SingleNodeEnvBuilder:
    """Build environment variables for single-node deployment."""

    def __init__(self, common_envs: dict):
        self.base_envs = common_envs

    def build(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.base_envs.items()}


# =============================================================================
# Configuration Loader
# =============================================================================

class UnifiedConfigLoader:
    """
    Unified configuration loader that supports both single-node and multi-node tests.
    
    For single-node tests:
    - num_nodes = 1
    - Nodes list contains single localhost entry
    - Server command is built from server_args
    
    For multi-node tests:
    - num_nodes > 1
    - Nodes list contains entries for each node
    - Server commands come from deployment section
    """

    DEFAULT_SINGLE_NODE_CONFIG = "Qwen3-32B.yaml"
    DEFAULT_MULTI_NODE_CONFIG = "DeepSeek-V3.yaml"

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None, mode: Optional[DeploymentMode] = None) -> TestConfig:
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file (relative or absolute)
            mode: Deployment mode (auto-detected if not provided)
        
        Returns:
            TestConfig instance ready for test execution
        """
        config_dict = cls._load_yaml(yaml_path, mode)
        detected_mode = cls._detect_mode(config_dict)
        cls._validate_config(config_dict, detected_mode)

        if detected_mode == DeploymentMode.SINGLE_NODE:
            return cls._build_single_node_config(config_dict)
        else:
            return cls._build_multi_node_config(config_dict)

    @classmethod
    def _load_yaml(cls, yaml_path: Optional[str], mode: Optional[DeploymentMode]) -> dict:
        """Load and parse YAML configuration file."""
        if not yaml_path:
            yaml_path = os.getenv("CONFIG_YAML_PATH")
        
        if not yaml_path:
            # Use default based on mode
            if mode == DeploymentMode.SINGLE_NODE:
                yaml_path = cls.DEFAULT_SINGLE_NODE_CONFIG
            else:
                yaml_path = cls.DEFAULT_MULTI_NODE_CONFIG

        # Determine base path
        if os.path.isabs(yaml_path):
            full_path = yaml_path
        else:
            # Try single_node config first, then multi_node
            single_node_path = os.path.join(SINGLE_NODE_CONFIG_BASE_PATH, yaml_path)
            multi_node_path = os.path.join(MULTI_NODE_CONFIG_BASE_PATH, yaml_path)
            
            if os.path.exists(single_node_path):
                full_path = single_node_path
            elif os.path.exists(multi_node_path):
                full_path = multi_node_path
            else:
                raise FileNotFoundError(
                    f"Config file not found: {yaml_path}\n"
                    f"Searched: {single_node_path}, {multi_node_path}"
                )

        logger.info("Loading config yaml: %s", full_path)

        with open(full_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def _detect_mode(cls, config: dict) -> DeploymentMode:
        """Detect deployment mode from configuration."""
        num_nodes = config.get("num_nodes", 1)
        if num_nodes == 1:
            return DeploymentMode.SINGLE_NODE
        return DeploymentMode.MULTI_NODE

    @classmethod
    def _validate_config(cls, config: dict, mode: DeploymentMode):
        """Validate configuration has all required fields."""
        common_required = ["model"]
        
        if mode == DeploymentMode.SINGLE_NODE:
            # server_args or deployment can be used
            if "server_args" not in config and "deployment" not in config:
                raise KeyError("Single-node config must have 'server_args' or 'deployment'")
        else:
            required = common_required + ["deployment", "num_nodes", "npu_per_node"]
            missing = [k for k in required if k not in config]
            if missing:
                raise KeyError(f"Missing required config fields: {missing}")

    @classmethod
    def _parse_benchmarks(cls, config: dict) -> list[BenchmarkCase]:
        """Parse benchmark configurations."""
        benchmarks_raw = config.get("benchmarks", {})
        if not benchmarks_raw:
            return []
        
        benchmarks = []
        
        # Handle dict format (acc/perf keys)
        if isinstance(benchmarks_raw, dict):
            for key, bench_config in benchmarks_raw.items():
                if bench_config:
                    benchmarks.append(BenchmarkCase(**bench_config))
        # Handle list format
        elif isinstance(benchmarks_raw, list):
            for bench_config in benchmarks_raw:
                if bench_config:
                    benchmarks.append(BenchmarkCase(**bench_config))
        
        return benchmarks

    @classmethod
    def _parse_disaggregated_prefill(cls, config: dict) -> Optional[DisaggregatedPrefillConfig]:
        """Parse disaggregated prefill configuration."""
        dp_config = config.get("disaggregated_prefill")
        if not dp_config or not dp_config.get("enabled", False):
            return None
        
        return DisaggregatedPrefillConfig(
            enabled=True,
            prefiller_host_index=dp_config.get("prefiller_host_index", []),
            decoder_host_index=dp_config.get("decoder_host_index", []),
            proxy_script=config.get("env_common", {}).get(
                "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
                "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
            )
        )

    @classmethod
    def _build_single_node_config(cls, config: dict) -> TestConfig:
        """Build TestConfig for single-node deployment."""
        model = config["model"]
        test_name = config.get("test_name", f"test {model}")
        npu_per_node = config.get("npu_per_node", 8)
        env_common = config.get("env_common", {})
        
        # Parse server args
        server_args = config.get("server_args", [])
        if isinstance(server_args, str):
            server_args = server_args.split()
        
        # For single node with deployment section (compatibility with multi-node format)
        if "deployment" in config and config["deployment"]:
            deployment = config["deployment"][0]
            server_cmd = deployment.get("server_cmd", "")
            node_envs = deployment.get("envs", {})
        else:
            server_cmd = ""
            node_envs = {}

        # Build server_args list if we have server_cmd instead
        if server_cmd and not server_args:
            # Parse server command to extract args
            server_args = cls._parse_server_cmd_to_args(server_cmd, model)

        # Create single localhost node
        node = NodeInfo(
            index=0,
            ip="localhost",
            server_cmd=server_cmd or cls._build_server_cmd_from_args(model, server_args),
            envs=node_envs,
            headless=False,
        )

        # Build environment
        env_builder = SingleNodeEnvBuilder({**env_common, **node_envs})
        envs = env_builder.build()

        test_config = TestConfig(
            test_name=test_name,
            model=model,
            mode=DeploymentMode.SINGLE_NODE,
            num_nodes=1,
            npu_per_node=npu_per_node,
            nodes=[node],
            env_common=env_common,
            server_args=server_args,
            benchmarks=cls._parse_benchmarks(config),
            disaggregated_prefill=None,
            cur_index=0,
            cur_node=node,
            envs=envs,
        )

        return test_config

    @classmethod
    def _build_multi_node_config(cls, config: dict) -> TestConfig:
        """Build TestConfig for multi-node deployment."""
        model = config["model"]
        test_name = config.get("test_name", f"test {model}")
        num_nodes = config["num_nodes"]
        npu_per_node = config.get("npu_per_node", 16)
        env_common = config.get("env_common", {})
        
        # Parse disaggregated prefill
        disagg_config = cls._parse_disaggregated_prefill(config)
        
        # Parse nodes
        nodes = cls._parse_nodes(config)
        
        # Determine current node index
        cur_index = cls._resolve_cur_index(nodes)
        cur_node = nodes[cur_index]
        
        # Determine master IP for this node
        if disagg_config:
            master_ip = disagg_config.master_ip_for_node(cur_index, nodes)
        else:
            master_ip = nodes[0].ip
        
        # Build environment
        env_builder = DistEnvBuilder(
            cur_node=cur_node,
            master_ip=master_ip,
            common_envs=env_common,
        )
        envs = env_builder.build()
        
        # Get proxy port
        proxy_port = get_available_port()

        # Expand environment variables in server command
        server_cmd = cls._expand_env(cur_node.server_cmd, envs)

        # Update node with expanded command
        updated_node = NodeInfo(
            index=cur_node.index,
            ip=cur_node.ip,
            server_cmd=server_cmd,
            envs=cur_node.envs,
            headless=cur_node.headless,
        )

        test_config = TestConfig(
            test_name=test_name,
            model=model,
            mode=DeploymentMode.MULTI_NODE,
            num_nodes=num_nodes,
            npu_per_node=npu_per_node,
            nodes=nodes,
            env_common=env_common,
            server_args=[],  # Multi-node uses server_cmd in nodes
            benchmarks=cls._parse_benchmarks(config),
            disaggregated_prefill=disagg_config,
            cur_index=cur_index,
            cur_node=updated_node,
            proxy_port=proxy_port,
            envs=envs,
        )

        logger.info("Node %d envs: %s", cur_index, envs)
        return test_config

    @classmethod
    def _parse_nodes(cls, config: dict) -> list[NodeInfo]:
        """Parse node information from deployment configuration."""
        num_nodes = config["num_nodes"]
        deployments = config["deployment"]

        if len(deployments) != num_nodes:
            raise AssertionError(
                f"deployment size ({len(deployments)}) != num_nodes ({num_nodes})"
            )

        cluster_ips = cls._resolve_cluster_ips(config, num_nodes)

        nodes: list[NodeInfo] = []
        for idx, deploy in enumerate(deployments):
            cmd = deploy.get("server_cmd", "")
            envs = deploy.get("envs", {})
            nodes.append(
                NodeInfo(
                    index=idx,
                    ip=cluster_ips[idx],
                    server_cmd=cmd,
                    envs=envs,
                    headless="--headless" in cmd,
                ))
        return nodes

    @classmethod
    def _resolve_cluster_ips(cls, config: dict, num_nodes: int) -> list[str]:
        """Resolve IP addresses for cluster nodes."""
        if "cluster_hosts" in config and config["cluster_hosts"]:
            logger.info("Using cluster_hosts from config (non-Kubernetes environment)")
            ips = config["cluster_hosts"]
            if len(ips) != num_nodes:
                raise AssertionError("cluster_hosts size mismatch")
            return ips

        logger.info("Resolving cluster IPs via DNS...")
        return get_cluster_ips(num_nodes)

    @classmethod
    def _resolve_cur_index(cls, nodes: list[NodeInfo]) -> int:
        """Determine the current node index."""
        if (idx := os.environ.get("LWS_WORKER_INDEX")):
            return int(idx)

        local_ips = get_all_ipv4()
        for i, node in enumerate(nodes):
            if node.ip in local_ips:
                return i

        # Default to 0 for single-node or when running locally
        return 0

    @staticmethod
    def _expand_env(cmd: str, envs: dict) -> str:
        """Expand environment variables in command string."""
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m):
            key = m.group(1) or m.group(2)
            return envs.get(key, m.group(0))

        return pattern.sub(repl, cmd)

    @staticmethod
    def _parse_server_cmd_to_args(server_cmd: str, model: str) -> list[str]:
        """Parse server command string to args list."""
        # Skip the 'vllm serve <model>' part
        parts = server_cmd.split()
        args = []
        skip_next = False
        for i, part in enumerate(parts):
            if skip_next:
                skip_next = False
                continue
            if part in ["vllm", "serve"] or part == model or part.startswith('"'):
                continue
            args.append(part)
        return args

    @staticmethod
    def _build_server_cmd_from_args(model: str, server_args: list[str]) -> str:
        """Build vllm serve command from model and args."""
        return f'vllm serve "{model}" ' + " ".join(server_args)


# Backward compatibility alias
ConfigLoader = UnifiedConfigLoader
