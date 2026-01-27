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
Unified Test Runner Module

This module provides a unified test runner that can execute both single-node
and multi-node tests based on YAML configuration.

The test runner:
1. Loads configuration from YAML file
2. Optionally starts proxy server (for disaggregated prefill)
3. Starts vLLM server(s)
4. Runs benchmark tests
5. Reports results
"""

import logging
import os
import subprocess
from typing import Optional

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly_v2.unified_config import (
    DeploymentMode,
    DisaggregatedPrefillConfig,
    NodeInfo,
    TestConfig,
    UnifiedConfigLoader,
)
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)


class ProxyLauncher:
    """
    Launcher for disaggregated prefill proxy server.
    
    This is used in disaggregated prefill mode to route requests
    between prefiller and decoder nodes.
    """

    def __init__(
        self,
        *,
        nodes: list[NodeInfo],
        envs: dict,
        proxy_port: int,
        cur_index: int,
        disagg_cfg: Optional[DisaggregatedPrefillConfig] = None,
        server_port: int = 8080,
    ):
        self.nodes = nodes
        self.cfg = disagg_cfg
        self.server_port = server_port
        self.proxy_port = proxy_port
        self.proxy_script = (
            disagg_cfg.proxy_script if disagg_cfg else
            'examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py'
        )
        self.envs = envs
        self.is_master = cur_index == 0
        self.cur_ip = nodes[cur_index].ip if nodes else "localhost"
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self):
        if not self.is_master or self.cfg is None or not self.cfg.enabled:
            logger.info("Not launching proxy (not master or disaggregated prefill disabled)")
            return self
        
        prefiller_ips = [self.nodes[i].ip for i in self.cfg.prefiller_host_index]
        decoder_ips = [self.nodes[i].ip for i in self.cfg.decoder_host_index]

        cmd = [
            "python",
            self.proxy_script,
            "--host",
            self.cur_ip,
            "--port",
            str(self.proxy_port),
            "--prefiller-hosts",
            *prefiller_ips,
            "--prefiller-ports",
            *[str(self.server_port)] * len(prefiller_ips),
            "--decoder-hosts",
            *decoder_ips,
            "--decoder-ports",
            *[str(self.server_port)] * len(decoder_ips),
        ]

        logger.info("Launching proxy: %s", " ".join(cmd))
        self.process = subprocess.Popen(cmd, env={**os.environ, **self.envs})
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.process:
            return
        logger.info("Stopping proxy server...")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class UnifiedTestRunner:
    """
    Unified test runner for single-node and multi-node tests.
    
    This class provides a consistent interface for running tests
    regardless of the deployment mode.
    """

    def __init__(self, config: TestConfig):
        self.config = config

    def run(self):
        """Execute the test based on configuration."""
        if self.config.mode == DeploymentMode.SINGLE_NODE:
            return self._run_single_node_test()
        else:
            return self._run_multi_node_test()

    def _run_single_node_test(self):
        """Run single-node test."""
        config = self.config
        
        # Build server args
        server_args = self._build_server_args()
        
        logger.info("Starting single-node test for model: %s", config.model)
        logger.info("Server args: %s", server_args)
        
        with RemoteOpenAIServer(
            model=config.model,
            vllm_serve_args=server_args,
            server_port=config.server_port,
            env_dict=config.envs,
            auto_port=False,
        ) as server:
            # Run validation request
            client = server.get_async_client()
            
            # Run benchmark tests
            host, port = config.benchmark_endpoint
            aisbench_cases = config.get_aisbench_cases()
            
            if aisbench_cases:
                run_aisbench_cases(
                    model=config.model,
                    port=port,
                    aisbench_cases=aisbench_cases,
                    host_ip=host,
                )
            else:
                logger.info("No benchmark cases configured, skipping benchmarks")

    def _run_multi_node_test(self):
        """Run multi-node test."""
        config = self.config

        with ProxyLauncher(
            nodes=config.nodes,
            disagg_cfg=config.disaggregated_prefill,
            envs=config.envs,
            proxy_port=config.proxy_port,
            cur_index=config.cur_index,
            server_port=config.server_port,
        ) as proxy:
            
            with RemoteOpenAIServer(
                model=config.model,
                vllm_serve_args=config.server_cmd,
                server_port=config.server_port,
                server_host=config.master_ip,
                env_dict=config.envs,
                auto_port=False,
                proxy_port=proxy.proxy_port,
                disaggregated_prefill=config.disaggregated_prefill,
                nodes_info=config.nodes,
                max_wait_seconds=2800,
            ) as server:

                host, port = config.benchmark_endpoint

                if config.is_master:
                    # Master node runs benchmarks
                    aisbench_cases = []
                    if config.acc_cmd:
                        aisbench_cases.append(config.acc_cmd)
                    if config.perf_cmd:
                        aisbench_cases.append(config.perf_cmd)
                    
                    run_aisbench_cases(
                        model=config.model,
                        port=port,
                        aisbench_cases=aisbench_cases,
                        host_ip=host,
                    )
                else:
                    # Worker nodes wait for master
                    server.hang_until_terminated(
                        f"http://{host}:{config.server_port}/health"
                    )

    def _build_server_args(self) -> list[str]:
        """Build server arguments for the test."""
        config = self.config
        
        if config.server_args:
            return config.server_args
        
        # Build from configuration
        args = []
        
        # Add port
        if config.server_port != 8080:
            args.extend(["--port", str(config.server_port)])
        
        return args


# =============================================================================
# Pytest Integration
# =============================================================================

def run_unified_test(yaml_path: Optional[str] = None):
    """
    Main entry point for running a unified test.
    
    This function is designed to be called from pytest test files.
    
    Args:
        yaml_path: Path to the YAML configuration file (or use CONFIG_YAML_PATH env var)
    """
    config = UnifiedConfigLoader.from_yaml(yaml_path)
    runner = UnifiedTestRunner(config)
    runner.run()


@pytest.mark.asyncio
async def test_from_yaml():
    """
    Generic test function that runs tests based on YAML configuration.
    
    The configuration file path should be set via CONFIG_YAML_PATH environment variable.
    """
    run_unified_test()
