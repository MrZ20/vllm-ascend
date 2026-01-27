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
Multi Node Test Runner

This module provides the pytest entry point for running multi-node tests
based on YAML configuration files. It uses the unified configuration system
and can run both single-node and multi-node tests.

Usage:
    # Set configuration via environment variable
    CONFIG_YAML_PATH=DeepSeek-V3.yaml pytest tests/e2e/nightly_v2/multi_node/test_multi_node.py
"""

import logging
import os

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly_v2.unified_config import UnifiedConfigLoader, DeploymentMode
from tests.e2e.nightly_v2.unified_runner import ProxyLauncher
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_multi_node() -> None:
    """
    Multi-node test that runs based on YAML configuration.
    
    This test:
    1. Loads configuration from YAML file
    2. Starts proxy server if using disaggregated prefill
    3. Starts vLLM server with the specified model and arguments
    4. On master node: runs benchmark tests
    5. On worker nodes: waits for master to complete
    
    Configuration file path should be set via CONFIG_YAML_PATH environment variable.
    """
    yaml_path = os.getenv("CONFIG_YAML_PATH")
    if not yaml_path:
        pytest.skip("No configuration file specified. Set CONFIG_YAML_PATH")
    
    config = UnifiedConfigLoader.from_yaml(yaml_path)
    
    logger.info("=" * 60)
    logger.info("Running test: %s", config.test_name)
    logger.info("Model: %s", config.model)
    logger.info("Mode: %s", config.mode.value)
    logger.info("Nodes: %d, NPUs per node: %d", config.num_nodes, config.npu_per_node)
    logger.info("Current node index: %d", config.cur_index)
    logger.info("Is master: %s", config.is_master)
    logger.info("=" * 60)

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
                
                if aisbench_cases:
                    run_aisbench_cases(
                        model=config.model,
                        port=port,
                        aisbench_cases=aisbench_cases,
                        host_ip=host,
                    )
                else:
                    logger.info("No benchmark cases configured")
            else:
                # Worker nodes wait for master
                logger.info("Worker node waiting for master to complete...")
                server.hang_until_terminated(
                    f"http://{host}:{config.server_port}/health"
                )
    
    logger.info("Test completed: %s", config.test_name)
