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
Single Node Test Runner

This module provides the pytest entry point for running single-node tests
based on YAML configuration files.

Usage:
    # Run a specific test
    CONFIG_YAML_PATH=Qwen3-32B.yaml pytest tests/e2e/nightly_v2/single_node/test_single_node.py

    # Run with pytest directly
    pytest tests/e2e/nightly_v2/single_node/test_single_node.py --config Qwen3-32B.yaml
"""

import os
import logging
from typing import Any

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly_v2.unified_config import UnifiedConfigLoader, DeploymentMode
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

# Simple validation prompt
VALIDATION_PROMPT = "San Francisco is a"
VALIDATION_MAX_TOKENS = 10


def pytest_addoption(parser):
    """Add command line options for test configuration."""
    parser.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to YAML configuration file"
    )


@pytest.fixture
def config_path(request):
    """Get configuration path from command line or environment."""
    return request.config.getoption("--config") or os.getenv("CONFIG_YAML_PATH")


@pytest.mark.asyncio
async def test_single_node(config_path: str = None):
    """
    Single node test that runs based on YAML configuration.
    
    This test:
    1. Loads configuration from YAML file
    2. Starts a vLLM server with the specified model and arguments
    3. Validates the server is working with a simple completion request
    4. Runs benchmark tests (accuracy and/or performance)
    
    Args:
        config_path: Path to the YAML configuration file.
                    Can also be set via CONFIG_YAML_PATH environment variable.
    """
    # Load configuration
    yaml_path = config_path or os.getenv("CONFIG_YAML_PATH")
    if not yaml_path:
        pytest.skip("No configuration file specified. Set CONFIG_YAML_PATH or use --config")
    
    config = UnifiedConfigLoader.from_yaml(yaml_path, mode=DeploymentMode.SINGLE_NODE)
    
    logger.info("=" * 60)
    logger.info("Running test: %s", config.test_name)
    logger.info("Model: %s", config.model)
    logger.info("NPUs: %d", config.npu_per_node)
    logger.info("=" * 60)
    
    # Get available port
    port = get_open_port()
    
    # Build server args with port
    server_args = list(config.server_args)
    
    # Add port if not already specified
    if "--port" not in server_args:
        server_args.extend(["--port", str(port)])
    else:
        # Update existing port
        port_idx = server_args.index("--port")
        if port_idx + 1 < len(server_args):
            server_args[port_idx + 1] = str(port)
            port = int(server_args[port_idx + 1])
    
    logger.info("Server args: %s", " ".join(server_args))
    logger.info("Environment: %s", config.envs)
    
    # Start server and run tests
    with RemoteOpenAIServer(
        model=config.model,
        vllm_serve_args=server_args,
        server_port=port,
        env_dict=config.envs,
        auto_port=False,
    ) as server:
        # Validate server is working
        client = server.get_async_client()
        
        logger.info("Validating server with completion request...")
        batch = await client.completions.create(
            model=config.model,
            prompt=VALIDATION_PROMPT,
            max_tokens=VALIDATION_MAX_TOKENS,
        )
        
        choices: list[openai.types.CompletionChoice] = batch.choices
        assert choices[0].text, "Server returned empty response"
        logger.info("Server validation successful: %s", choices[0].text[:50])
        
        # Run benchmark tests
        aisbench_cases = config.get_aisbench_cases()
        if aisbench_cases:
            logger.info("Running %d benchmark cases...", len(aisbench_cases))
            run_aisbench_cases(
                model=config.model,
                port=port,
                aisbench_cases=aisbench_cases,
                server_args=" ".join(server_args),
            )
        else:
            logger.info("No benchmark cases configured, skipping benchmarks")
    
    logger.info("Test completed successfully: %s", config.test_name)


# For direct execution
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_single_node())
