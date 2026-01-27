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
Conftest for nightly_v2 tests.

This module provides pytest configuration and fixtures for the unified
test framework.
"""

import os
from pathlib import Path
from typing import Optional

import pytest

# Configuration paths
SINGLE_NODE_CONFIG_DIR = Path(__file__).parent / "single_node" / "config"
MULTI_NODE_CONFIG_DIR = Path(__file__).parent / "multi_node" / "config"


def pytest_addoption(parser):
    """Add command line options for test configuration."""
    parser.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to YAML configuration file"
    )
    parser.addoption(
        "--config-dir",
        action="store",
        default=None,
        help="Directory containing configuration files (for parametrized tests)"
    )


def get_config_files(config_dir: Path) -> list[str]:
    """Get all YAML configuration files in the given directory."""
    if not config_dir.exists():
        return []
    return sorted([f.name for f in config_dir.glob("*.yaml")])


def pytest_generate_tests(metafunc):
    """
    Dynamically parametrize tests based on configuration files.
    
    This allows running tests for all configurations in a directory
    when no specific configuration is specified.
    """
    # Only apply to tests that use the config_path fixture
    if "config_path" not in metafunc.fixturenames:
        return
    
    # Check if a specific config is provided
    config = metafunc.config.getoption("config")
    if config:
        metafunc.parametrize("config_path", [config])
        return
    
    # Check environment variable
    env_config = os.getenv("CONFIG_YAML_PATH")
    if env_config:
        metafunc.parametrize("config_path", [env_config])
        return
    
    # Check config-dir option
    config_dir = metafunc.config.getoption("config_dir")
    if config_dir:
        config_files = get_config_files(Path(config_dir))
        if config_files:
            metafunc.parametrize("config_path", config_files)
            return
    
    # Determine which config directory to use based on test location
    test_file = metafunc.definition.fspath
    if "single_node" in str(test_file):
        config_files = get_config_files(SINGLE_NODE_CONFIG_DIR)
    elif "multi_node" in str(test_file):
        config_files = get_config_files(MULTI_NODE_CONFIG_DIR)
    else:
        config_files = []
    
    if config_files:
        # If running parametrized tests over all configs
        if os.getenv("RUN_ALL_CONFIGS", "").lower() == "true":
            metafunc.parametrize("config_path", config_files)
        else:
            # Default to no test if no config specified
            pytest.skip("No configuration specified. Set CONFIG_YAML_PATH or use --config")


@pytest.fixture
def config_path(request) -> Optional[str]:
    """
    Fixture to provide the configuration file path.
    
    Priority:
    1. Command line --config option
    2. CONFIG_YAML_PATH environment variable
    3. Parametrized value from pytest_generate_tests
    """
    # This will be populated by pytest_generate_tests or return None
    if hasattr(request, "param"):
        return request.param
    
    config = request.config.getoption("config")
    if config:
        return config
    
    return os.getenv("CONFIG_YAML_PATH")


@pytest.fixture(scope="session")
def single_node_config_dir() -> Path:
    """Return the single node configuration directory."""
    return SINGLE_NODE_CONFIG_DIR


@pytest.fixture(scope="session")
def multi_node_config_dir() -> Path:
    """Return the multi node configuration directory."""
    return MULTI_NODE_CONFIG_DIR


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on configuration.
    """
    for item in items:
        # Add markers based on test location
        if "single_node" in str(item.fspath):
            item.add_marker(pytest.mark.single_node)
        elif "multi_node" in str(item.fspath):
            item.add_marker(pytest.mark.multi_node)


def pytest_configure(config):
    """
    Register custom markers.
    """
    config.addinivalue_line(
        "markers", "single_node: mark test as single-node test"
    )
    config.addinivalue_line(
        "markers", "multi_node: mark test as multi-node test"
    )
