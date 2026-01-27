#!/bin/bash
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

set -euo pipefail

# =============================================================================
# Unified Test Runner Script for vLLM-Ascend
# =============================================================================
# This script provides a unified entry point for running both single-node and
# multi-node tests. It handles environment setup, dependency checks, and test
# execution.
#
# Usage:
#   Single node:
#     CONFIG_YAML_PATH=Qwen3-32B.yaml ./run.sh single
#
#   Multi node (K8s environment):
#     CONFIG_YAML_PATH=DeepSeek-V3.yaml ./run.sh multi
#
# Environment Variables:
#   CONFIG_YAML_PATH - Path to the YAML configuration file (required)
#   WORKSPACE - Path to the workspace directory (default: /vllm-workspace)
#   FAIL_TAG - Tag for failure messages in multi-node mode

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Configuration
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages"
export BENCHMARK_HOME="${WORKSPACE:-/vllm-workspace}/vllm-ascend/benchmark"

# Logging configurations
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export GLOG_minloglevel=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# =============================================================================
# Utility Functions
# =============================================================================

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

print_failure() {
    echo -e "${RED}${FAIL_TAG:-test_failed} ✗ ERROR: $1${NC}"
    exit 1
}

# =============================================================================
# Environment Checks
# =============================================================================

check_npu_info() {
    print_section "Checking NPU Info"
    npu-smi info || print_warning "npu-smi not available"
    
    local info_file="/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info"
    if [[ -f "$info_file" ]]; then
        cat "$info_file"
    else
        print_warning "Ascend toolkit info file not found"
    fi
}

show_vllm_info() {
    print_section "vLLM Version Information"
    
    echo "Installed vLLM-related Python packages:"
    pip list | grep -i vllm || echo "No vllm packages found."
    
    local workspace="${WORKSPACE:-$(pwd)}"
    
    for repo in "vllm" "vllm-ascend"; do
        local repo_path="$workspace/$repo"
        if [[ -d "$repo_path/.git" ]]; then
            echo ""
            echo "============================"
            echo "$repo Git information"
            echo "============================"
            cd "$repo_path"
            echo "Branch:      $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
            echo "Commit hash: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
            echo "Author:      $(git log -1 --pretty=format:'%an <%ae>' 2>/dev/null || echo 'N/A')"
            echo "Date:        $(git log -1 --pretty=format:'%ad' --date=iso 2>/dev/null || echo 'N/A')"
            echo "Message:     $(git log -1 --pretty=format:'%s' 2>/dev/null || echo 'N/A')"
        fi
    done
}

check_and_config() {
    print_section "Configuring Environment"
    
    # Configure git proxy if needed
    if [[ "${USE_GIT_PROXY:-true}" == "true" ]]; then
        git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/" || true
    fi
    
    # Configure pip mirror
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple || true
    export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
    
    print_success "Environment configured"
}

show_triton_ascend_info() {
    print_section "Triton Ascend Info"
    
    clang -v 2>&1 | head -n 1 || print_warning "clang not found"
    which bishengir-compile || print_warning "bishengir-compile not found"
    pip show triton-ascend 2>/dev/null || print_warning "triton-ascend not installed"
}

# =============================================================================
# Process Management
# =============================================================================

kill_npu_processes() {
    print_section "Cleaning up NPU processes"
    
    pgrep python3 | xargs -r kill -9 2>/dev/null || true
    pgrep VLLM | xargs -r kill -9 2>/dev/null || true
    
    sleep 4
    print_success "Cleanup completed"
}

# =============================================================================
# Test Execution
# =============================================================================

run_single_node_test() {
    print_section "Running Single Node Test"
    
    local config_path="${CONFIG_YAML_PATH:-}"
    if [[ -z "$config_path" ]]; then
        print_error "CONFIG_YAML_PATH environment variable not set"
    fi
    
    echo "Configuration: $config_path"
    
    set +e
    pytest -sv --show-capture=no tests/e2e/nightly_v2/single_node/test_single_node.py
    local ret=$?
    set -e
    
    if [[ $ret -eq 0 ]]; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed"
    fi
    
    return $ret
}

run_multi_node_test() {
    print_section "Running Multi Node Test"
    
    local config_path="${CONFIG_YAML_PATH:-}"
    if [[ -z "$config_path" ]]; then
        print_error "CONFIG_YAML_PATH environment variable not set"
    fi
    
    echo "Configuration: $config_path"
    echo "Node index: ${LWS_WORKER_INDEX:-0}"
    
    set +e
    kill_npu_processes
    pytest -sv --show-capture=no tests/e2e/nightly_v2/multi_node/test_multi_node.py
    local ret=$?
    set -e
    
    # In multi-node mode, only master node reports final status
    if [[ "${LWS_WORKER_INDEX:-0}" -eq 0 ]]; then
        if [[ $ret -eq 0 ]]; then
            print_success "All tests passed!"
        else
            print_failure "Some tests failed, please check the error stack above for details. \
If this is insufficient to pinpoint the error, please download and review the logs of all other nodes from the job's summary."
        fi
    fi
    
    return $ret
}

# =============================================================================
# Main Entry Point
# =============================================================================

usage() {
    echo "Usage: $0 [single|multi]"
    echo ""
    echo "Commands:"
    echo "  single    Run single-node test"
    echo "  multi     Run multi-node test (K8s environment)"
    echo ""
    echo "Environment Variables:"
    echo "  CONFIG_YAML_PATH    Path to YAML configuration file (required)"
    echo "  WORKSPACE           Workspace directory (default: /vllm-workspace)"
    echo "  FAIL_TAG            Failure tag for multi-node mode"
    echo ""
    echo "Examples:"
    echo "  CONFIG_YAML_PATH=Qwen3-32B.yaml $0 single"
    echo "  CONFIG_YAML_PATH=DeepSeek-V3.yaml $0 multi"
}

main() {
    local mode="${1:-}"
    
    # Check required environment
    if [[ -z "${CONFIG_YAML_PATH:-}" ]]; then
        print_error "CONFIG_YAML_PATH environment variable not set"
    fi
    
    # Common setup
    check_npu_info
    check_and_config
    show_vllm_info
    show_triton_ascend_info
    
    # Change to workspace directory
    local workspace="${WORKSPACE:-$(pwd)}"
    if [[ -d "$workspace/vllm-ascend" ]]; then
        cd "$workspace/vllm-ascend"
    fi
    
    # Run appropriate test
    case "$mode" in
        single)
            run_single_node_test
            ;;
        multi)
            run_multi_node_test
            ;;
        *)
            # Auto-detect based on environment
            if [[ -n "${LWS_WORKER_INDEX:-}" ]] || [[ -n "${LWS_LEADER_ADDRESS:-}" ]]; then
                echo "Detected K8s LeaderWorkerSet environment, running multi-node test"
                run_multi_node_test
            else
                echo "Running single-node test (use 'multi' for multi-node)"
                run_single_node_test
            fi
            ;;
    esac
}

# Parse arguments
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

main "$@"
