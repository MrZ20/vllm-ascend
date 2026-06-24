#!/bin/bash
set -euo pipefail

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

INTERNAL_DP_TEST_PATH="tests/e2e/schedule/scripts/multi_node/internal_dp/test_multi_node.py"
EXTERNAL_DP_TEST_PATH="tests/e2e/schedule/scripts/multi_node/external_dp/test_external_dp.py"

if [ -z "${MULTI_NODE_TEST_PATH:-}" ]; then
    if [[ "${CONFIG_YAML_PATH:-}" == *"external_dp"* ]]; then
        MULTI_NODE_TEST_PATH="$EXTERNAL_DP_TEST_PATH"
    else
        MULTI_NODE_TEST_PATH="$INTERNAL_DP_TEST_PATH"
    fi
fi

# Configuration
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# cann and atb environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh

set +eu
source /usr/local/Ascend/nnal/atb/set_env.sh
set -eu

# Home path for aisbench
export BENCHMARK_HOME=${WORKSPACE}/vllm-ascend/benchmark

# Logging configurations
export VLLM_LOGGING_LEVEL="INFO"
# Reduce glog verbosity for mooncake
export GLOG_minloglevel=1
# Set transformers to offline mode to avoid downloading models during tests
export HF_HUB_OFFLINE="1"
# Default is 600s
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_failure() {
    echo -e "${RED}${FAIL_TAG:-test_failed} ✗ ERROR: $1${NC}"
    exit 1
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages and exit
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

show_vllm_info() {
    cd "$WORKSPACE"
    echo "Installed vLLM-related Python packages:"
    pip list | grep vllm || echo "No vllm packages found."

    echo ""
    echo "============================"
    echo "vLLM Git information"
    echo "============================"
    cd vllm
    if [ -d .git ]; then
    echo "Branch:      $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit hash: $(git rev-parse HEAD)"
    echo "Author:      $(git log -1 --pretty=format:'%an <%ae>')"
    echo "Date:        $(git log -1 --pretty=format:'%ad' --date=iso)"
    echo "Message:     $(git log -1 --pretty=format:'%s')"
    echo "Tags:        $(git tag --points-at HEAD || echo 'None')"
    echo "Remote:      $(git remote -v | head -n1)"
    echo ""
    else
    echo "No .git directory found in vllm"
    fi
    cd ..

    echo ""
    echo "============================"
    echo "vLLM-Ascend Git information"
    echo "============================"
    cd vllm-ascend
    if [ -d .git ]; then
    echo "Branch:      $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit hash: $(git rev-parse HEAD)"
    echo "Author:      $(git log -1 --pretty=format:'%an <%ae>')"
    echo "Date:        $(git log -1 --pretty=format:'%ad' --date=iso)"
    echo "Message:     $(git log -1 --pretty=format:'%s')"
    echo "Tags:        $(git tag --points-at HEAD || echo 'None')"
    echo "Remote:      $(git remote -v | head -n1)"
    echo ""
    else
    echo "No .git directory found in vllm-ascend"
    fi
    cd ..
}

check_npu_info() {
    echo "====> Check NPU info"
    npu-smi info
    cat "/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info"
}

check_and_config() {
    echo "====> Configure mirrors and git proxy"
    git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi"
}

install_extra_components() {
    echo "====> Installing extra components for DeepSeek-v3.2-exp-bf16"

    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-sfa-linux.aarch64.run; then
        echo "Failed to download CANN-custom_ops-sfa-linux.aarch64.run"
        return 1
    fi
    chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
    ./CANN-custom_ops-sfa-linux.aarch64.run --quiet

    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl; then
        echo "Failed to download custom_ops wheel"
        return 1
    fi
    pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl

    export ASCEND_CUSTOM_OPP_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    rm -f CANN-custom_ops-sfa-linux.aarch64.run \
          custom_ops-1.0-cp311-cp311-linux_aarch64.whl
    echo "====> Extra components installation completed"
}

show_triton_ascend_info() {
    echo "====> Check triton ascend info"
    clang -v
    which bishengir-compile
    pip show triton-ascend
}

kill_npu_processes() {
  pgrep python3 | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  sleep 4
}

run_tests_with_log() {
    set +e
    kill_npu_processes
    echo "====> Run pytest entry: $MULTI_NODE_TEST_PATH"
    pytest -sv --show-capture=no "$MULTI_NODE_TEST_PATH"
    ret=$?
    set -e
    if [ "$LWS_WORKER_INDEX" -eq 0 ]; then
        if [ $ret -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_failure "Some tests failed, please check the error stack above for details. \
If this is insufficient to pinpoint the error, please download and review the logs of all other nodes from the job's summary."
        fi
    fi
}

clear_logs() {
    print_section "Clearing logs from previous runs"
    rm -fr "$HOME/ascend/log" || true
}

backup_ascend_logs() {
    if [ -n "${LOG_PREFIX:-}" ]; then
        local dest="${LOG_PREFIX}/node_${LWS_WORKER_INDEX:-unknown}_plogs"
        mkdir -p "$dest"
        cp -r /root/ascend/log/. "$dest/" 2>/dev/null || true
        echo "Ascend logs backed up to $dest"
    fi
}

main() {
    trap backup_ascend_logs EXIT
    check_npu_info
    clear_logs
    check_and_config
    show_vllm_info
    show_triton_ascend_info
    if [[ "$CONFIG_YAML_PATH" == *"DeepSeek-V3_2-Exp-bf16.yaml" ]]; then
        install_extra_components
    fi
    cd "$WORKSPACE/vllm-ascend"
    run_tests_with_log
}

main "$@"
