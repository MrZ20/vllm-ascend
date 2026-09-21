#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
fi
export REPO_ROOT
: "${CONFIG_YAML_PATH:?Set CONFIG_YAML_PATH to one V2 model YAML}"
test -f "$REPO_ROOT/tests/e2e/schedule/test_models.py"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Toolkit scripts can reference unset variables. Preserve inherited paths.
set +u
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
if [[ -n "${COORD_DIR:-}" ]]; then
    : "${RUN_ID:?Shared coordination requires the same RUN_ID on every node}"
fi
export RUN_ID="${RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export LOG_PREFIX="${LOG_PREFIX:-$(mktemp -d /tmp/schedule-v2.XXXXXX)}"
export SCHEDULE_CLEANUP_PROCESSES="${SCHEDULE_CLEANUP_PROCESSES:-0}"
NODE_LOG_DIR="$LOG_PREFIX/$RUN_ID/node-${LWS_WORKER_INDEX:-0}"
mkdir -p "$NODE_LOG_DIR"

case "$SCHEDULE_CLEANUP_PROCESSES" in
    0) echo "Startup process cleanup disabled; only owned test PGIDs will be stopped." ;;
    1)
        # Opt in only inside a dedicated PID-isolated test container. Never enable
        # this in hostPID/shared-PID/privileged validation containers.
        python -m tests.e2e.schedule.runtime --cleanup
        ;;
    *) echo 'SCHEDULE_CLEANUP_PROCESSES must be 0 or 1' >&2; exit 2 ;;
esac

# Called by the EXIT trap.
# shellcheck disable=SC2329
backup_logs() {
    if [[ -d "$HOME/ascend/log" ]]; then
        mkdir -p "$NODE_LOG_DIR/ascend"
        timeout 20 cp -r "$HOME/ascend/log/." "$NODE_LOG_DIR/ascend/" || true
    fi
}
trap backup_logs EXIT

cd "$REPO_ROOT"
# --noconftest avoids the V1 import-time setup and NPU state in its pytest parent.
# Keep the pytest PID so container termination reaches its scoped cleanup handler.
PIPE_DIR="$(mktemp -d "$NODE_LOG_DIR/.output.XXXXXX")"
mkfifo "$PIPE_DIR/stdout"
tee "$NODE_LOG_DIR/pytest.log" < "$PIPE_DIR/stdout" &
TEE_PID=$!
python -m pytest --noconftest -sv tests/e2e/schedule/test_models.py > "$PIPE_DIR/stdout" 2>&1 &
TEST_PID=$!
# Called by the TERM/INT trap.
# shellcheck disable=SC2329
forward_termination() {
    kill -TERM "$TEST_PID" 2>/dev/null || true
    wait "$TEST_PID" || true
    wait "$TEE_PID" || true
    rm -f "$PIPE_DIR/stdout"
    rmdir "$PIPE_DIR"
    exit 143
}
trap forward_termination TERM INT
set +e
wait "$TEST_PID"
TEST_STATUS=$?
trap - TERM INT
wait "$TEE_PID"
TEE_STATUS=$?
set -e
rm -f "$PIPE_DIR/stdout"
rmdir "$PIPE_DIR"
if [[ "$TEST_STATUS" -ne 0 ]]; then
    exit "$TEST_STATUS"
fi
exit "$TEE_STATUS"
