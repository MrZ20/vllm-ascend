#!/bin/bash

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
# shellcheck disable=SC1090,SC1091

set -Eeuo pipefail

DOCTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_DIR="$(cd "${DOCTEST_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${E2E_DIR}/../.." && pwd)"
DOCTEST_HELPER="${DOCTEST_DIR}/doctest_helper.py"

QUICKSTART_DOC="docs/source/getting_started/quick_start.md"
QUICKSTART_VERIFY_DOC="docs/source/getting_started/quick_start/ascend_image/verify_container.inc.md"
A2_OFFLINE_DOC="docs/source/getting_started/quick_start/offline/qwen3-0.6b.inc.md"
A2_ONLINE_DOC="docs/source/getting_started/quick_start/online/qwen3-0.6b.inc.md"
DUO_OFFLINE_DOC="docs/source/getting_started/quick_start/offline/qwen3-0.6b-310p.inc.md"
DUO_ONLINE_DOC="docs/source/getting_started/quick_start/online/qwen3-0.6b-310p.inc.md"

COMMON_MARKERS=(
  "${QUICKSTART_DOC}|quickstart-modelscope"
  "${QUICKSTART_VERIFY_DOC}|quickstart-container-verify"
)
A2_MARKERS=(
  "docs/source/getting_started/quick_start/ascend_image/atlas-a2.inc.md|quickstart-image-a2-ubuntu"
  "docs/source/getting_started/quick_start/ascend_image/atlas-a2.inc.md|quickstart-image-a2-openeuler"
  "${A2_OFFLINE_DOC}|quickstart-standard-offline"
  "${A2_OFFLINE_DOC}|quickstart-standard-offline-run"
  "${A2_ONLINE_DOC}|quickstart-standard-online-serve"
  "${A2_ONLINE_DOC}|quickstart-standard-online-model-list"
  "${A2_ONLINE_DOC}|quickstart-standard-online-completion"
  "${A2_ONLINE_DOC}|quickstart-standard-online-stop"
)
DUO_MARKERS=(
  "docs/source/getting_started/quick_start/ascend_image/atlas-300i-duo.inc.md|quickstart-image-300i-duo-ubuntu"
  "docs/source/getting_started/quick_start/ascend_image/atlas-300i-duo.inc.md|quickstart-image-300i-duo-openeuler"
  "${DUO_OFFLINE_DOC}|quickstart-300i-duo-offline"
  "${DUO_OFFLINE_DOC}|quickstart-300i-duo-offline-run"
  "${DUO_ONLINE_DOC}|quickstart-300i-duo-online-serve"
  "${DUO_ONLINE_DOC}|quickstart-300i-duo-online-model-list"
  "${DUO_ONLINE_DOC}|quickstart-300i-duo-online-completion"
  "${DUO_ONLINE_DOC}|quickstart-300i-duo-online-stop"
)
WORKER_PATH="tests/e2e/doctests/001-quickstart-test.sh"

VLLM_PID=""
RUNTIME_DIR=""

function any_marker_changed() {
  python3 "${DOCTEST_HELPER}" changed-any "$@"
}

function worker_changed() {
  local base="$1"
  local head="$2"
  local command_status
  if git -C "${REPO_ROOT}" diff --quiet "${base}" "${head}" -- "${WORKER_PATH}"; then
    return 1
  else
    command_status=$?
  fi
  [[ ${command_status} -eq 1 ]] && return 0
  return "${command_status}"
}

function select_quickstart() {
  local base="$1"
  local head="$2"
  local run_a2=false
  local run_310p=false
  local command_status

  if worker_changed "${base}" "${head}"; then
    run_a2=true
    run_310p=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi
  if any_marker_changed "${base}" "${head}" "${COMMON_MARKERS[@]}"; then
    run_a2=true
    run_310p=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi
  if any_marker_changed "${base}" "${head}" "${A2_MARKERS[@]}"; then
    run_a2=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi
  if any_marker_changed "${base}" "${head}" "${DUO_MARKERS[@]}"; then
    run_310p=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi

  [[ "${run_a2}" == true ]] && echo a2
  [[ "${run_310p}" == true ]] && echo 310p
  return 0
}

function run_shell_block() {
  local path="$1"
  local marker="$2"
  local block
  block="$(python3 "${DOCTEST_HELPER}" block --expand "${path}" "${marker}")" || return $?
  source /dev/stdin <<<"${block}"
}

function run_offline() {
  local path="$1"
  local python_marker="$2"
  local run_marker="$3"
  python3 "${DOCTEST_HELPER}" block --expand "${path}" "${python_marker}" >"${RUNTIME_DIR}/example.py"
  (
    cd "${RUNTIME_DIR}"
    run_shell_block "${path}" "${run_marker}"
  )
}

function run_online() {
  local path="$1"
  local serve_marker="$2"
  local model_list_marker="$3"
  local completion_marker="$4"
  local stop_marker="$5"
  pushd "${RUNTIME_DIR}" >/dev/null
  run_shell_block "${path}" "${serve_marker}"
  VLLM_PID="$!"
  popd >/dev/null
  wait_url_ready "vllm serve" "localhost:8000/v1/models"
  run_shell_block "${path}" "${model_list_marker}"
  run_shell_block "${path}" "${completion_marker}"
  run_shell_block "${path}" "${stop_marker}"
  wait_for_exit "${VLLM_PID}"
  VLLM_PID=""
}

function cleanup_quickstart() {
  local exit_code=$?
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -2 "${VLLM_PID}" 2>/dev/null || true
    wait_for_exit "${VLLM_PID}" || true
  fi
  if [[ -n "${RUNTIME_DIR}" && -d "${RUNTIME_DIR}" ]]; then
    rm -rf "${RUNTIME_DIR}"
  fi
  return "${exit_code}"
}

function run_quickstart() {
  local device="$1"
  local offline_doc
  local online_doc
  local marker_prefix
  source "${E2E_DIR}/common.sh"
  export MODELSCOPE_HUB_FILE_LOCK=false
  export HF_HUB_OFFLINE=1
  trap cleanup_quickstart EXIT
  RUNTIME_DIR="$(mktemp -d)"

  case "${device}" in
    a2)
      offline_doc="${A2_OFFLINE_DOC}"
      online_doc="${A2_ONLINE_DOC}"
      marker_prefix=quickstart-standard
      ;;
    310p)
      offline_doc="${DUO_OFFLINE_DOC}"
      online_doc="${DUO_ONLINE_DOC}"
      marker_prefix=quickstart-300i-duo
      ;;
  esac

  run_shell_block "${QUICKSTART_DOC}" quickstart-modelscope
  run_shell_block "${QUICKSTART_VERIFY_DOC}" quickstart-container-verify
  run_offline "${offline_doc}" "${marker_prefix}-offline" "${marker_prefix}-offline-run"
  run_online \
    "${online_doc}" \
    "${marker_prefix}-online-serve" \
    "${marker_prefix}-online-model-list" \
    "${marker_prefix}-online-completion" \
    "${marker_prefix}-online-stop"
}

case "${1:-}" in
  select)
    [[ $# -eq 3 ]] || { echo "Usage: $0 select BASE HEAD" >&2; exit 1; }
    select_quickstart "$2" "$3"
    ;;
  run)
    [[ $# -eq 2 ]] || { echo "Usage: $0 run {a2|310p}" >&2; exit 1; }
    run_quickstart "$2"
    ;;
  *)
    echo "Usage: $0 {select BASE HEAD|run {a2|310p}}" >&2
    exit 1
    ;;
esac
