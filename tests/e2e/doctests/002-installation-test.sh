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

INSTALLATION_DOC="docs/source/getting_started/installation/install_vllm_ascend.inc.md"
QUICKSTART_DOC="docs/source/getting_started/quick_start.md"
QUICKSTART_VERIFY_DOC="docs/source/getting_started/quick_start/ascend_image/verify_container.inc.md"
QUICKSTART_OFFLINE_DOC="docs/source/getting_started/quick_start/offline/qwen3-0.6b.inc.md"

COMMON_MARKERS=(
  "docs/source/getting_started/installation/cann_image/atlas-a2.inc.md|installation-cann-image-a2-ubuntu"
  "docs/source/getting_started/installation/cann_image/atlas-a2.inc.md|installation-cann-image-a2-openeuler"
  "${INSTALLATION_DOC}|installation-common-prerequisites-ubuntu"
  "${INSTALLATION_DOC}|installation-common-prerequisites-openeuler"
  "${INSTALLATION_DOC}|installation-post-standard"
)
PIP_MARKERS=(
  "${INSTALLATION_DOC}|installation-pip-install"
  "${INSTALLATION_DOC}|installation-pip-device-check"
)
UV_MARKERS=(
  "${INSTALLATION_DOC}|installation-uv-bootstrap"
  "${INSTALLATION_DOC}|installation-uv-install"
  "${INSTALLATION_DOC}|installation-uv-device-check"
)
SOURCE_MARKERS=("${INSTALLATION_DOC}|installation-source-install")
WORKER_PATH="tests/e2e/doctests/002-installation-test.sh"

INSTALL_WORK_DIR=""
VERIFY_RUNTIME_DIR=""

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

function select_installation() {
  local base="$1"
  local head="$2"
  local pip_changed=false
  local uv_changed=false
  local source_changed=false
  local common_changed=false
  local command_status

  if worker_changed "${base}" "${head}"; then
    printf '%s\n' pip uv source
    return
  else
    command_status=$?
  fi
  [[ ${command_status} -eq 1 ]] || return "${command_status}"
  if any_marker_changed "${base}" "${head}" "${PIP_MARKERS[@]}"; then
    pip_changed=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi
  if any_marker_changed "${base}" "${head}" "${UV_MARKERS[@]}"; then
    uv_changed=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi
  if any_marker_changed "${base}" "${head}" "${SOURCE_MARKERS[@]}"; then
    source_changed=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi
  if any_marker_changed "${base}" "${head}" "${COMMON_MARKERS[@]}"; then
    common_changed=true
  else
    command_status=$?
    [[ ${command_status} -eq 1 ]] || return "${command_status}"
  fi

  [[ "${pip_changed}" == true ]] && echo pip
  [[ "${uv_changed}" == true ]] && echo uv
  [[ "${source_changed}" == true ]] && echo source
  if [[ "${pip_changed}" == false && "${uv_changed}" == false && "${source_changed}" == false && "${common_changed}" == true ]]; then
    echo source
  fi
  return 0
}

function run_shell_block() {
  local path="$1"
  local marker="$2"
  local block
  block="$(python3 "${DOCTEST_HELPER}" block --expand "${path}" "${marker}")" || return $?
  source /dev/stdin <<<"${block}"
}

function detect_os() {
  [[ -r /etc/os-release ]] || _err "Cannot detect the operating system: /etc/os-release is missing."
  local id
  id="$(. /etc/os-release && echo "${ID,,}")"
  case "${id}" in
    ubuntu) echo ubuntu ;;
    openeuler) echo openeuler ;;
    *) _err "Unsupported operating system '${id}'. Expected Ubuntu or openEuler." ;;
  esac
}

function run_prerequisites() {
  local os_variant="$1"
  run_shell_block "${INSTALLATION_DOC}" "installation-common-prerequisites-${os_variant}"
}

function verify_installation_with_quickstart() {
  VERIFY_RUNTIME_DIR="$(mktemp -d)"
  export MODELSCOPE_HUB_FILE_LOCK=false
  export HF_HUB_OFFLINE=1
  run_shell_block "${QUICKSTART_DOC}" quickstart-modelscope
  run_shell_block "${QUICKSTART_VERIFY_DOC}" quickstart-container-verify
  python3 "${DOCTEST_HELPER}" block --expand \
    "${QUICKSTART_OFFLINE_DOC}" quickstart-standard-offline >"${VERIFY_RUNTIME_DIR}/example.py"
  (
    cd "${VERIFY_RUNTIME_DIR}"
    run_shell_block "${QUICKSTART_OFFLINE_DOC}" quickstart-standard-offline-run
  )
}

function run_source_installation() {
  INSTALL_WORK_DIR="$(mktemp -d)"
  (
    cd "${INSTALL_WORK_DIR}"
    run_shell_block "${INSTALLATION_DOC}" installation-source-install
  )
}

function cleanup_installation() {
  local exit_code=$?
  if [[ -n "${INSTALL_WORK_DIR}" && -d "${INSTALL_WORK_DIR}" ]]; then
    rm -rf "${INSTALL_WORK_DIR}"
  fi
  if [[ -n "${VERIFY_RUNTIME_DIR}" && -d "${VERIFY_RUNTIME_DIR}" ]]; then
    rm -rf "${VERIFY_RUNTIME_DIR}"
  fi
  return "${exit_code}"
}

function run_installation() {
  local method="$1"
  local os_variant
  source "${E2E_DIR}/common.sh"
  trap cleanup_installation EXIT
  os_variant="$(detect_os)"
  run_prerequisites "${os_variant}"
  case "${method}" in
    pip)
      run_shell_block "${INSTALLATION_DOC}" installation-pip-install
      run_shell_block "${INSTALLATION_DOC}" installation-pip-device-check
      ;;
    uv)
      run_shell_block "${INSTALLATION_DOC}" installation-uv-bootstrap
      run_shell_block "${INSTALLATION_DOC}" installation-uv-install
      run_shell_block "${INSTALLATION_DOC}" installation-uv-device-check
      ;;
    source)
      run_source_installation
      ;;
  esac
  run_shell_block "${INSTALLATION_DOC}" installation-post-standard
  verify_installation_with_quickstart
}

case "${1:-}" in
  select)
    [[ $# -eq 3 ]] || { echo "Usage: $0 select BASE HEAD" >&2; exit 1; }
    select_installation "$2" "$3"
    ;;
  run)
    [[ $# -eq 2 ]] || { echo "Usage: $0 run {pip|uv|source}" >&2; exit 1; }
    run_installation "$2"
    ;;
  *)
    echo "Usage: $0 {select BASE HEAD|run {pip|uv|source}}" >&2
    exit 1
    ;;
esac
