# SPDX-License-Identifier: Apache-2.0
"""Inspect only this invocation's visible NPUs in a short-lived process."""

import argparse
import json
import os
import time
from pathlib import Path

from tests.e2e.schedule.config import MIN_NPU_FREE_RATIO, NPU_RELEASE_WAIT_SECONDS
from tests.e2e.schedule.deployment import parse_device_ids


def query_devices() -> dict:
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend in this child only

    count = torch.npu.device_count()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    pool = parse_device_ids(visible) if visible is not None else tuple(range(count))
    if not count or len(pool) != count:
        raise RuntimeError(f"NPU query reports {count} devices, visible pool is {pool}")
    devices = []
    for index, device_id in enumerate(pool):
        free, total = torch.npu.mem_get_info(index)
        if total <= 0:
            raise RuntimeError(f"NPU {device_id}: invalid total memory {total}")
        devices.append(
            {
                "device_id": device_id,
                "application_index": index,
                "free": free,
                "total": total,
                "free_ratio": free / total,
            }
        )
    return {"device_pool": list(pool), "devices": devices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + NPU_RELEASE_WAIT_SECONDS
    while True:
        result = query_devices()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if all(device["free_ratio"] >= MIN_NPU_FREE_RATIO for device in result["devices"]):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Visible NPUs did not reach free memory ratio {MIN_NPU_FREE_RATIO}")
        time.sleep(2)


if __name__ == "__main__":
    main()
