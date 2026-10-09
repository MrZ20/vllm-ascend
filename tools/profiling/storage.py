"""Shared OBS transfer and storage utilities for profiling artifacts."""

import argparse
import os
import threading
from pathlib import Path

DEFAULT_BUCKET = "obs-guiiyang1-ascend-test"
DEFAULT_ENDPOINT = "https://obs.cn-southwest-2.myhuaweicloud.com"
DEFAULT_REGION = "cn-southwest-2"
GZIP_COMPRESSION_LEVEL = 1


# =============================================================================
# Local and OBS transfer helpers
# =============================================================================


def directory_size_bytes(path: Path) -> int:
    """Return the total size in bytes of regular files below ``path``."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.exists() else 0


def create_obs_client(endpoint: str, region: str):
    """Create the S3-compatible client used to access Huawei OBS."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "virtual", "payload_signing_enabled": False},
        ),
    )


def create_transfer_config():
    """Return bounded multipart-transfer settings shared by uploads/downloads."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=64 * 1024**2,
        multipart_chunksize=64 * 1024**2,
        max_concurrency=4,
        use_threads=True,
    )


def make_transfer_progress_callback(label: str, total_bytes: int):
    """Build a thread-safe callback that reports transfer progress by decile."""
    transferred = 0
    reported = 0
    lock = threading.Lock()

    def callback(amount: int) -> None:
        """Accumulate one transfer chunk and print newly crossed deciles."""
        nonlocal transferred, reported
        with lock:
            transferred += amount
            percent = min(100, transferred * 100 // total_bytes)
            step = percent // 10
            if step > reported:
                reported = step
                print(f"[Profiling] {label}: {percent}% ({transferred}/{total_bytes} bytes)", flush=True)

    return callback


def upload_archive(
    client,
    transfer_config,
    root: Path,
    prefix: str,
    bucket: str,
    target_record: dict,
    stage: str,
) -> None:
    """Upload one archive, verify its remote size, and update its manifest record."""
    archive_path = root / target_record["archive"]
    object_key = f"{prefix}/{target_record['archive']}"
    size_bytes = archive_path.stat().st_size
    print(f"[Profiling] Upload {stage} obs://{bucket}/{object_key}: {size_bytes} bytes", flush=True)
    client.upload_file(
        str(archive_path),
        bucket,
        object_key,
        ExtraArgs={"ContentType": "application/gzip"},
        Config=transfer_config,
        Callback=make_transfer_progress_callback(f"Upload {stage} {target_record['archive']}", size_bytes),
    )
    uploaded_size = client.head_object(Bucket=bucket, Key=object_key)["ContentLength"]
    if uploaded_size != size_bytes:
        raise RuntimeError(f"OBS size mismatch: local={size_bytes}, remote={uploaded_size}")
    target_record["obs_url"] = f"obs://{bucket}/{object_key}"


def upload_json(client, source_path: Path, bucket: str, object_key: str) -> None:
    """Upload a JSON artifact and fail if the stored byte size differs."""
    client.upload_file(str(source_path), bucket, object_key, ExtraArgs={"ContentType": "application/json"})
    if client.head_object(Bucket=bucket, Key=object_key)["ContentLength"] != source_path.stat().st_size:
        raise RuntimeError(f"OBS size mismatch: {object_key}")


# =============================================================================
# OBS diagnostics
# =============================================================================


def log_obs_storage(bucket: str, endpoint: str) -> None:
    """Print a best-effort OBS capacity snapshot; accounting may be delayed."""
    try:
        from obs import ObsClient

        client = ObsClient(
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            security_token=os.getenv("AWS_SESSION_TOKEN"),
            server=endpoint,
        )
        try:
            storage = client.getBucketStorageInfo(bucket)
            quota = client.getBucketQuota(bucket)
            if storage.status >= 300 or quota.status >= 300:
                raise RuntimeError(f"storage status={storage.status}, quota status={quota.status}")
            used_bytes = int(storage.body.size)
            quota_bytes = int(quota.body.quota)
            remaining = f"{max(0, quota_bytes - used_bytes)} bytes (estimated)" if quota_bytes else "unlimited quota"
            print(
                f"[Profiling] OBS capacity {bucket}: used={used_bytes} bytes, quota={quota_bytes} bytes, "
                f"remaining={remaining}; usage is delayed",
                flush=True,
            )
        finally:
            client.close()
    except Exception as exc:
        print(f"[Profiling] OBS capacity unavailable: {exc}", flush=True)


# =============================================================================
# Command-line entry point
# =============================================================================


def main() -> None:
    """Run profiling storage diagnostics from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status"])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    log_obs_storage(args.bucket, args.endpoint)


if __name__ == "__main__":
    main()
