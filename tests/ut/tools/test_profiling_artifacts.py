import json
import sys
import tarfile
import types
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from tools.profiling import parse, storage


class MemoryObs:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = objects or {}

    def upload_file(self, source, _bucket, key, Callback=None, **_kwargs):
        data = Path(source).read_bytes()
        self.objects[key] = data
        if Callback:
            Callback(len(data))

    def download_file(self, bucket, key, destination, Callback=None, **_kwargs):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": key}}, "GetObject")
        data = self.objects[key]
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if Callback:
            Callback(len(data))

    def head_object(self, _Bucket=None, Key=None, **_kwargs):
        return {"ContentLength": len(self.objects[Key])}


def test_storage_upload_helpers_verify_remote_size(tmp_path: Path):
    client = MemoryObs()
    archive = tmp_path / "case" / "server.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"archive")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    record = {"archive": "case/server.tar.gz"}

    storage.upload_archive(client, object(), tmp_path, "job/raw", "bucket", record, "raw")
    storage.upload_json(client, manifest, "bucket", "job/raw/profile_manifest.json")

    assert storage.GZIP_COMPRESSION_LEVEL == 1
    assert record["obs_url"] == "obs://bucket/job/raw/case/server.tar.gz"
    assert client.objects["job/raw/case/server.tar.gz"] == b"archive"
    assert client.objects["job/raw/profile_manifest.json"] == b"{}"

    client.head_object = lambda **_kwargs: {"ContentLength": 0}
    with pytest.raises(RuntimeError, match="OBS size mismatch"):
        storage.upload_json(client, manifest, "bucket", "job/raw/profile_manifest.json")


def test_plan_artifact_shards_uses_manifest_node_indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = {
        "cases": [
            {"targets": [{"node_index": 2}, {"node_index": 0}]},
            {"targets": [{"node_index": 2}]},
        ]
    }
    client = MemoryObs({"job/raw/profile_manifest.json": json.dumps(manifest).encode()})
    monkeypatch.setattr(parse, "create_obs_client", lambda *_: client)
    monkeypatch.setattr(parse, "create_transfer_config", object)

    assert parse.plan_artifact_shards(tmp_path, "job", "bucket", "endpoint", "region") == [0, 2]


def test_parse_artifacts_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source" / "server" / "rank0_ascend_pt"
    source.mkdir(parents=True)
    (source / "profiler_info.json").write_text("{}")
    archive = tmp_path / "server.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source.parent, arcname="server")

    raw_manifest = {
        "status": "success",
        "cases": [
            {
                "case": "case-a",
                "status": "success",
                "targets": [
                    {
                        "name": "server",
                        "node_index": 0,
                        "archive": "case-a/server.tar.gz",
                        "obs_url": "obs://bucket/job/raw/case-a/server.tar.gz",
                        "size_bytes": archive.stat().st_size,
                        "output": "raw",
                    }
                ],
            }
        ],
    }
    client = MemoryObs(
        {
            "job/raw/profile_manifest.json": json.dumps(raw_manifest).encode(),
            "job/raw/case-a/server.tar.gz": archive.read_bytes(),
        }
    )
    monkeypatch.setattr(parse, "create_obs_client", lambda *_: client)
    monkeypatch.setattr(parse, "create_transfer_config", object)

    def analyse(trace_dir: str, max_process_number: int) -> None:
        assert max_process_number == 4
        output = Path(trace_dir) / "ASCEND_PROFILER_OUTPUT"
        output.mkdir()
        (output / "analyse.done").touch()
        (output / "trace_view.json").write_text("{}")

    profiler_module = types.ModuleType("torch_npu.profiler.profiler")
    profiler_module.analyse = analyse
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", types.ModuleType("torch_npu.profiler"))
    monkeypatch.setitem(sys.modules, "torch_npu.profiler.profiler", profiler_module)

    result = parse.parse_artifacts(tmp_path / "parse", "job", "bucket", "endpoint", "region", 4)

    target = result["cases"][0]["targets"][0]
    assert result["status"] == "success"
    assert target["raw_obs_url"] == "obs://bucket/job/raw/case-a/server.tar.gz"
    assert target["ranks"][0]["output"] == "parsed"
    assert target["ranks"][0]["obs_url"].startswith("obs://bucket/job/parsed/")
    assert "job/parsed/profile_manifest.json" in client.objects
    assert "job/raw/case-a/server.tar.gz" in client.objects


def test_parse_artifacts_skips_missing_raw_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = MemoryObs()
    monkeypatch.setattr(parse, "create_obs_client", lambda *_: client)
    monkeypatch.setattr(parse, "create_transfer_config", object)

    result = parse.parse_artifacts(tmp_path, "job", "bucket", "endpoint", "region")

    assert result == {"status": "skipped", "reason": "raw_manifest_missing"}
