import json
import threading
from pathlib import Path

import pytest

from tools.profiling import capture
from tools.profiling.capture import (
    ProfileController,
    ProfileSpec,
    ServeInstance,
    ServeManifest,
    collect_raw_artifacts,
    configure_profiler_command,
    record_first_request,
    upload_raw_artifacts,
)

PROFILE_ENV_NAMES = (
    "ASCEND_PROFILE_ENABLED",
    "ASCEND_PROFILE_START_AFTER",
    "ASCEND_PROFILE_DURATION",
    "ASCEND_PROFILE_WITH_STACK",
    "ASCEND_PROFILE_SCOPE",
    "ASCEND_PROFILE_OUTPUT",
    "ASCEND_PROFILE_CASES",
)


def test_profile_spec_reads_environment_and_filters_cases(monkeypatch: pytest.MonkeyPatch):
    for name in PROFILE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert ProfileSpec.from_env() == ProfileSpec()
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", "FaLsE")
    assert ProfileSpec.from_env() == ProfileSpec()

    values = {
        "ASCEND_PROFILE_ENABLED": "TrUe",
        "ASCEND_PROFILE_START_AFTER": "0",
        "ASCEND_PROFILE_DURATION": "12",
        "ASCEND_PROFILE_WITH_STACK": "TRUE",
        "ASCEND_PROFILE_SCOPE": "all",
        "ASCEND_PROFILE_OUTPUT": "raw",
        "ASCEND_PROFILE_CASES": "case-a, case-b,",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    spec = ProfileSpec.from_env()

    assert spec == ProfileSpec(True, 0, 12, True, "all", "raw", ("case-a", "case-b"))
    assert spec.includes_case("case-a", "performance")
    assert not spec.includes_case("case-c", "performance")
    assert not spec.includes_case("case-a", "accuracy")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ASCEND_PROFILE_ENABLED", "yes"),
        ("ASCEND_PROFILE_SCOPE", "first"),
        ("ASCEND_PROFILE_DURATION", "0"),
    ],
)
def test_profile_spec_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, name: str, value: str):
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", "true")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        ProfileSpec.from_env()


def test_serve_manifest_round_trip_and_representative_selection(tmp_path: Path):
    instances = (
        ServeInstance.from_endpoint("prefill/1", "http://prefill-1", "prefill", 1, root=tmp_path),
        ServeInstance.from_endpoint("prefill-0", "http://prefill-0", "prefill", 0, root=tmp_path),
        ServeInstance.from_endpoint("decode-0", "http://decode-0", "decode", 0, root=tmp_path),
    )
    path = tmp_path / "serve_manifest.json"
    ServeManifest(instances).write(path)

    manifest = ServeManifest.read(path)

    assert manifest == ServeManifest(instances)
    assert [target.name for target in manifest.select_targets("representative")] == ["prefill-0", "decode-0"]
    assert manifest.select_targets("all") == instances
    assert Path(instances[0].raw_trace_dir).name == "prefill_1"


def test_configure_profiler_command_preserves_unowned_options(tmp_path: Path):
    instance = ServeInstance.from_endpoint("server", "http://server", root=tmp_path)
    command = ["vllm", "serve", "model", "--profiler-config", '{"custom": 7, "max_iterations": 3}']

    updated = configure_profiler_command(command, instance, ProfileSpec(with_stack=True))
    profiler_config = json.loads(updated[updated.index("--profiler-config") + 1])

    assert profiler_config == {
        "custom": 7,
        "max_iterations": 0,
        "profiler": "torch",
        "torch_profiler_dir": instance.raw_trace_dir,
        "torch_profiler_with_stack": True,
        "ignore_frontend": True,
    }


def test_profile_controller_profiles_each_endpoint_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    marker = tmp_path / "first_request"
    record_first_request(marker)
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    stopped = threading.Event()

    class Client:
        def __init__(self):
            self.started: list[str] = []
            self.stopped: list[str] = []

        def start_profile(self, endpoint: str) -> None:
            self.started.append(endpoint)

        def stop_profile(self, endpoint: str) -> None:
            self.stopped.append(endpoint)
            stopped.set()

    client = Client()
    targets = (
        ServeInstance("rank-0", "http://server", str(tmp_path / "rank-0")),
        ServeInstance("rank-1", "http://server", str(tmp_path / "rank-1"), dp_rank=1),
    )
    controller = ProfileController(ProfileSpec(True, 0, 0.01), targets, marker, client)

    controller.start()
    assert stopped.wait(1)
    result = controller.finish()

    assert result["status"] == "success"
    assert result["reason"] == "duration_reached"
    assert client.started == ["http://server"]
    assert client.stopped == ["http://server"]


def test_collect_raw_artifacts_records_partial_results(tmp_path: Path):
    good_dir = tmp_path / "raw" / "good"
    good_dir.mkdir(parents=True)
    (good_dir / "trace.bin").write_bytes(b"trace")
    targets = (
        ServeInstance("good", "http://good", str(good_dir)),
        ServeInstance("missing", "http://missing", str(tmp_path / "raw" / "missing")),
    )

    collect_raw_artifacts(
        tmp_path,
        "case/name",
        targets,
        {"status": "success", "reason": "duration_reached", "actual_duration": 8, "targets": {}},
        ProfileSpec(enabled=True),
    )

    manifest = json.loads((tmp_path / "profile_manifest.json").read_text())
    good, missing = manifest["cases"][0]["targets"]
    assert capture.RAW_COMPRESSION_CONCURRENCY == 4
    assert manifest["status"] == "partial"
    assert good["archive"] == "case_name/good.tar.gz"
    assert (tmp_path / good["archive"]).is_file()
    assert not any(good_dir.iterdir())
    assert "artifact_error" in missing


def test_upload_raw_artifacts_publishes_manifest_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = tmp_path / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "success",
                "cases": [
                    {
                        "targets": [
                            {"name": "good", "archive": "case/good.tar.gz"},
                            {"name": "bad", "archive": "case/bad.tar.gz"},
                        ]
                    }
                ],
            }
        )
    )
    events: list[str] = []

    def upload_archive(_client, _config, _root, prefix, bucket, record, _stage):
        events.append(record["name"])
        if record["name"] == "bad":
            raise RuntimeError("upload failed")
        record["obs_url"] = f"obs://{bucket}/{prefix}/{record['archive']}"

    monkeypatch.setattr(capture, "create_obs_client", lambda *_: object())
    monkeypatch.setattr(capture, "create_transfer_config", object)
    monkeypatch.setattr(capture, "upload_archive", upload_archive)
    monkeypatch.setattr(capture, "upload_json", lambda *_: events.append("manifest"))

    manifest = upload_raw_artifacts(tmp_path, "/job/", "bucket", "endpoint", "region")

    good, bad = manifest["cases"][0]["targets"]
    assert manifest["status"] == "partial"
    assert good["obs_url"] == "obs://bucket/job/raw/case/good.tar.gz"
    assert bad["upload_error"] == "upload failed"
    assert events[-1] == "manifest"
