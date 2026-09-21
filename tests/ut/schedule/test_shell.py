# SPDX-License-Identifier: Apache-2.0
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def shell_environment(tmp_path, mode):
    command = tmp_path / "bin/python"
    command.parent.mkdir()
    command.write_text(f"""#!{sys.executable}
import os, signal, time
if os.environ['STUB_MODE'] == 'exit':
    print('expected test failure', flush=True)
    raise SystemExit(23)
def terminate(signum, frame):
    print('owned child cleanup', flush=True)
    raise SystemExit(143)
signal.signal(signal.SIGTERM, terminate)
print('child ready', flush=True)
time.sleep(30)
""")
    command.chmod(0o755)
    return {
        **os.environ,
        "PATH": str(command.parent) + os.pathsep + os.environ["PATH"],
        "REPO_ROOT": str(ROOT),
        "RUN_ID": "unit",
        "CONFIG_YAML_PATH": "unused-by-stub.yaml",
        "LOG_PREFIX": str(tmp_path / "logs"),
        "SCHEDULE_CLEANUP_PROCESSES": "0",
        "STUB_MODE": mode,
        "LWS_WORKER_INDEX": "0",
    }


def test_shell_preserves_pytest_failure_and_log(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "tests/e2e/schedule/scripts/run.sh")],
        env=shell_environment(tmp_path, "exit"),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert "expected test failure" in (tmp_path / "logs/unit/node-0/pytest.log").read_text()


def test_shell_forwards_termination_to_owned_child(tmp_path):
    process = subprocess.Popen(
        ["bash", str(ROOT / "tests/e2e/schedule/scripts/run.sh")],
        env=shell_environment(tmp_path, "signal"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output = b""
        while b"child ready" not in output:
            assert select.select([process.stdout], [], [], 5)[0], output
            chunk = os.read(process.stdout.fileno(), 4096)
            assert chunk, output
            output += chunk
        process.send_signal(signal.SIGTERM)
        remaining, _ = process.communicate(timeout=5)
        assert process.returncode == 143, output + remaining
        assert b"owned child cleanup" in output + remaining
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
