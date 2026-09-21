# SPDX-License-Identifier: Apache-2.0
"""Single entrypoint; config loading happens during execution, not collection."""

import signal

from tests.e2e.schedule.runtime import run_from_environment


def test_models():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        run_from_environment()
    finally:
        signal.signal(signal.SIGTERM, previous)
