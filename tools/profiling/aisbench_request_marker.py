"""Mark the first AISBench request used to schedule a profiling window.

This module stays separate because importing the adapter requires the optional
AISBench package, while the rest of the profiling workflow does not.
"""

import os
from pathlib import Path

from ais_bench.benchmark.models import VLLMCustomAPIChat

from tools.profiling.capture import record_first_request


class ProfiledVLLMCustomAPIChat(VLLMCustomAPIChat):
    """AISBench model adapter that records the first dispatched request."""

    async def stream_infer(self, request_body, output):
        """Mark the first request, then delegate streaming inference unchanged."""
        record_first_request(Path(os.environ["ASCEND_PROFILE_REQUEST_MARKER"]))
        return await super().stream_infer(request_body, output)

    async def text_infer(self, request_body, output):
        """Mark the first request, then delegate non-streaming inference unchanged."""
        record_first_request(Path(os.environ["ASCEND_PROFILE_REQUEST_MARKER"]))
        return await super().text_infer(request_body, output)
