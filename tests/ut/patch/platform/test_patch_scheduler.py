from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _dummy_scheduler(block_size: int = 16):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=False,
    )


def _dummy_request(num_tokens: int = 128):
    return SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
    )


def test_mamba_block_aligned_split_accepts_common_prefix_limit():
    result = Scheduler._mamba_block_aligned_split(
        _dummy_scheduler(),
        _dummy_request(),
        64,
        0,
        0,
        32,
    )

    assert result == 32
