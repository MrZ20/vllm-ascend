"""追踪 Qwen3.5-35B-A3B W8A8 e2e 推理路径。

这是教学插桩脚本，不是生产代码。它会在进程启动时 monkey-patch
一组选中的函数，并打印精简的进入/退出摘要。
必须以真实文件方式执行，因为 vLLM 使用 multiprocessing 的 "spawn"。
"""

from __future__ import annotations

import functools
import importlib
import inspect
import os
import sys
from collections import defaultdict
from typing import Any, Callable


TRACE_LIMIT_PER_FUNCTION = int(os.getenv("QWEN_TRACE_LIMIT_PER_FUNCTION", "12"))
TEXT_LIMIT = 160
_call_counts: dict[str, int] = defaultdict(int)
_patched: set[str] = set()


def _rank_prefix() -> str:
    pid = os.getpid()
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK") or os.getenv("VLLM_DP_RANK") or "-"
    return f"[TRACE pid={pid} rank={rank}]"


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _tensor_summary(value: Any) -> str | None:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return (
            "Tensor("
            f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
            ")"
        )
    return None


def _array_summary(value: Any) -> str | None:
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        preview = value.reshape(-1)[:8].tolist() if value.size else []
        return f"ndarray(shape={value.shape}, dtype={value.dtype}, head={preview})"
    return None


def _sampling_params_summary(value: Any) -> str | None:
    if value.__class__.__name__ != "SamplingParams":
        return None
    fields = {
        "temperature": getattr(value, "temperature", None),
        "max_tokens": getattr(value, "max_tokens", None),
        "top_p": getattr(value, "top_p", None),
        "top_k": getattr(value, "top_k", None),
        "output_kind": getattr(value, "output_kind", None),
    }
    return f"SamplingParams({fields})"


def _request_output_summary(value: Any) -> str | None:
    if value.__class__.__name__ != "RequestOutput":
        return None
    prompt = getattr(value, "prompt", None)
    prompt_ids = getattr(value, "prompt_token_ids", None)
    outputs = getattr(value, "outputs", None) or []
    samples: list[str] = []
    for sample in list(outputs)[:2]:
        text = getattr(sample, "text", None)
        token_ids = getattr(sample, "token_ids", None)
        samples.append(
            "Sample("
            f"text={_clip(repr(text))}, "
            f"token_ids_head={list(token_ids)[:12] if token_ids is not None else None}"
            ")"
        )
    return (
        "RequestOutput("
        f"prompt={_clip(repr(prompt))}, "
        f"prompt_ids_head={list(prompt_ids)[:12] if prompt_ids is not None else None}, "
        f"outputs={samples}"
        ")"
    )


def _object_summary(value: Any) -> str:
    tensor = _tensor_summary(value)
    if tensor is not None:
        return tensor

    array = _array_summary(value)
    if array is not None:
        return array

    sampling = _sampling_params_summary(value)
    if sampling is not None:
        return sampling

    req_output = _request_output_summary(value)
    if req_output is not None:
        return req_output

    if value is None or isinstance(value, bool | int | float | str):
        return _clip(repr(value))

    if isinstance(value, tuple):
        return "(" + ", ".join(_object_summary(item) for item in value[:6]) + (", ..." if len(value) > 6 else "") + ")"

    if isinstance(value, list):
        preview = ", ".join(_object_summary(item) for item in value[:4])
        suffix = ", ..." if len(value) > 4 else ""
        return f"list(len={len(value)}, [{preview}{suffix}])"

    if isinstance(value, dict):
        items = list(value.items())[:6]
        preview = ", ".join(f"{_clip(repr(k), 40)}: {_object_summary(v)}" for k, v in items)
        suffix = ", ..." if len(value) > 6 else ""
        return f"dict(len={len(value)}, {{{preview}{suffix}}})"

    interesting_attrs = []
    for name in (
        "prompt",
        "prompt_token_ids",
        "num_reqs",
        "num_computed_tokens",
        "total_num_scheduled_tokens",
        "sampling_params",
        "request_id",
        "req_id",
    ):
        if hasattr(value, name):
            try:
                interesting_attrs.append(f"{name}={_object_summary(getattr(value, name))}")
            except Exception:
                interesting_attrs.append(f"{name}=<error>")
    if interesting_attrs:
        return f"{value.__class__.__name__}({', '.join(interesting_attrs)})"

    return f"<{value.__class__.__module__}.{value.__class__.__name__}>"


def _call_summary(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    if args:
        parts.append(f"self={args[0].__class__.__name__}")
        for index, arg in enumerate(args[1:5], start=1):
            parts.append(f"arg{index}={_object_summary(arg)}")
        if len(args) > 5:
            parts.append(f"... {len(args) - 5} more args")
    for key, value in list(kwargs.items())[:8]:
        parts.append(f"{key}={_object_summary(value)}")
    if len(kwargs) > 8:
        parts.append(f"... {len(kwargs) - 8} more kwargs")
    return "; ".join(parts)


def _wrap(label: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _call_counts[label] += 1
        count = _call_counts[label]
        should_log = count <= TRACE_LIMIT_PER_FUNCTION
        if should_log:
            print(f"{_rank_prefix()} -> {label} #{count}: {_call_summary(args, kwargs)}", flush=True)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            print(f"{_rank_prefix()} !! {label} #{count}: {exc.__class__.__name__}: {exc}", flush=True)
            raise
        if should_log:
            print(f"{_rank_prefix()} <- {label} #{count}: {_object_summary(result)}", flush=True)
        elif count == TRACE_LIMIT_PER_FUNCTION + 1:
            print(f"{_rank_prefix()} .. {label}: further calls hidden", flush=True)
        return result

    return wrapped


def _patch_method(module_name: str, class_name: str, method_name: str) -> None:
    key = f"{module_name}.{class_name}.{method_name}"
    if key in _patched:
        return
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        static_attr = inspect.getattr_static(cls, method_name)
        original = getattr(cls, method_name)
    except Exception as exc:
        print(f"{_rank_prefix()} skip patch {key}: {exc.__class__.__name__}: {exc}", flush=True)
        return

    if isinstance(static_attr, staticmethod):
        setattr(cls, method_name, staticmethod(_wrap(key, static_attr.__func__)))
    elif isinstance(static_attr, classmethod):
        setattr(cls, method_name, classmethod(_wrap(key, static_attr.__func__)))
    else:
        setattr(cls, method_name, _wrap(key, original))
    _patched.add(key)
    print(f"{_rank_prefix()} patched {key}", flush=True)


def install_trace_patches() -> None:
    targets = [
        ("tests.e2e.conftest", "VllmRunner", "__init__"),
        ("tests.e2e.conftest", "VllmRunner", "get_inputs"),
        ("tests.e2e.conftest", "VllmRunner", "generate"),
        ("tests.e2e.conftest", "VllmRunner", "generate_greedy"),
        ("tests.e2e.conftest", "VllmRunner", "_finalize_generate_outputs"),
        ("vllm.entrypoints.llm", "LLM", "__init__"),
        ("vllm.entrypoints.llm", "LLM", "generate"),
        ("vllm.entrypoints.offline_utils", "OfflineInferenceMixin", "_run_completion"),
        ("vllm.entrypoints.offline_utils", "OfflineInferenceMixin", "_render_and_run_requests"),
        ("vllm.entrypoints.offline_utils", "OfflineInferenceMixin", "_render_and_add_requests"),
        ("vllm.entrypoints.offline_utils", "OfflineInferenceMixin", "_add_request"),
        ("vllm.entrypoints.offline_utils", "OfflineInferenceMixin", "_run_engine"),
        ("vllm.engine.llm_engine", "LLMEngine", "add_request"),
        ("vllm.engine.llm_engine", "LLMEngine", "step"),
        ("vllm.v1.engine.llm_engine", "LLMEngine", "add_request"),
        ("vllm.v1.engine.llm_engine", "LLMEngine", "step"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "execute_model"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_prepare_inputs"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_preprocess"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_build_attention_metadata"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_model_forward"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "sample_tokens"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_sample"),
        ("vllm.v1.sample.sampler", "Sampler", "forward"),
        ("vllm.v1.sample.sampler", "Sampler", "sample"),
        ("vllm.v1.sample.sampler", "Sampler", "greedy_sample"),
    ]
    for module_name, class_name, method_name in targets:
        _patch_method(module_name, class_name, method_name)


install_trace_patches()


def main() -> None:
    from tests.e2e.conftest import VllmRunner

    prompts = ["Hello, my name is"]
    print(f"{_rank_prefix()} PROMPTS={prompts!r}", flush=True)
    runner = VllmRunner(
        "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
        max_model_len=4096,
        tensor_parallel_size=2,
        enable_expert_parallel=False,
        quantization="ascend",
        gpu_memory_utilization=0.9,
        distributed_executor_backend="mp",
        cudagraph_capture_sizes=[1, 2, 4, 8],
    )
    try:
        result = runner.generate_greedy(prompts, max_tokens=5)
        print(f"{_rank_prefix()} GENERATE_RESULT={result!r}", flush=True)
    finally:
        runner.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())
