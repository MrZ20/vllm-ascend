"""打印 ModelSlim 量化描述文件的结构摘要。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from walk(value, path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj[:20]):
            path = f"{prefix}[{index}]"
            yield path, value
            yield from walk(value, path)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspect_quant_description.py /path/to/quant_model_description.json")
    path = Path(sys.argv[1])
    obj = json.loads(path.read_text())
    print(f"QUANT_PATH={path}")
    if isinstance(obj, dict):
        top_keys = list(obj)
        print(f"top_key_count={len(top_keys)}")
        print(f"top_keys_first_20={json.dumps(top_keys[:20], ensure_ascii=False)}")
    else:
        print(f"top_type={type(obj).__name__}")

    if isinstance(obj, dict):
        suffix_counter = Counter()
        value_counter = Counter()
        for key in obj:
            if key.startswith("model.language_model.") or key.startswith("mtp."):
                suffix_counter[key.rsplit(".", 1)[-1]] += 1
                value_counter[str(obj[key])] += 1
        print(f"language_or_mtp_suffix_count={dict(suffix_counter)}")
        print(f"language_or_mtp_value_count={dict(value_counter)}")

        probe_keys = [
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_offset",
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            "model.language_model.layers.3.self_attn.qkv_proj.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
            "lm_head.weight",
            "version",
            "model_quant_type",
            "group_size",
            "metadata",
        ]
        print("PROBE_ENTRIES")
        for key in probe_keys:
            if key in obj:
                value = obj[key]
                text = json.dumps(value, ensure_ascii=False, default=str)
                if len(text) > 800:
                    text = text[:800] + "...<truncated>"
                print(f"{key}={text}")

    interesting_names = {
        "quant_type",
        "quant_method",
        "ascend_quant_method",
        "weight_type",
        "act_type",
        "input_quant_type",
        "weight_quant_type",
        "activation_quant_type",
        "group_size",
        "version",
    }
    counters: dict[str, Counter[str]] = {name: Counter() for name in interesting_names}
    samples: list[tuple[str, object]] = []
    for node_path, value in walk(obj):
        leaf = node_path.split(".")[-1]
        if leaf in interesting_names:
            counters[leaf][str(value)] += 1
            if len(samples) < 80:
                samples.append((node_path, value))

    print("COUNTERS")
    for key in sorted(counters):
        if counters[key]:
            print(f"{key}={dict(counters[key])}")

    print("SAMPLES")
    for node_path, value in samples:
        print(f"{node_path}={json.dumps(value, ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    main()
