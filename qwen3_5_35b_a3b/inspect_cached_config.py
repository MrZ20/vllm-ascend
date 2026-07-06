"""从容器内已定位到的 config.json 打印关键模型结构。

用法：
    python /tmp/inspect_cached_config.py /path/to/config.json
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


KEYS = [
    "architectures",
    "model_type",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "norm_topk_prob",
    "shared_expert_intermediate_size",
    "rms_norm_eps",
    "hidden_act",
    "max_position_embeddings",
    "attn_output_gate",
    "qkv_bias",
    "tie_word_embeddings",
    "rope_parameters",
    "quantization_config",
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspect_cached_config.py /path/to/config.json")
    path = Path(sys.argv[1])
    obj = json.loads(path.read_text())
    text_obj = obj.get("text_config") or obj.get("language_config") or obj
    print(f"CONFIG_PATH={path}")
    print("TOP_LEVEL")
    print(f"architectures={json.dumps(obj.get('architectures'), ensure_ascii=False, default=str)}")
    print(f"model_type={json.dumps(obj.get('model_type'), ensure_ascii=False, default=str)}")
    print("CONFIG_SUMMARY")
    for key in KEYS:
        print(f"{key}={json.dumps(text_obj.get(key), ensure_ascii=False, default=str)}")
    layer_types = text_obj.get("layer_types")
    if isinstance(layer_types, list):
        print(f"layer_types_count={json.dumps(dict(Counter(layer_types)), ensure_ascii=False)}")
        print(f"layer_types_first_32={json.dumps(layer_types[:32], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
