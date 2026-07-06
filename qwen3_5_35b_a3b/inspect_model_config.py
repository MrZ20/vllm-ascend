"""打印 Qwen3.5-35B-A3B-W8A8-MTP 的关键配置。

这个脚本需要在真实测试容器里运行，因为模型文件缓存在远端容器中。
它只读取配置，不启动 vLLM 引擎，也不加载权重。
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoConfig


MODEL = "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp"


def _safe_get(obj, name: str):
    return getattr(obj, name, None)


def _pick_config(cfg):
    # Qwen3.5 MoE / VL 类模型经常把纯文本配置放在 text_config 里。
    return getattr(cfg, "text_config", None) or getattr(cfg, "language_config", None) or cfg


def _find_quant_descriptions() -> list[str]:
    roots = [
        Path("/root/.cache"),
        Path("/home"),
        Path("/vllm-workspace"),
        Path("/workspace"),
    ]
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("quant_model_description.json"):
            text = str(path)
            if "Qwen3.5-35B-A3B" in text or "Eco-Tech" in text or "qwen" in text.lower():
                hits.append(text)
    return sorted(set(hits))


def main() -> None:
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    text_cfg = _pick_config(cfg)
    keys = [
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "max_position_embeddings",
        "hidden_act",
        "num_experts",
        "num_experts_per_tok",
        "norm_topk_prob",
        "shared_expert_intermediate_size",
        "layer_types",
        "tie_word_embeddings",
        "attn_output_gate",
        "qkv_bias",
        "rope_parameters",
    ]
    print("CONFIG_SUMMARY")
    for key in keys:
        value = _safe_get(text_cfg, key)
        if key == "layer_types" and isinstance(value, list):
            counts: dict[str, int] = {}
            for item in value:
                counts[item] = counts.get(item, 0) + 1
            print(f"{key}={json.dumps(counts, ensure_ascii=False)}")
            print(f"{key}_first_16={json.dumps(value[:16], ensure_ascii=False)}")
        else:
            print(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}")

    print("TOP_LEVEL")
    print(f"top_model_type={json.dumps(_safe_get(cfg, 'model_type'), ensure_ascii=False, default=str)}")
    print(f"top_architectures={json.dumps(_safe_get(cfg, 'architectures'), ensure_ascii=False, default=str)}")
    print(f"quantization_config={json.dumps(_safe_get(cfg, 'quantization_config'), ensure_ascii=False, default=str)}")

    print("QUANT_DESCRIPTION_FILES")
    for path in _find_quant_descriptions():
        print(path)


if __name__ == "__main__":
    main()
