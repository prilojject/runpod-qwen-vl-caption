"""Where models, datasets and outputs live."""

from __future__ import annotations

import os

NETWORK_VOLUME = "/runpod-volume"
QWEN3_VL_SHA = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DEFAULT_DTYPE = "bfloat16"

# Official visual-token budget from the Qwen3-VL README (compression 32).
# Do not use the processor default longest_edge=16_777_216.
DEFAULT_MAX_PIXELS = 1280 * 32 * 32  # 1_310_720
DEFAULT_MIN_PIXELS = 256 * 32 * 32   # 262_144

# Allowed Instruct checkpoints only. Thinking / 32B / MoE / Qwen2.5-VL are rejected.
MODEL_REVISIONS = {
    "Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "Qwen/Qwen3-VL-2B-Instruct": "89644892e4d85e24eaac8bacfd4f463576704203",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
}

# Shard sizes on the pinned 4B revision (bytes). Used to verify a complete download.
EXPECTED_SHARDS_4B = {
    "model-00001-of-00002.safetensors": 4_967_229_296,
    "model-00002-of-00002.safetensors": 3_908_490_048,
}


def has_volume() -> bool:
    return os.path.isdir(NETWORK_VOLUME) and os.access(NETWORK_VOLUME, os.W_OK)


def model_cache_dir() -> str:
    if os.environ.get("MODEL_CACHE_DIR"):
        return os.environ["MODEL_CACHE_DIR"]
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "qwen-vl", "models")
    return "/tmp/qwen-vl/models"


def output_root() -> str:
    if os.environ.get("OUTPUT_ROOT"):
        return os.environ["OUTPUT_ROOT"]
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "qwen-vl", "runs")
    return "/tmp/qwen-vl/runs"


def configure_caches() -> None:
    if has_volume():
        os.environ.setdefault("HF_HOME", os.path.join(NETWORK_VOLUME, "qwen-vl", "hf-home"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(NETWORK_VOLUME, "qwen-vl", "hf-hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(NETWORK_VOLUME, "qwen-vl", "hf-hub"))
    os.makedirs(model_cache_dir(), exist_ok=True)
    os.makedirs(output_root(), exist_ok=True)
