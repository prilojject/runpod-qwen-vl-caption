"""Download and verify Qwen3-VL Instruct weights into the volume (or ephemeral) cache."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .paths import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    EXPECTED_SHARDS_4B,
    model_cache_dir,
)
from .secret_utils import redact

log = logging.getLogger(__name__)

Progress = Callable[[dict], None]


class ModelError(RuntimeError):
    pass


def local_model_dir(cfg) -> str:
    cache = model_cache_dir()
    # Keep a stable folder name so a volume cache is reused across jobs.
    safe = cfg.model_id.replace("/", "__")
    return os.path.join(cache, safe)


def expected_shards(cfg) -> dict[str, int] | None:
    if cfg.model_id == DEFAULT_MODEL_ID and cfg.model_revision == DEFAULT_MODEL_REVISION:
        return EXPECTED_SHARDS_4B
    return None


def shards_ok(dest: str, expected: dict[str, int] | None) -> bool:
    if not os.path.isfile(os.path.join(dest, "config.json")):
        return False
    if expected is None:
        # Non-default checkpoints: require at least one safetensors / bin shard.
        names = [n for n in os.listdir(dest) if n.endswith((".safetensors", ".bin"))] if os.path.isdir(dest) else []
        return bool(names)
    for name, size in expected.items():
        path = os.path.join(dest, name)
        if not os.path.isfile(path) or os.path.getsize(path) != size:
            return False
    return True


def check_access(cfg, token: str | None) -> dict[str, Any]:
    """HEAD-level reachability check for a dry run (no full 8.3 GiB download)."""
    from huggingface_hub import HfApi

    info: dict[str, Any] = {
        "model_id": cfg.model_id,
        "revision": cfg.model_revision,
        "ok": False,
    }
    try:
        api = HfApi(token=token)
        model = api.model_info(cfg.model_id, revision=cfg.model_revision)
        siblings = {s.rfilename: getattr(s, "size", None) for s in (model.siblings or [])}
        info["files"] = len(siblings)
        info["gated"] = bool(getattr(model, "gated", False))
        expected = expected_shards(cfg)
        if expected:
            missing = [n for n in expected if n not in siblings]
            if missing:
                info["error"] = f"missing shards on the Hub: {missing}"
                return info
            info["size_gb"] = round(sum(expected.values()) / 2**30, 2)
        info["ok"] = True
    except Exception as exc:
        info["error"] = redact(str(exc)).splitlines()[0][:400]
    return info


def ensure_models(cfg, token: str | None, progress: Progress | None = None) -> str:
    dest = local_model_dir(cfg)
    os.makedirs(dest, exist_ok=True)
    expected = expected_shards(cfg)
    if shards_ok(dest, expected):
        log.info("using cached weights at %s", dest)
        return dest
    if progress:
        progress({"stage": "downloading_weights", "dest": dest})
    log.info("downloading %s @ %s into cache", cfg.model_id, cfg.model_revision)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=cfg.model_id,
            revision=cfg.model_revision,
            local_dir=dest,
            token=token,
            # Public Apache-2.0 weights; token is only used if the caller set one.
        )
    except Exception as exc:
        raise ModelError(
            f"failed to download {cfg.model_id}@{cfg.model_revision}: "
            f"{redact(str(exc)).splitlines()[0][:400]}"
        ) from exc
    if expected and not shards_ok(dest, expected):
        sizes = {
            name: (os.path.getsize(os.path.join(dest, name)) if os.path.isfile(os.path.join(dest, name)) else None)
            for name in expected
        }
        raise ModelError(f"weight download incomplete or size mismatch: {sizes}")
    if not shards_ok(dest, expected):
        raise ModelError(f"weight download looks incomplete under {dest}")
    return dest
