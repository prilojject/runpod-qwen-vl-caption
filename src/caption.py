"""Run official Qwen3-VL generate() in-process so the model can stay loaded."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable

from .paths import DEFAULT_DTYPE, DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS
from .secret_utils import redact

log = logging.getLogger(__name__)

Progress = Callable[[dict], None]

_PROCESSOR = None
_MODEL = None
_MODEL_KEY: tuple | None = None


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu": None}
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu": torch.cuda.get_device_name(idx),
            "vram_gb": round(props.total_memory / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
        })
    except Exception as exc:
        info["error"] = redact(str(exc)).splitlines()[0][:200]
    return info


def _set_processor_pixel_budget(processor, max_pixels: int, min_pixels: int) -> None:
    """Always override the processor default (16.7M) with the official budget."""
    size = {"longest_edge": int(max_pixels), "shortest_edge": int(min_pixels)}
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("AutoProcessor has no image_processor")
    image_processor.size = size
    # Some processor configs also expose min_pixels / max_pixels directly.
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = int(max_pixels)
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = int(min_pixels)


def _load_model(model_path: str, max_pixels: int, min_pixels: int):
    """Load (or reuse) Qwen3VLForConditionalGeneration in bfloat16 on GPU 0."""
    global _PROCESSOR, _MODEL, _MODEL_KEY
    key = (os.path.abspath(model_path), int(max_pixels), int(min_pixels), DEFAULT_DTYPE)
    if _MODEL is not None and _MODEL_KEY == key:
        return _PROCESSOR, _MODEL
    if _MODEL is not None:
        try:
            del _MODEL
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        _MODEL = None
        _PROCESSOR = None

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    _set_processor_pixel_budget(processor, max_pixels, min_pixels)

    # Official path uses torch_dtype=torch.bfloat16 (native weight dtype). Not fp16.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    model.eval()
    _PROCESSOR, _MODEL, _MODEL_KEY = processor, model, key
    return processor, model


def _move_inputs(inputs, device):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            pass
    # Fallback: move tensor values on a dict-like batch.
    moved = type(inputs)() if not isinstance(inputs, dict) else {}
    for key, val in inputs.items():
        if hasattr(val, "to"):
            moved[key] = val.to(device)
        else:
            moved[key] = val
    return moved


def _write_caption(path: str, text: str) -> None:
    txt = os.path.splitext(path)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(text)


def run_caption(
    cfg,
    image_paths: list[str],
    model_path: str,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Caption (or short-VQA) images with the official chat template + generate loop.

    The model is kept in process memory so refresh_worker=false is useful on Serverless.
    Preprocess uses AutoProcessor only (official size override). qwen-vl-utils is
    installed but not used here, so the two resize paths are never mixed.
    """
    if not image_paths:
        return {"captioned": 0, "failed": 0}

    max_pixels = int(getattr(cfg, "max_pixels", DEFAULT_MAX_PIXELS) or DEFAULT_MAX_PIXELS)
    min_pixels = int(getattr(cfg, "min_pixels", DEFAULT_MIN_PIXELS) or DEFAULT_MIN_PIXELS)
    if progress:
        progress({"stage": "loading_model", "dtype": DEFAULT_DTYPE, "model_id": cfg.model_id})
    processor, model = _load_model(model_path, max_pixels, min_pixels)

    import torch

    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = cfg.prompt_text
    prepend = cfg.prepend or ""
    append = cfg.append or ""
    batch_size = int(cfg.batch_size or 1)
    timeout = int(getattr(cfg, "caption_timeout_seconds", 0) or 0)
    started = time.time()
    captioned = 0
    failed = 0
    timed_out = False
    if progress:
        progress({"stage": "captioning", "images": len(image_paths)})

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for start in range(0, len(image_paths), batch_size):
        if timeout and (time.time() - started) > timeout:
            timed_out = True
            failed += len(image_paths) - start
            log.error("caption watchdog fired after %ss", timeout)
            break
        chunk = image_paths[start:start + batch_size]
        conversations = []
        try:
            for path in chunk:
                # Official README path: {"type": "image", "image": image_path}.
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ])
            # Official chat template. padding=True is required when batch_size > 1.
            inputs = processor.apply_chat_template(
                conversations if batch_size > 1 or len(conversations) > 1 else conversations[0],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = _move_inputs(inputs, device)

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(cfg.max_new_tokens),
                "use_cache": True,
            }
            if cfg.greedy:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": float(cfg.temperature),
                    "top_p": float(cfg.top_p),
                })
                if int(cfg.top_k or 0) > 0:
                    gen_kwargs["top_k"] = int(cfg.top_k)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            input_ids = inputs["input_ids"] if isinstance(inputs, dict) or hasattr(inputs, "keys") else inputs.input_ids
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            texts = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for path, text in zip(chunk, texts):
                caption = prepend + (text or "").strip() + append
                try:
                    _write_caption(path, caption)
                    captioned += 1
                except Exception as exc:
                    log.error("failed to write caption for %s: %s", path, redact(str(exc)))
                    failed += 1
        except Exception as exc:
            log.error("captioning failed for batch starting %s: %s", chunk[0], redact(str(exc)))
            for path in chunk:
                # Leave no empty sidecar behind for a failed image.
                failed += 1
            # Surface the first failure message in the last result via logging only.
        if progress:
            progress({"stage": "captioning", "captioned": captioned, "failed": failed, "total": len(image_paths)})

    out = {"captioned": captioned, "failed": failed}
    if timed_out:
        out["timed_out"] = True
        out["timeout_seconds"] = timeout
    return out


def collect_captions(images_dir: str, image_paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in image_paths:
        rel = os.path.relpath(path, images_dir)
        txt = os.path.splitext(path)[0] + ".txt"
        caption = ""
        if os.path.isfile(txt):
            try:
                caption = open(txt, encoding="utf-8").read()
            except OSError:
                caption = ""
        rows.append({"file": rel.replace("\\", "/"), "caption": caption})
    return rows
