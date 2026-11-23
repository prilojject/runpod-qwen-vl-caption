"""Runpod Serverless entry point: headless Qwen3-VL batch captioning.

Writes kohya-style sidecar .txt next to each image, plus captions.json.
Optional short VQA (one question → one answer). Not a general chat endpoint.

Local dry run (no GPU, no model weights):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset_url": "https://.../tiny_images.zip"}}'
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, resolve_config  # noqa: E402
from src.dataset import DatasetError, prepare_dataset, select_images_for_caption  # noqa: E402
from src.paths import configure_caches, has_volume, output_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("qwen-vl-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        # After captioning, only log. A blocking progress_update here can
        # leave the job stuck at the last stage if the worker then exits.
        if payload.get("stage") in {"uploading", "done"}:
            return
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def handler(job: dict) -> dict[str, Any]:
    result = _handle(job)
    # Keep the ~8.3 GiB 4B model in worker memory across jobs. Do not copy
    # refresh_worker=true from the trainer listings.
    result["refresh_worker"] = False
    import json as _json
    return _json.loads(_json.dumps(result, default=str))


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    # Keep the resolved run_name on the config object used for HF commit messages.
    cfg.values["run_name"] = run_name
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = has_volume()
    run_dir = os.path.join(output_root(), run_name)
    work = os.path.join("/tmp/qwen-vl/work", run_name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    result: dict[str, Any] = {
        "worker_version": __version__,
        "run_name": run_name,
        "status": "error",
        "config": cfg.public_dict(),
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        result["warnings"].append(
            "no network volume: Qwen3-VL-4B-Instruct weights (~8.27 GiB) are re-downloaded "
            "on every cold start (attach one to cache them under /runpod-volume/qwen-vl/models)"
        )

    try:
        progress({"stage": "preparing_dataset"})
        report = prepare_dataset(cfg, work)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", **{k: result[k] for k in ("run_name", "config")}}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}"}

    to_caption, skipped = select_images_for_caption(report.image_paths, cfg.overwrite)
    report.skipped_existing_txt = len(skipped)
    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    if cfg.dry_run:
        from src.models import check_access
        access = check_access(cfg, token)
        result.update({
            "dry_run": True,
            "status": "ok" if access.get("ok") else "error",
            "model_access": access,
            "prompt": cfg.prompt_text,
            "images_to_caption": len(to_caption),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if result["status"] == "error":
            result["error"] = "weights not reachable: " + str(access.get("error"))
        shutil.rmtree(work, ignore_errors=True)
        return result

    from src.caption import collect_captions, gpu_info, run_caption
    from src.models import ModelError, ensure_models
    from src.outputs import deliver, write_captions_json, zip_captions

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu.get("gpu"):
        return {"error": "no CUDA GPU visible to the worker", **{k: result[k] for k in ("run_name", "config")}}

    try:
        model_path = ensure_models(cfg, token, progress)
    except ModelError as exc:
        return {"error": str(exc), **{k: result[k] for k in ("run_name", "config")}}

    if to_caption:
        try:
            cap = run_caption(cfg, to_caption, model_path, progress)
        except Exception as exc:
            log.error("captioning crashed: %s", redact(traceback.format_exc()))
            return {
                "error": f"captioning crashed: {redact(str(exc)).splitlines()[0][:500]}",
                "run_name": run_name,
                "gpu": gpu,
            }
        result["captioning"] = cap
        if cap.get("timed_out"):
            result["warnings"].append(
                f"caption watchdog fired after {cap.get('timeout_seconds')}s; "
                f"{cap.get('failed', 0)} image(s) were not captioned"
            )
    else:
        result["captioning"] = {"captioned": 0, "failed": 0}
        result["warnings"].append(
            "every image already had a sidecar .txt; nothing was captioned "
            "(set overwrite=true to replace them)"
        )

    captions = collect_captions(report.images_dir, report.image_paths)
    empty = [row["file"] for row in captions if not (row["caption"] or "").strip()]
    if empty:
        result["warnings"].append(f"{len(empty)} image(s) have an empty sidecar .txt: {empty[:8]}")

    # Copy images + sidecars into the persistent run dir.
    dest_images = os.path.join(run_dir, "dataset")
    if os.path.isdir(dest_images):
        shutil.rmtree(dest_images)
    shutil.copytree(report.images_dir, dest_images)
    copied_paths = [os.path.join(dest_images, os.path.relpath(p, report.images_dir)) for p in report.image_paths]

    json_path = os.path.join(run_dir, "captions.json")
    write_captions_json(json_path, collect_captions(dest_images, copied_paths), cfg)
    zip_path = os.path.join(run_dir, "captions.zip")
    zip_captions(dest_images, copied_paths, zip_path, json_path)

    progress({"stage": "uploading"})
    try:
        artifacts, warns = deliver(cfg, run_name, run_dir, zip_path, json_path, token, persistent)
        result["warnings"] += warns
        result["artifacts"] = artifacts
    except Exception as exc:
        log.error("delivery failed: %s", redact(traceback.format_exc()))
        result["warnings"].append(f"delivery failed: {redact(str(exc)).splitlines()[0][:300]}")
        result["artifacts"] = {"zip_path": zip_path, "captions_json_path": json_path}
    if cfg.return_captions_in_response:
        result["captions"] = captions
    failed = int((result.get("captioning") or {}).get("failed") or 0)
    result.update({
        "status": "completed" if failed == 0 else "error",
        "elapsed_seconds": round(time.time() - started, 1),
    })
    if failed and "error" not in result:
        result["error"] = f"captioning failed for {failed} image(s)"
    shutil.rmtree(work, ignore_errors=True)
    # Guarantees the Serverless runtime can JSON-encode the payload.
    import json as _json
    return _json.loads(_json.dumps(result, default=str))


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
