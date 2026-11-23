"""Deliver caption artifacts: zip of images + sidecar .txt + captions.json."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
UPSTREAM_SHA = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def write_captions_json(
    path: str,
    captions: list[dict[str, str]],
    cfg,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": cfg.model_id,
        "revision": cfg.model_revision,
        "dtype": "bfloat16",
        "mode": cfg.mode,
        "language": cfg.language,
        "length": cfg.length,
        "prompt": cfg.prompt_text,
        "captions": [
            {
                "filename": row["file"],
                "caption": row["caption"],
                "mode": cfg.mode,
                "prompt": cfg.prompt_text,
            }
            for row in captions
        ],
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def zip_captions(images_dir: str, image_paths: list[str], zip_path: str, captions_json: str) -> str:
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in image_paths:
            rel = os.path.relpath(path, images_dir)
            z.write(path, rel)
            txt = os.path.splitext(path)[0] + ".txt"
            if os.path.isfile(txt):
                z.write(txt, os.path.splitext(rel)[0] + ".txt")
        z.write(captions_json, "captions.json")
    return zip_path


def push_to_hf(cfg, token: str, files: list[str], card: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=cfg.hf_repo_id, repo_type="model", private=cfg.hf_private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    for path in files:
        ops.append(CommitOperationAdd(path_in_repo=os.path.basename(path), path_or_fileobj=path))
    api.create_commit(
        repo_id=cfg.hf_repo_id, repo_type="model", operations=ops,
        commit_message=f"Qwen3-VL {cfg.mode} captions ({cfg.run_name or 'run'})",
    )
    return f"https://huggingface.co/{cfg.hf_repo_id}"


def upload_bucket(path: str, prefix: str) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket
    return upload_file_to_bucket(
        file_name=os.path.basename(path), file_location=path,
        bucket_name=os.environ.get("BUCKET_NAME") or None, prefix=prefix,
    )


def put_presigned(url: str, path: str) -> int:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=(15, 900))
    if r.status_code >= 300:
        raise RuntimeError(f"presigned PUT returned HTTP {r.status_code}: {r.text[:200]}")
    return r.status_code


def _card(cfg, dataset: dict[str, Any]) -> str:
    return "\n".join([
        "---",
        "tags:",
        "- qwen3-vl",
        "- image-captioning",
        "- lora-dataset",
        "- runpod",
        "---",
        "",
        "# Qwen3-VL captions",
        "",
        "Sidecar `.txt` captions produced with "
        "[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) "
        f"(`{UPSTREAM_SHA}`) and "
        f"[{cfg.model_id}](https://huggingface.co/{cfg.model_id}) "
        f"(revision `{cfg.model_revision}`) on Runpod Serverless using "
        "[prilojject/runpod-qwen-vl-caption](https://github.com/prilojject/runpod-qwen-vl-caption).",
        "",
        f"- mode: `{cfg.mode}` / language: `{cfg.language}` / length: `{cfg.length}`",
        f"- prompt: `{cfg.prompt_text}`",
        f"- dtype: bfloat16",
        f"- pixel budget: longest_edge={cfg.max_pixels}, shortest_edge={cfg.min_pixels}",
        f"- images: {dataset.get('images_used')}",
        "",
        "Qwen3-VL code and the 2B/4B/8B Instruct weights are licensed under Apache-2.0, "
        "Copyright © Alibaba Cloud / the Qwen team. This worker code is MIT.",
        "",
    ])


def deliver(cfg, run_name: str, run_dir: str, zip_path: str, json_path: str,
            token: str | None, persistent: bool) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {"persistent_storage": persistent}
    warnings: list[str] = []
    files: list[str] = []
    if zip_path and os.path.isfile(zip_path):
        out["zip_file"] = os.path.basename(zip_path)
        out["zip_path"] = zip_path
        out["zip_size_mb"] = round(os.path.getsize(zip_path) / 2**20, 2)
        files.append(zip_path)
        if persistent:
            out["zip_volume_path"] = zip_path
    if json_path and os.path.isfile(json_path):
        out["captions_json"] = os.path.basename(json_path)
        out["captions_json_path"] = json_path
        files.append(json_path)
        if persistent:
            out["captions_json_volume_path"] = json_path
    if persistent:
        out["volume_dir"] = run_dir

    delivered = persistent and bool(files)
    card = _card(cfg, {})
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    if cfg.hf_repo_id and token and files:
        try:
            out["hf_repo_url"] = push_to_hf(cfg, token, files, card)
            delivered = True
        except Exception as exc:
            out["hf_error"] = redact(str(exc)).splitlines()[0][:400]
            log.error("HF upload failed: %s", out["hf_error"])
    if cfg.upload_url and zip_path and os.path.isfile(zip_path):
        try:
            out["upload_url_status"] = put_presigned(cfg.upload_url, zip_path)
            delivered = True
        except Exception as exc:
            out["upload_url_error"] = redact(str(exc))[:400].replace(cfg.upload_url, "<presigned url>")
    if cfg.upload_to_bucket and bucket_configured() and files:
        try:
            prefix = f"qwen-vl/{run_name}"
            for path in files:
                key = "bucket_zip_url" if path.endswith(".zip") else "bucket_json_url"
                out[key] = upload_bucket(path, prefix)
            delivered = True
        except Exception as exc:
            out["bucket_error"] = redact(str(exc))[:400]
    if not delivered:
        warnings.append(
            "the sidecar .txt files were not delivered anywhere persistent: attach a network volume, "
            "or set hf_repo_id (+HF_TOKEN), upload_url, or BUCKET_* env vars. They are lost when the worker stops."
        )
    return out, warnings
