"""Input schema for the headless Qwen3-VL batch caption worker.

Every parameter comes from (highest priority first):

1. the job ``input`` object,
2. an environment variable (what the Runpod Hub deploy form sets),
3. the preset defaults,
4. the built-in default.

Dependency-free so it can be unit-tested and used for a dry run without a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .paths import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    MODEL_REVISIONS,
)
from .prompts import LENGTHS, assemble_prompt

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
LANG_RE = re.compile(r"^[a-z]{2,8}(?:-[a-z0-9]{1,8})?$")

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}

MODES = ("caption", "vqa")
MAX_NEW_TOKENS_BY_LENGTH = {"short": 128, "medium": 256, "long": 512}
MAX_NEW_TOKENS_DEFAULT = 256

PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "mode": "caption",
        "length": "medium",
        "language": "en",
        "max_new_tokens": 256,
        "greedy": False,
        "batch_size": 1,
    },
    "quick": {
        "mode": "caption",
        "length": "medium",
        "language": "en",
        "max_new_tokens": 128,
        "greedy": True,
        "batch_size": 1,
    },
    "chinese": {
        "mode": "caption",
        "length": "medium",
        "language": "zh",
        "max_new_tokens": 256,
        "greedy": False,
        "batch_size": 1,
    },
}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


PARAMS: list[Param] = [
    Param("image_url", str, "", "IMAGE_URL",
          help="https URL of a single image (jpg/png/webp/...). Use instead of, or together with, dataset_url."),
    Param("dataset_url", str, "", "DATASET_URL",
          help="https URL of a .zip / .tar(.gz) of images. Sidecar .txt files are written next to each image."),
    Param("model_id", str, DEFAULT_MODEL_ID, "MODEL_ID",
          help="Hugging Face model id. Default is Qwen3-VL-4B-Instruct. 2B and 8B Instruct are optional."),
    Param("model_revision", str, DEFAULT_MODEL_REVISION, "MODEL_REVISION",
          help="Pinned weights revision. Do not switch to a floating main."),
    Param("mode", str, "caption", "MODE", choices=MODES,
          help="caption (default) or short vqa (requires question). Not a general chat endpoint."),
    Param("length", str, "medium", "LENGTH", choices=LENGTHS,
          help="Caption length: short / medium (default) / long. Ignored for vqa and custom prompt."),
    Param("language", str, "en", "LANGUAGE",
          help="Caption language. en (default) and zh are first-class; other codes add an explicit instruction."),
    Param("prompt", str, "", "PROMPT",
          help="Custom caption prompt. Overrides mode/length/language templates. Ignored in vqa mode."),
    Param("question", str, "", "QUESTION",
          help="Required for mode=vqa. A single question; the worker returns a single answer. No chat history."),
    Param("max_new_tokens", int, MAX_NEW_TOKENS_DEFAULT, "MAX_NEW_TOKENS", min=1, max=2048,
          help="Maximum new tokens. Defaults: 128 short / 256 medium / 512 long."),
    Param("greedy", bool, False, "GREEDY",
          help="Greedy decoding (useful for smoke tests). Default is sampling at T=0.7, top-p=0.8, top-k=20."),
    Param("temperature", float, 0.7, "TEMPERATURE", min=0.0, max=2.0),
    Param("top_p", float, 0.8, "TOP_P", min=0.0, max=1.0),
    Param("top_k", int, 20, "TOP_K", min=0, max=1000,
          help="Top-k. Official generation_config default is 20. 0 disables top-k."),
    Param("max_pixels", int, DEFAULT_MAX_PIXELS, "MAX_PIXELS", min=32 * 32, max=16_777_216,
          help="Processor longest_edge in pixels. Default is the official 1280*32*32 budget, not 16.7M."),
    Param("min_pixels", int, DEFAULT_MIN_PIXELS, "MIN_PIXELS", min=32 * 32, max=16_777_216,
          help="Processor shortest_edge in pixels. Default is the official 256*32*32 budget."),
    Param("batch_size", int, 1, "BATCH_SIZE", min=1, max=8,
          help="Images per generate() call. Default 1. Keep small on a 24 GB GPU."),
    Param("overwrite", bool, False, "OVERWRITE",
          help="Overwrite existing sidecar .txt files. Existing files are skipped by default."),
    Param("prepend", str, "", "PREPEND", help="String prepended to every caption."),
    Param("append", str, "", "APPEND", help="String appended to every caption."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO",
          help="Upload the captions zip + captions.json here (needs HF_TOKEN with write)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("upload_url", str, "", None, help="Presigned HTTP PUT URL for the captions zip."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET",
          help="If BUCKET_* env vars are set, upload the artifacts and return URLs."),
    Param("return_captions_in_response", bool, True, "RETURN_CAPTIONS_IN_RESPONSE",
          help="Include {file, caption} objects in the job output (no image bytes)."),
    Param("run_name", str, "", "RUN_NAME",
          help="Folder name for outputs. Default: the Runpod job id."),
    Param("seed", int, 1234, "SEED", min=0),
    Param("caption_timeout_seconds", int, 0, "CAPTION_TIMEOUT_SECONDS", min=0, max=86400,
          help="Advisory watchdog in seconds (0 = off). Keep below the endpoint execution timeout."),
    Param("dry_run", bool, False, "DRY_RUN",
          help="Download + validate images, check that weights are reachable. No GPU work."),
    Param("preset", str, "default", "PRESET", choices=tuple(PRESETS),
          help="Named defaults: default (caption medium en), quick (greedy), chinese (caption medium zh)."),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

ALIASES = {
    "dataset": "dataset_url",
    "zip_url": "dataset_url",
    "image": "image_url",
    "url": "image_url",
    "caption_mode": "mode",
    "caption_length": "length",
    "lang": "language",
    "max_tokens": "max_new_tokens",
    "output_repo": "hf_repo_id",
    "presigned_url": "upload_url",
    "longest_edge": "max_pixels",
    "shortest_edge": "min_pixels",
    "q": "question",
}


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    prompt_text: str = ""

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def public_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if out.get("upload_url"):
            out["upload_url"] = "<presigned url>"
        out["prompt"] = self.prompt_text
        out["dtype"] = "bfloat16"
        out["max_pixels"] = int(out.get("max_pixels") or DEFAULT_MAX_PIXELS)
        out["min_pixels"] = int(out.get("min_pixels") or DEFAULT_MIN_PIXELS)
        return out


def _coerce(param: Param, raw: Any) -> Any:
    t = param.type
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def _reject_disallowed_model(model_id: str) -> str | None:
    mid = model_id.strip()
    lowered = mid.lower()
    if "qwen2.5" in lowered or "qwen2-5" in lowered or "qwen2_5" in lowered:
        return (
            "Qwen2.5-VL is not supported (the 3B weights use the Qwen Research License, "
            "non-commercial). Use Qwen3-VL-4B/2B/8B-Instruct."
        )
    if "florence" in lowered:
        return "Florence-2 is not part of this worker"
    if "thinking" in lowered:
        return "Thinking checkpoints are not supported in v1; use an Instruct model"
    if mid not in MODEL_REVISIONS:
        allowed = ", ".join(MODEL_REVISIONS)
        return f"model_id: expected one of {allowed}, got {mid!r}"
    return None


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    env = dict(os.environ if env is None else env)
    raw_in = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    # This worker is a batch captioner, not a chat / vLLM endpoint.
    for banned_chat in ("messages", "chat", "conversation", "history", "tools"):
        if banned_chat in {k.lower() for k in raw_in}:
            errors.append(
                "this is a batch caption / short-VQA worker, not a general chat endpoint; "
                "send image_url or dataset_url, and for vqa a single 'question'"
            )
            break

    normalized_input: dict[str, Any] = {}
    for key, val in raw_in.items():
        name = ALIASES.get(key, key)
        if name in normalized_input and name != key:
            warnings.append(f"alias {key} ignored; {name} already set")
            continue
        normalized_input[name] = val

    preset_name = "default"
    for candidate, source in (
        (normalized_input.get("preset"), "input"),
        (env.get("PRESET"), "env"),
    ):
        if candidate is not None and str(candidate).strip() != "":
            name = str(candidate).strip().lower()
            if name not in PRESETS:
                errors.append(f"preset: unknown value {candidate!r} (use default, quick, or chinese)")
            else:
                preset_name = name
            break

    preset = dict(PRESETS[preset_name])
    max_tokens_explicit = (
        "max_new_tokens" in normalized_input and normalized_input["max_new_tokens"] not in (None, "")
    ) or bool(env.get("MAX_NEW_TOKENS"))

    for param in PARAMS:
        raw = None
        source = "default"
        if param.name in normalized_input and normalized_input[param.name] is not None and normalized_input[param.name] != "":
            raw = normalized_input[param.name]
            source = "input"
        elif param.env and env.get(param.env) not in (None, ""):
            raw = env[param.env]
            source = "env"
        elif param.name in preset:
            raw = preset[param.name]
            source = f"preset:{preset_name}"
        else:
            raw = param.default
            source = "default"

        if raw is None:
            values[param.name] = param.default
            sources[param.name] = source
            continue
        try:
            val = _coerce(param, raw)
            if param.name in {"mode", "length", "language", "preset"}:
                val = str(val).strip().lower()
            if param.name == "language" and not LANG_RE.match(val):
                raise ValueError(f"language: expected a short language code such as en or zh, got {val!r}")
            if param.choices and val not in param.choices:
                raise ValueError(f"{param.name}: expected one of {param.choices}, got {val!r}")
            if param.min is not None and isinstance(val, (int, float)) and val < param.min:
                raise ValueError(f"{param.name}: {val} is below the minimum {param.min}")
            if param.max is not None and isinstance(val, (int, float)) and val > param.max:
                raise ValueError(f"{param.name}: {val} is above the maximum {param.max}")
            values[param.name] = val
            sources[param.name] = source
        except ValueError as exc:
            errors.append(str(exc))

    # Tokens must never arrive in the job input.
    for banned in ("hf_token", "token", "authorization", "api_key", "secret", "password",
                   "aws_secret_access_key", "bucket_secret_access_key"):
        if banned in {k.lower() for k in raw_in}:
            errors.append("secrets must not be sent in the job input; use Runpod secrets / env")
            break

    if not values.get("image_url") and not values.get("dataset_url"):
        errors.append("image_url or dataset_url is required")
    if values.get("image_url") and not URL_RE.match(values["image_url"]):
        errors.append("image_url must be an http(s) URL")
    if values.get("dataset_url") and not URL_RE.match(values["dataset_url"]):
        errors.append("dataset_url must be an http(s) URL")
    if values.get("upload_url") and not URL_RE.match(values["upload_url"]):
        errors.append("upload_url must be an http(s) URL")
    if values.get("hf_repo_id") and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id must look like 'user/repo'")
    if values.get("run_name") and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name may only contain letters, digits, '.', '_' and '-'")

    if "model_id" in values:
        mid_err = _reject_disallowed_model(values["model_id"])
        if mid_err:
            errors.append(mid_err)
        else:
            # If the caller picked 2B/8B but left the 4B revision (default/env leftover),
            # swap in the pinned revision for that checkpoint.
            mid = values["model_id"]
            pinned = MODEL_REVISIONS[mid]
            rev_source = sources.get("model_revision", "default")
            if values.get("model_revision") in {DEFAULT_MODEL_REVISION, "", None} and mid != DEFAULT_MODEL_ID:
                values["model_revision"] = pinned
                sources["model_revision"] = "pinned-for-model"
            elif rev_source == "default":
                values["model_revision"] = pinned
                sources["model_revision"] = "default"
            if mid.endswith("8B-Instruct"):
                warnings.append(
                    "Qwen3-VL-8B-Instruct is optional and tight on 24 GB; the official "
                    "1280*32*32 pixel budget is required to avoid OOM. 8B is not the default or smoke path."
                )

    if values.get("min_pixels") and values.get("max_pixels") and values["min_pixels"] > values["max_pixels"]:
        errors.append("min_pixels must be <= max_pixels")

    # If the caller picked a length but did not set max_new_tokens, use the length default.
    # Do not override a preset that already chose a token cap (e.g. quick=128).
    if not errors and not max_tokens_explicit and values.get("mode") == "caption":
        length = values.get("length") or "medium"
        length_source = sources.get("length", "default")
        token_source = sources.get("max_new_tokens", "default")
        if length_source in {"input", "env"} or token_source == "default":
            values["max_new_tokens"] = MAX_NEW_TOKENS_BY_LENGTH[length]
            sources["max_new_tokens"] = f"length:{length}"

    prompt_text = ""
    if not errors:
        try:
            prompt_text, prompt_warnings = assemble_prompt(
                mode=values["mode"],
                length=values["length"],
                language=values["language"],
                custom_prompt=values["prompt"] or None,
                question=values["question"] or None,
            )
            warnings.extend(prompt_warnings)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ConfigError(errors)

    values["preset"] = preset_name
    sources.setdefault("preset", "default" if preset_name == "default" else sources.get("preset", "input"))
    return ResolvedConfig(
        values=values,
        sources=sources,
        warnings=warnings,
        prompt_text=prompt_text,
    )
