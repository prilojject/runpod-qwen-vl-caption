# Agent notes — runpod-qwen-vl-caption

## What this worker is

A Runpod Serverless **headless Qwen3-VL batch captioner** for LoRA / diffusion datasets. Input: one `image_url` and/or a zip of images. Output: kohya-style sidecar `.txt` next to each image, plus `captions.json`. Optional short VQA (one `question` → one answer).

Use Hugging Face Transformers and the official chat template on a released `transformers>=4.57.0`. Do **not** start Gradio or a vLLM / OpenAI-compatible chat server. Do **not** fold this into `runpod-joycaption` or `runpod-image-lora-trainer`. Do **not** add Florence-2 or Qwen2.5-VL. Do **not** bake weights into the image.

## Pinned upstream

- Code (documentation / API pin): `QwenLM/Qwen3-VL` @ SHA `96588727e44c78b25ba03ea03b8e12f7e64fd0da` (written to `/opt/qwen3-vl.commit`). There are no upstream tags. Do **not** clone a floating `main` as the runtime.
- Weights: `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` (~8.27 GiB, public, Apache-2.0, no token). Cache under `/runpod-volume/qwen-vl/models/`. Optional: 2B `@ 89644892e4d85e24eaac8bacfd4f463576704203`, 8B `@ 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- The SHA is the upstream pin only. GitHub releases of *this* repo must use `targetCommitish=main` (a SHA target breaks Runpod Hub's `git clone -b`).
- The 4B model card still tells people to install transformers from git-main because “4.57.0 is not released”. That is stale. Install a released pin (`transformers==4.57.6` in the Dockerfile). `qwen-vl-utils==0.0.14`.

## Defaults to keep

- 4B Instruct, **bfloat16** (not fp16), default GPU 24 GB (RTX 4090), `batch_size=1`
- Mode: caption, medium, English. Default prompt: `Describe this image.`
- Official pixel budget: `longest_edge=1280*32*32`, `shortest_edge=256*32*32`. Never leave the processor default of 16.7M pixels.
- `max_new_tokens=256` for medium (128 short / 512 long)
- Decode: sampling T=0.7, top-p=0.8, top-k=20; greedy is an option (use it for smoke)
- `refresh_worker=false` (load the model in-process and keep it in memory)
- `containerDiskInGb=80`
- Official `apply_chat_template` + trim `out_ids[len(in_ids):]`. Do not invent a message wrapper. Do not mix AutoProcessor resize with `qwen-vl-utils.process_vision_info` in the same run.
- 8B is an option only; do not smoke it and do not make it the default.

## License (strict)

- Worker code: MIT
- Upstream code and 2B/4B/8B Instruct weights: **Apache-2.0**. README and NOTICE must say so and attribute Qwen / Alibaba Cloud. Ship `LICENSE.upstream`. The weight repos have no LICENSE file — do not invent a different one. Never substitute Qwen2.5-VL-3B (Research License, non-commercial).

## Tests

- Unit tests (`tests/`) must run without a GPU and without model weights.
- The Hub smoke test is one case only: `qwen3_vl_4b_caption_smoke` — 4B, caption medium en, greedy, 128 tokens, `tiny_images.zip` (NASA public-domain images documented in `examples/SOURCES.md`), RTX 4090, timeout 20 min.
- After a paid GPU test: delete the endpoint, pod and registry auth. Confirm with GraphQL that leftover smoke endpoints are gone. Leave protected pods `92anyumh6q7jrf` and `oejzh2afh8hcym` (and volumes `13faqbphuo`, `fv2c638zex`) untouched.

## Secrets

Never print `RUNPOD_API_KEY`, `HF_TOKEN`, `gh` tokens, or `BUCKET_*` values. Never accept tokens in the job input.
