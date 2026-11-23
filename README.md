# Qwen-VL Batch Captioner for Runpod

[![Runpod](https://img.shields.io/badge/Runpod-Hub-7c3aed)](https://console.runpod.io/hub/listing/prilojject/runpod-qwen-vl-caption)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A **Runpod Serverless** worker that batch-captions images for **LoRA / diffusion datasets** with official [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

**Send** a zip (or one `image_url`). **Get** a kohya-style sidecar **`.txt` next to each image**, plus a `captions.json` manifest. Optionally ask a **single VQA question** and get a single answer — this is not a general chat or vLLM endpoint.

It is a headless Hugging Face Transformers worker using the official `apply_chat_template` + `generate` loop. Gradio is never started.

This is **not** [runpod-joycaption](https://github.com/prilojject/runpod-joycaption) (that one writes formal / descriptive / tag-style captions with JoyCaption Beta One). It is also **not** [runpod-image-lora-trainer](https://github.com/prilojject/runpod-image-lora-trainer) (that trains a LoRA; this only writes captions those trainers can read).

Qwen3-VL code and the 2B / 4B / 8B Instruct weights are licensed under **Apache-2.0** (© Alibaba Cloud / the Qwen team). The worker code in this repository is MIT. See [License](#license).

## What you send / what you get

**Send** `image_url` and/or `dataset_url` (a `.zip` or `.tar.gz` of jpg/png/webp/…). Nested folders are kept. Existing sidecar `.txt` files are skipped unless you set `overwrite: true`.

```
images.zip
  001.jpg
  002.png
  sub/003.webp
```

**Get** the same tree with captions beside each file, plus a manifest:

```
001.jpg
001.txt
002.png
002.txt
sub/003.webp
sub/003.txt
captions.json
```

Delivered as a zip to a Hugging Face repo, an S3-compatible bucket, a presigned PUT URL, or a Runpod network volume (`/runpod-volume/qwen-vl/runs/<run_name>/`). If none of those are configured, the job can still return the captions in the response, and a warning that the files will disappear when the worker stops.

## Defaults

| Setting | Default | Notes |
| --- | --- | --- |
| Model | **Qwen3-VL-4B-Instruct** | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Dtype | **bfloat16** (not fp16) | Native weight dtype; ~8.27 GiB on disk, comfortable on a 24 GB GPU |
| GPU | **RTX 4090 24 GB** | Default / smoke path; batch size 1 |
| Mode | **caption**, length **medium**, language **en** | Prompt: `Describe this image.` |
| Pixel budget | `longest_edge=1280*32*32`, `shortest_edge=256*32*32` | Official README range. The processor default of 16.7M pixels is **not** used |
| Decode | sampling T=0.7, top-p=0.8, top-k=20 | Set `greedy: true` for deterministic smoke tests |
| `max_new_tokens` | **256** (medium) | 128 short / 512 long |
| `refresh_worker` | **false** | The model stays in worker memory |

Public weights (~8.27 GiB, no token, Apache-2.0) are downloaded at job start and cached under `/runpod-volume/qwen-vl/models/` when a network volume is attached. They are **not** baked into the image. Without a volume, every cold start re-downloads them — attach a volume for anything other than a one-off test.

Upstream Qwen3-VL code is pinned to SHA `96588727e44c78b25ba03ea03b8e12f7e64fd0da` (the official repo has no tags). That SHA is the documentation / API pin only. Releases of *this* repo use `targetCommitish=main`. The image installs a **released** `transformers==4.57.6` (≥ 4.57.0, when Qwen3-VL landed). Do not follow the October 2025 model card advice to install transformers from git-main; that card is stale.

Optional checkpoints (same Apache-2.0 license, not the smoke path): `Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen3-VL-8B-Instruct`. 8B (~16.33 GiB weights) is tight on 24 GB and **requires** the official pixel budget; it is not the default.

## Modes

- **`caption`** (default): short / medium / long English or Chinese prompts. Medium English is the official `Describe this image.` Long English is `Write a detailed caption for this image.` Chinese is first-class (`请描述这张图片。` for medium). Other language codes add an explicit “Write the caption in …” instruction.
- **`vqa`**: send a single `question`. The worker returns one answer written to the sidecar `.txt`. There is no chat history, no tool use, and no OpenAI-compatible server.

A custom `prompt` overrides the caption templates. In `vqa` mode the question is what gets sent.

## Example job

```json
{
  "input": {
    "dataset_url": "https://example.com/images.zip",
    "mode": "caption",
    "length": "medium",
    "language": "en"
  }
}
```

Quick test (greedy, 2–3 images):

```json
{
  "input": {
    "dataset_url": "https://raw.githubusercontent.com/prilojject/runpod-qwen-vl-caption/main/examples/datasets/tiny_images.zip",
    "mode": "caption",
    "length": "medium",
    "language": "en",
    "greedy": true,
    "max_new_tokens": 128,
    "run_name": "hub-test-qwen-vl"
  }
}
```

Short VQA:

```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg",
    "mode": "vqa",
    "question": "What is written on the sign?"
  }
}
```

Other useful input fields: `overwrite`, `prepend`, `append`, `max_pixels` / `min_pixels`, `hf_repo_id`, `upload_url`, `return_captions_in_response`, `seed`, `dry_run`. Do **not** put tokens in the job input; use Runpod secrets / environment variables (`HF_TOKEN`, `BUCKET_*`).

## FAQ

**How is this different from JoyCaption?** JoyCaption (a separate listing) is built around official JoyCaption Beta One / Llama 3.1 formal, descriptive, and tag-style prompts. This worker uses Qwen3-VL, official English/Chinese caption prompts, OCR-friendly multilingual coverage, and optional short VQA. The sidecar `.txt` + `captions.json` output shape is the same idea, so either can feed a LoRA trainer.

**Is this a Qwen-VL chat API?** No. There is no vLLM server, no OpenAI-compatible `/v1/chat/completions`, and no conversation history. Send images plus a caption request or one question.

**Why not Qwen2.5-VL-3B?** The 3B Qwen2.5-VL weights use the Qwen Research License (non-commercial). This listing stays on Qwen3-VL Instruct checkpoints, which the Hugging Face cards mark Apache-2.0.

**Will it fit a 24 GB GPU?** Yes for the default 4B Instruct in bfloat16 with the official 1280×32×32 pixel budget. 8B can fit only with that budget; do not leave the processor’s 16.7M-pixel default in place.

**Are the weights in the Docker image?** No. They download at job start and cache on a network volume when one is attached.

**Can I keep the model loaded between jobs?** Yes. The worker sets `refresh_worker=false` and runs `generate()` in-process, so the 4B model stays in memory for the next job on the same worker.

## License

- **Worker code** in this repository: [MIT](LICENSE), © 2026 prilojject.
- **Upstream Qwen3-VL code** (QwenLM/Qwen3-VL @ `96588727e44c78b25ba03ea03b8e12f7e64fd0da`) and **qwen-vl-utils**: [Apache-2.0](LICENSE.upstream), © Alibaba Cloud / the Qwen team.
- **Default and optional Instruct weights** (4B / 2B / 8B): Apache-2.0 according to the Hugging Face model cards (`gated: false`). The weight repos do not ship a LICENSE file; this README and [NOTICE](NOTICE) record the license and attribution.

Example images in `examples/datasets/tiny_images.zip` are small public-domain NASA photographs. See [examples/SOURCES.md](examples/SOURCES.md).
