"""Static checks of .runpod/hub.json and tests.json against the spec."""
import json
import os

from src.config import PARAMS
from src.paths import DEFAULT_MAX_PIXELS, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, QWEN3_VL_SHA
from src.prompts import DEFAULT_PROMPT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN"}


def load(name):
    with open(os.path.join(ROOT, ".runpod", name)) as fh:
        return json.load(fh)


def test_required_files():
    for f in (
        "handler.py", "Dockerfile", "README.md", "LICENSE", "LICENSE.upstream",
        "NOTICE",
        ".runpod/hub.json", ".runpod/tests.json", ".runpod/icon.png",
        "examples/datasets/tiny_images.zip", "examples/SOURCES.md",
        "llms.txt", "AGENTS.md", "CITATION.cff", "openapi.yaml", "context7.json",
        "index.html",
    ):
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config", "iconUrl"):
        assert hub.get(k), k
    assert hub["type"] == "serverless" and hub["category"] == "image"
    assert "JoyCaption" in hub["description"] or "joycaption" in hub["description"].lower()
    assert "vLLM" in hub["description"] or "chat" in hub["description"].lower()
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU" and cfg["containerDiskInGb"] == 80 and cfg["gpuCount"] == 1
    assert "ADA_24" in cfg["gpuIds"] and "Blackwell" in cfg["gpuIds"] and cfg["gpuIds"].startswith("ADA_24")
    assert "1g.24gb" in cfg["gpuIds"] and "2g.48gb" in cfg["gpuIds"]
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in cfg["env"]]
    assert len(keys) == len(set(keys))
    required = {
        "MODEL_ID", "MODEL_REVISION", "MODE", "LANGUAGE", "DATASET_URL", "IMAGE_URL",
        "MAX_NEW_TOKENS", "GREEDY", "HF_OUTPUT_REPO", "HF_TOKEN",
        "CAPTION_TIMEOUT_SECONDS",
    }
    assert required <= set(keys)
    assert set(keys) <= known, set(keys) - known
    for preset in cfg["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]
    names = [p["name"] for p in cfg["presets"]]
    assert any("LoRA" in n or "dataset" in n.lower() or "caption" in n.lower() for n in names)
    assert any("Quick test" in n for n in names)
    assert any("Chinese" in n or "zh" in n.lower() for n in names)
    for e in cfg["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        if inp["type"] == "number" and "min" in inp:
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}
        for o in inp.get("options", []):
            assert o["label"] and o["value"] is not None


def test_tests_json():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert t["config"]["gpuCount"] == 1
    assert len(t["tests"]) == 1
    case = t["tests"][0]
    assert case["name"] == "qwen3_vl_4b_caption_smoke"
    assert case["input"]["model_id"] == DEFAULT_MODEL_ID
    assert case["input"]["mode"] == "caption"
    assert case["input"]["length"] == "medium"
    assert case["input"]["language"] == "en"
    assert case["input"]["greedy"] is True
    assert case["input"]["max_new_tokens"] == 128
    assert "tiny_images.zip" in case["input"]["dataset_url"]
    assert "prilojject/runpod-qwen-vl-caption/main" in case["input"]["dataset_url"]
    assert case.get("timeout", 0) >= 1_200_000


def test_dockerfile_pins_and_does_not_bake_weights():
    text = open(os.path.join(ROOT, "Dockerfile")).read()
    assert f"QWEN3_VL_SHA={QWEN3_VL_SHA}" in text
    lowered = text.lower()
    assert "git+https://github.com/huggingface/transformers" not in lowered
    assert "transformers==4.57.6" in text or "TRANSFORMERS_VERSION=4.57.6" in text
    assert "qwen-vl-utils==0.0.14" in open(os.path.join(ROOT, "requirements.txt")).read()
    assert "2.7.1" in text or "torch==2.7" in text
    assert "snapshot_download" not in text
    assert "huggingface-cli download" not in text
    assert "hf download" not in lowered
    assert "install vllm" not in lowered and "vllm==" not in lowered
    assert "install gradio" not in lowered and "gradio==" not in lowered
    assert "florence" not in lowered
    assert "Qwen2.5-VL" not in text
    assert 'CMD ["python", "-u", "/app/handler.py"]' in text


def test_readme_and_notice_cover_apache():
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    notice = open(os.path.join(ROOT, "NOTICE"), encoding="utf-8").read()
    for text in (readme, notice):
        assert "Apache-2.0" in text or "Apache License" in text
        assert "Alibaba" in text or "Qwen" in text
    assert "Qwen Research License" in readme or "Qwen2.5-VL-3B" in readme
    assert "JoyCaption" in readme or "joycaption" in readme.lower()
    assert "vLLM" in readme or "chat" in readme.lower()
    assert DEFAULT_MODEL_ID in readme
    assert QWEN3_VL_SHA in readme
    assert DEFAULT_MODEL_REVISION in readme
    assert DEFAULT_PROMPT in readme
    assert "1280*32*32" in readme or str(DEFAULT_MAX_PIXELS) in readme
    assert "bfloat16" in readme
    assert "not baked" in readme.lower() or "not** baked" in readme.lower() or "are **not** baked" in readme


def test_handler_refresh_worker_false():
    text = open(os.path.join(ROOT, "handler.py"), encoding="utf-8").read()
    assert "refresh_worker\"] = False" in text or "refresh_worker'] = False" in text


def test_prompts_are_not_joycaption_templates():
    text = open(os.path.join(ROOT, "src/prompts.py"), encoding="utf-8").read()
    assert "Write a long descriptive caption for this image in a formal tone." not in text
    assert "Danbooru" not in text
    assert DEFAULT_PROMPT in text
