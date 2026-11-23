import pytest

from src.config import ConfigError, resolve_config
from src.paths import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    MODEL_REVISIONS,
)
from src.prompts import DEFAULT_PROMPT


def test_default_caption_medium_en():
    cfg = resolve_config({"dataset_url": "https://example.com/a.zip"})
    assert cfg.mode == "caption"
    assert cfg.length == "medium"
    assert cfg.language == "en"
    assert cfg.max_new_tokens == 256
    assert cfg.greedy is False
    assert cfg.overwrite is False
    assert cfg.batch_size == 1
    assert cfg.model_id == DEFAULT_MODEL_ID
    assert cfg.model_revision == DEFAULT_MODEL_REVISION
    assert cfg.max_pixels == DEFAULT_MAX_PIXELS == 1280 * 32 * 32
    assert cfg.min_pixels == DEFAULT_MIN_PIXELS == 256 * 32 * 32
    assert cfg.prompt_text == DEFAULT_PROMPT == "Describe this image."
    pub = cfg.public_dict()
    assert pub["prompt"] == DEFAULT_PROMPT
    assert pub["dtype"] == "bfloat16"
    assert pub["max_pixels"] == DEFAULT_MAX_PIXELS


def test_length_sets_max_new_tokens_when_unset():
    short = resolve_config({"dataset_url": "https://example.com/a.zip", "length": "short"})
    assert short.max_new_tokens == 128
    assert short.prompt_text == "Write a short caption for this image."
    long = resolve_config({"dataset_url": "https://example.com/a.zip", "length": "long"})
    assert long.max_new_tokens == 512
    assert long.prompt_text == "Write a detailed caption for this image."


def test_explicit_max_new_tokens_wins_over_length():
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "length": "long",
        "max_new_tokens": 64,
    })
    assert cfg.max_new_tokens == 64


def test_quick_preset_is_greedy():
    cfg = resolve_config({"image_url": "https://example.com/a.jpg", "preset": "quick"})
    assert cfg.greedy is True
    assert cfg.max_new_tokens == 128
    assert cfg.prompt_text == DEFAULT_PROMPT


def test_chinese_preset():
    cfg = resolve_config({"dataset_url": "https://example.com/a.zip", "preset": "chinese"})
    assert cfg.language == "zh"
    assert cfg.prompt_text == "请描述这张图片。"


def test_zh_long_prompt():
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "language": "zh",
        "length": "long",
    })
    assert cfg.prompt_text == "请为这张图片写一个详细的描述。"
    assert cfg.max_new_tokens == 512


def test_other_language_adds_instruction():
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "language": "fr",
    })
    assert "Write the caption in fr." in cfg.prompt_text
    assert any("no first-class template" in w for w in cfg.warnings)


def test_vqa_requires_question():
    with pytest.raises(ConfigError, match="question is required"):
        resolve_config({"dataset_url": "https://example.com/a.zip", "mode": "vqa"})


def test_vqa_uses_question_as_prompt():
    cfg = resolve_config({
        "image_url": "https://example.com/a.jpg",
        "mode": "vqa",
        "question": "What color is the car?",
    })
    assert cfg.prompt_text == "What color is the car?"
    assert cfg.mode == "vqa"


def test_rejects_chat_messages():
    with pytest.raises(ConfigError, match="not a general chat endpoint"):
        resolve_config({
            "dataset_url": "https://example.com/a.zip",
            "messages": [{"role": "user", "content": "hi"}],
        })


def test_rejects_qwen25_and_unknown_models():
    with pytest.raises(ConfigError, match="Qwen Research License"):
        resolve_config({
            "dataset_url": "https://example.com/a.zip",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        })
    with pytest.raises(ConfigError, match="Florence"):
        resolve_config({
            "dataset_url": "https://example.com/a.zip",
            "model_id": "microsoft/Florence-2-large",
        })
    with pytest.raises(ConfigError, match="Thinking"):
        resolve_config({
            "dataset_url": "https://example.com/a.zip",
            "model_id": "Qwen/Qwen3-VL-4B-Thinking",
        })


def test_2b_gets_pinned_revision():
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    })
    assert cfg.model_revision == MODEL_REVISIONS["Qwen/Qwen3-VL-2B-Instruct"]
    eight = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    })
    assert eight.model_revision == MODEL_REVISIONS["Qwen/Qwen3-VL-8B-Instruct"]
    assert any("8B" in w and "24 GB" in w for w in eight.warnings)


def test_input_overrides_env_and_preset(monkeypatch):
    monkeypatch.setenv("MODE", "vqa")
    monkeypatch.setenv("MAX_NEW_TOKENS", "128")
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "mode": "caption",
        "max_new_tokens": 256,
    })
    assert cfg.mode == "caption"
    assert cfg.max_new_tokens == 256
    assert cfg.sources["mode"] == "input"
    assert cfg.sources["max_new_tokens"] == "input"


def test_env_overrides_preset(monkeypatch):
    monkeypatch.setenv("DATASET_URL", "https://example.com/from-env.zip")
    monkeypatch.setenv("GREEDY", "true")
    cfg = resolve_config({}, env=None)
    assert cfg.dataset_url.endswith("from-env.zip")
    assert cfg.greedy is True
    assert cfg.sources["dataset_url"] == "env"


def test_requires_image_or_dataset():
    with pytest.raises(ConfigError, match="image_url or dataset_url"):
        resolve_config({"mode": "caption"})


def test_rejects_bad_urls():
    with pytest.raises(ConfigError, match="dataset_url must be an http"):
        resolve_config({"dataset_url": "ftp://x/y.zip"})
    with pytest.raises(ConfigError, match="image_url must be an http"):
        resolve_config({"image_url": "not-a-url"})


def test_rejects_secrets_in_input():
    with pytest.raises(ConfigError, match="secrets must not"):
        resolve_config({"dataset_url": "https://example.com/a.zip", "hf_token": "hf_xxx"})
    with pytest.raises(ConfigError, match="secrets must not"):
        resolve_config({"dataset_url": "https://example.com/a.zip", "api_key": "x"})


def test_redacts_upload_url_in_public_dict():
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "upload_url": "https://bucket.example/presigned?sig=secret",
    })
    assert cfg.public_dict()["upload_url"] == "<presigned url>"


def test_custom_prompt_overrides_template():
    cfg = resolve_config({
        "dataset_url": "https://example.com/a.zip",
        "prompt": "Describe the objects in one sentence.",
    })
    assert cfg.prompt_text == "Describe the objects in one sentence."


def test_unknown_mode():
    with pytest.raises(ConfigError, match="mode"):
        resolve_config({"dataset_url": "https://example.com/a.zip", "mode": "descriptive"})
