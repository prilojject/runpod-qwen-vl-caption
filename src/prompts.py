"""Short official-style prompts for Qwen3-VL caption and VQA.

These are simple English / Chinese instructions in the style of the official
Qwen3-VL README ("Describe this image."), not JoyCaption formal templates.
"""

from __future__ import annotations

# Official README caption prompt for the default (medium, English) path.
DEFAULT_PROMPT = "Describe this image."

CAPTION_PROMPTS = {
    ("en", "short"): "Write a short caption for this image.",
    ("en", "medium"): DEFAULT_PROMPT,
    ("en", "long"): "Write a detailed caption for this image.",
    ("zh", "short"): "请为这张图片写一个简短的描述。",
    ("zh", "medium"): "请描述这张图片。",
    ("zh", "long"): "请为这张图片写一个详细的描述。",
}

LENGTHS = ("short", "medium", "long")


def assemble_prompt(
    mode: str,
    length: str,
    language: str,
    custom_prompt: str | None = None,
    question: str | None = None,
) -> tuple[str, list[str]]:
    """Return (prompt_text, warnings)."""
    warnings: list[str] = []
    mode = (mode or "caption").strip().lower()
    length = (length or "medium").strip().lower()
    language = (language or "en").strip().lower()

    if mode not in {"caption", "vqa"}:
        raise ValueError("mode: expected 'caption' or 'vqa'")
    if length not in LENGTHS:
        raise ValueError(f"length: expected one of {LENGTHS}, got {length!r}")

    if mode == "vqa":
        q = (question or "").strip()
        if not q:
            raise ValueError("question is required for vqa mode (this is not a general chat endpoint)")
        if custom_prompt:
            warnings.append("custom prompt is ignored in vqa mode; the question is sent as the user text")
        return q, warnings

    if custom_prompt:
        return custom_prompt.strip(), warnings

    key = (language, length)
    if key in CAPTION_PROMPTS:
        return CAPTION_PROMPTS[key], warnings

    # Other languages: keep the official English instruction and add an explicit
    # language request (should-path in the spec).
    base = CAPTION_PROMPTS[("en", length)]
    prompt = f"{base} Write the caption in {language}."
    warnings.append(
        f"language {language!r} has no first-class template; the English {length} prompt "
        "was used with an explicit 'Write the caption in …' instruction"
    )
    return prompt, warnings
