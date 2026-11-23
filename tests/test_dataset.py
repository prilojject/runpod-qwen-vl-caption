import io
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.dataset import (
    MAX_IMAGE_PIXELS,
    DatasetError,
    _extract,
    _safe_member,
    collect_images,
    select_images_for_caption,
    validate_image,
)


def _rgb(path: Path, size=(32, 32), color=(10, 20, 30)):
    Image.new("RGB", size, color).save(path)


def test_zip_slip_is_rejected(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("ok/hello.txt", "safe")
        z.writestr("../escape.txt", "nope")
        z.writestr("/abs/escape2.txt", "nope")
        info = zipfile.ZipInfo("nested/../../outside.txt")
        z.writestr(info, "nope")
    dest = tmp_path / "out"
    slipped = _extract(str(zpath), str(dest))
    assert slipped >= 1
    # Nothing escaped the dest root.
    root = dest.resolve()
    for p in dest.rglob("*"):
        assert str(p.resolve()).startswith(str(root))
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "outside.txt").exists()
    assert (dest / "ok" / "hello.txt").is_file()


def test_safe_member_blocks_dotdot(tmp_path):
    base = str(tmp_path / "root")
    os.makedirs(base)
    assert _safe_member(base, "../x") is None
    assert _safe_member(base, "/etc/passwd") is None
    assert _safe_member(base, "ok/img.jpg") is not None


def test_pixel_limit_rejects_huge_image(tmp_path, monkeypatch):
    # Use a temporarily low cap so the test stays cheap.
    import src.dataset as ds
    monkeypatch.setattr(ds, "MAX_IMAGE_PIXELS", 1000)
    path = tmp_path / "big.jpg"
    # 40x40 = 1600 pixels > 1000
    Image.new("RGB", (40, 40), (1, 2, 3)).save(path)
    ok, err = ds.validate_image(str(path))
    assert ok is False
    assert err and ("exceeds" in err or "decompression" in err.lower() or "bomb" in err.lower())


def test_pixel_limit_accepts_small_image(tmp_path):
    path = tmp_path / "ok.png"
    _rgb(path, (64, 48))
    ok, err = validate_image(str(path))
    assert ok is True and err is None


def test_byte_limit_rejects_large_file(tmp_path, monkeypatch):
    import src.dataset as ds
    monkeypatch.setattr(ds, "MAX_IMAGE_BYTES", 200)
    path = tmp_path / "fat.jpg"
    # Uncompressed-ish PNG/JPEG of 200x200 will exceed 200 bytes.
    Image.new("RGB", (200, 200), (9, 9, 9)).save(path, quality=95)
    ok, err = ds.validate_image(str(path))
    assert ok is False
    assert "byte limit" in (err or "")


def test_skip_and_overwrite_sidecar_txt(tmp_path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    _rgb(img1)
    _rgb(img2)
    (tmp_path / "a.txt").write_text("already captioned", encoding="utf-8")
    to_cap, skipped = select_images_for_caption([str(img1), str(img2)], overwrite=False)
    assert [Path(p).name for p in skipped] == ["a.jpg"]
    assert [Path(p).name for p in to_cap] == ["b.jpg"]
    assert (tmp_path / "a.txt").read_text() == "already captioned"

    to_cap, skipped = select_images_for_caption([str(img1), str(img2)], overwrite=True)
    assert skipped == []
    assert {Path(p).name for p in to_cap} == {"a.jpg", "b.jpg"}
    assert not (tmp_path / "a.txt").exists()


def test_collect_images_ignores_junk_and_keeps_nested(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "__MACOSX").mkdir()
    _rgb(tmp_path / "sub" / "cat.png")
    _rgb(tmp_path / ".hidden.jpg")
    _rgb(tmp_path / "__MACOSX" / "._x.jpg")
    (tmp_path / "notes.txt").write_text("not an image")
    found = collect_images(str(tmp_path))
    assert [os.path.relpath(p, tmp_path) for p in found] == [os.path.join("sub", "cat.png")]


def test_default_pixel_cap_is_a_sane_decode_guard():
    # File-level cap is a decompression-bomb guard applied *before* the
    # processor resizes to 1280*32*32. 40 MP is well under PIL's historical
    # 933 MP official-script-style caps and far above the visual-token budget.
    assert MAX_IMAGE_PIXELS <= 50_000_000
    assert MAX_IMAGE_PIXELS >= 1_000_000
