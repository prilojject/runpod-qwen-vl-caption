"""Fetch, unpack and validate images for the Qwen-VL caption worker (zip-slip and decompression-bomb safe)."""

from __future__ import annotations

import io
import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageFile, UnidentifiedImageError

from .secret_utils import redact

log = logging.getLogger(__name__)

# Enforce our own pixel/byte limits before images reach the model. This is
# a zip-bomb / decompression-bomb guard on top of the official 1280*32*32
# processor budget (which only applies after a successful decode).
MAX_DOWNLOAD_BYTES = 2 * 2**30
MAX_UNPACKED_BYTES = 2 * 2**30
MAX_IMAGE_BYTES = 40 * 2**20  # 40 MiB per file
MAX_IMAGE_PIXELS = 40_000_000  # 40 megapixels (e.g. ~6324×6324)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")
CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


class DatasetError(ValueError):
    pass


@dataclass
class DatasetReport:
    source: str
    images_found: int = 0
    images_used: int = 0
    skipped_existing_txt: int = 0
    read_errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    example_files: list[str] = field(default_factory=list)
    images_dir: str = ""
    image_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "images_found": self.images_found,
            "images_used": self.images_used,
            "skipped_existing_txt": self.skipped_existing_txt,
            "read_errors": self.read_errors[:20],
            "skipped": self.skipped[:20],
            "example_files": self.example_files[:8],
            "images_dir": self.images_dir,
        }


def _download(url: str, dest: str, limit: int = MAX_DOWNLOAD_BYTES) -> tuple[str, str]:
    log.info("downloading %s", "image" if dest.lower().endswith(IMAGE_EXTS) else "dataset archive")
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as r:
        if r.status_code != 200:
            raise DatasetError(f"download returned HTTP {r.status_code}")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > limit:
                    raise DatasetError(f"download is larger than {limit} bytes")
                f.write(chunk)
        return dest, r.headers.get("Content-Type", "")


def _safe_member(base: str, name: str) -> str | None:
    # Zip-slip: refuse members that would escape dest (including names with .. or abs paths).
    # Also reject absolute paths and Windows drive prefixes before joining.
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/") or cleaned.startswith("\\") or (len(cleaned) > 1 and cleaned[1] == ":"):
        return None
    target = os.path.realpath(os.path.join(base, cleaned))
    root = os.path.realpath(base)
    if not target.startswith(root + os.sep) and target != root:
        return None
    return target


def _extract(archive: str, dest: str) -> int:
    """Extract a zip/tar into dest. Returns the number of members skipped as zip-slip."""
    os.makedirs(dest, exist_ok=True)
    total = 0
    slipped = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                # Some zips lie about file_size; also cap the bytes we actually write.
                total += max(m.file_size, 0)
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 2 GiB")
                target = _safe_member(dest, m.filename)
                if target is None:
                    slipped += 1
                    log.warning("zip-slip member skipped: %s", m.filename)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                written = 0
                with z.open(m) as src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_UNPACKED_BYTES:
                            raise DatasetError("dataset archive unpacks to more than 2 GiB")
                        out.write(chunk)
        return slipped
    try:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                total += max(m.size, 0)
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 2 GiB")
                target = _safe_member(dest, m.name)
                if target is None:
                    slipped += 1
                    log.warning("tar-slip member skipped: %s", m.name)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = t.extractfile(m)
                if src is None:
                    continue
                written = 0
                with src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_UNPACKED_BYTES:
                            raise DatasetError("dataset archive unpacks to more than 2 GiB")
                        out.write(chunk)
    except tarfile.ReadError as exc:
        raise DatasetError("dataset_url must point to a .zip or .tar(.gz) archive") from exc
    return slipped


def _is_junk(name: str) -> bool:
    base = os.path.basename(name)
    rel = name.replace("\\", "/")
    return base.startswith(".") or base.startswith("__MACOSX") or "/__MACOSX/" in rel or base == "Thumbs.db"


def collect_images(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        for f in files:
            if _is_junk(f) or not f.lower().endswith(IMAGE_EXTS):
                continue
            out.append(os.path.join(dirpath, f))
    out.sort()
    return out


def validate_image(path: str) -> tuple[bool, str | None]:
    """Open the image under our pixel/byte limits before it reaches the processor."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"{os.path.basename(path)}: {exc}"
    if size > MAX_IMAGE_BYTES:
        return False, f"{os.path.basename(path)}: {size} bytes exceeds the {MAX_IMAGE_BYTES} byte limit"
    if size < 32:
        return False, f"{os.path.basename(path)}: file is too small to be an image"
    # Pillow's own decompression-bomb check uses MAX_IMAGE_PIXELS.
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    # Do not load truncated images as if they were complete.
    previous_trunc = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            pixels = int(w) * int(h)
            if pixels > MAX_IMAGE_PIXELS:
                return False, f"{os.path.basename(path)}: {w}x{h} exceeds the {MAX_IMAGE_PIXELS} pixel limit"
            # Force a decode of the first frame so bombs surface here, not in generate().
            im.load()
        return True, None
    except Image.DecompressionBombError as exc:
        return False, f"{os.path.basename(path)}: rejected as a decompression bomb ({exc})"
    except UnidentifiedImageError:
        return False, f"{os.path.basename(path)}: not a recognizable image"
    except Exception as exc:
        return False, f"{os.path.basename(path)}: {exc}"
    finally:
        Image.MAX_IMAGE_PIXELS = previous
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_trunc


def _guess_ext(url: str, content_type: str, data_head: bytes) -> str:
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1].lower()
    if suffix in IMAGE_EXTS:
        return suffix
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in CONTENT_TYPE_EXT:
        return CONTENT_TYPE_EXT[ctype]
    # sniff
    if data_head.startswith(b"\x89PNG"):
        return ".png"
    if data_head.startswith(b"\xff\xd8"):
        return ".jpg"
    if data_head.startswith(b"RIFF") and b"WEBP" in data_head[:16]:
        return ".webp"
    if data_head.startswith(b"GIF8"):
        return ".gif"
    return ".jpg"


def _add_single_image(url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(dest_dir, "download.bin")
    try:
        _, ctype = _download(url, tmp, limit=MAX_IMAGE_BYTES)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"failed to download image_url: {redact(str(exc)).splitlines()[0][:300]}") from exc
    with open(tmp, "rb") as f:
        head = f.read(16)
    ext = _guess_ext(url, ctype, head)
    dest = os.path.join(dest_dir, f"input{ext}")
    os.replace(tmp, dest)
    return dest


def prepare_dataset(cfg, work: str) -> DatasetReport:
    images_dir = os.path.join(work, "images")
    os.makedirs(images_dir, exist_ok=True)
    sources: list[str] = []

    if getattr(cfg, "dataset_url", ""):
        sources.append(cfg.dataset_url)
        raw_dir = os.path.join(work, "archive")
        archive = os.path.join(work, "dataset.bin")
        try:
            _download(cfg.dataset_url, archive)
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"failed to download dataset_url: {redact(str(exc)).splitlines()[0][:300]}") from exc
        try:
            slipped = _extract(archive, raw_dir)
        finally:
            try:
                os.remove(archive)
            except OSError:
                pass
        # Keep folder structure (kohya-style sidecar .txt lives next to each file).
        for path in collect_images(raw_dir):
            rel = os.path.relpath(path, raw_dir)
            dest = os.path.join(images_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.realpath(path) != os.path.realpath(dest):
                shutil.copy2(path, dest)
        # Also copy existing sidecar .txt so overwrite/skip can see them.
        for dirpath, dirnames, files in os.walk(raw_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
            for f in files:
                if f.lower().endswith(".txt") and not _is_junk(f):
                    src = os.path.join(dirpath, f)
                    rel = os.path.relpath(src, raw_dir)
                    dest = os.path.join(images_dir, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    if os.path.realpath(src) != os.path.realpath(dest):
                        shutil.copy2(src, dest)
        if slipped:
            # Counted as skipped members; not a hard error if real images remain.
            pass

    if getattr(cfg, "image_url", ""):
        sources.append(cfg.image_url)
        _add_single_image(cfg.image_url, os.path.join(images_dir, "_single"))

    found = collect_images(images_dir)
    report = DatasetReport(
        source=" + ".join(sources),
        images_found=len(found),
        images_dir=images_dir,
    )
    if not found:
        raise DatasetError("no jpg/jpeg/png/webp (or other Pillow-readable) images found")

    used: list[str] = []
    for path in found:
        ok, err = validate_image(path)
        if not ok:
            report.read_errors.append(err or os.path.basename(path))
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        used.append(path)
        rel = os.path.relpath(path, images_dir)
        if len(report.example_files) < 8:
            report.example_files.append(rel)

    report.image_paths = used
    report.images_used = len(used)
    if not used:
        raise DatasetError(
            "no usable images remained after validation "
            f"({len(report.read_errors)} rejected for size, pixels, or format)"
        )
    log.info("dataset ready: %s images", len(used))
    return report


def select_images_for_caption(paths: list[str], overwrite: bool) -> tuple[list[str], list[str]]:
    """Skip images that already have a sidecar .txt unless overwrite is set (kohya-style)."""
    to_caption: list[str] = []
    skipped: list[str] = []
    for path in paths:
        txt = os.path.splitext(path)[0] + ".txt"
        if os.path.isfile(txt) and not overwrite:
            skipped.append(path)
            continue
        if overwrite and os.path.isfile(txt):
            os.remove(txt)
        to_caption.append(path)
    return to_caption, skipped
