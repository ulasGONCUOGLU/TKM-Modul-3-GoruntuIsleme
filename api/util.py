"""
api/util.py
───────────
Küçük yardımcılar: dosya adı temizleme, JPEG kodlama.
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np

_UNSAFE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def safe_filename(name: str, default: str = "dosya") -> str:
    """Yol bileşenlerini ve tehlikeli karakterleri atar; uzantıyı küçük harfe çevirir."""
    base = Path(str(name).replace("\\", "/")).name
    stem, ext = os.path.splitext(base)
    stem = _UNSAFE.sub("_", stem).strip(" ._")[:80] or default
    ext = _UNSAFE.sub("", ext).lower()[:10]
    return stem + ext


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG kodlanamadı")
    return buf.tobytes()


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    return encode_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), quality)
