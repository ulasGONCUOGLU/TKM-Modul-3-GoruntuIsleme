"""
core/text.py
────────────
cv2.putText Türkçe karakterleri (ı, ş, ğ, İ ...) çizemez ve '?' basar.
Burada PIL + sistemde bulunan bir TrueType yazı tipi kullanılır; yazı tipi
bulunamazsa ASCII'ye çevrilerek cv2'ye düşülür. Yalnızca yazının
kapladığı küçük bölge dönüştürüldüğü için maliyeti düşüktür.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


@lru_cache(maxsize=1)
def find_font_path() -> str | None:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=16)
def _font(size: int):
    path = find_font_path()
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def put_text(frame: np.ndarray, text: str, org: tuple[int, int], size: int = 20,
             color_bgr=(255, 255, 255), bg_bgr=None) -> None:
    """`org` = yazının SOL-ÜST köşesi. frame yerinde değiştirilir."""
    x, y = int(org[0]), int(org[1])
    font = _font(size)
    h_img, w_img = frame.shape[:2]

    if font is None:  # yedek yol: ASCII
        t = text.translate(_TR_MAP)
        scale = size / 30.0
        (tw, th), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if bg_bgr is not None:
            cv2.rectangle(frame, (x - 3, y - 3), (x + tw + 3, y + th + base + 3), bg_bgr, -1)
        cv2.putText(frame, t, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, 2, cv2.LINE_AA)
        return

    l, t_, r, b = font.getbbox(text)
    tw, th = r - l, b - t_
    pad = 4
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + tw + pad + 2, w_img), min(y + th + pad + 2, h_img)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    if bg_bgr is not None:
        d.rectangle([0, 0, x1 - x0, y1 - y0], fill=tuple(bg_bgr[::-1]))
    d.text((x - x0 - l + pad // 2, y - y0 - t_ + pad // 2), text, font=font,
           fill=tuple(color_bgr[::-1]))
    frame[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
