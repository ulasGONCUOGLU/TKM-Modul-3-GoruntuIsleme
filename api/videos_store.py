"""
api/videos_store.py
───────────────────
Yüklenen videoların diskteki yerleşimi ve okunması.

Dosya adı:  dosyalar/yuklenen/<12 haneli_id>__<temizlenmiş ad>.<uzantı>
Alanlar  :  dosyalar/alanlar/video_<id>.json   (core/areas.py biçimi)
"""

import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from api.config import UPLOAD_DIR
from core.paths import AREAS_DIR

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def valid_id(video_id: str) -> bool:
    return bool(_ID_RE.match(video_id or ""))


def video_filename(video_id: str, safe_name: str) -> str:
    return f"{video_id}__{safe_name}"


def find_video(video_id: str) -> Path | None:
    if not valid_id(video_id) or not UPLOAD_DIR.exists():
        return None
    for p in UPLOAD_DIR.glob(f"{video_id}__*"):
        if p.is_file() and not p.name.endswith(".part"):
            return p
    return None


def display_name(path: Path) -> str:
    return path.name.split("__", 1)[1] if "__" in path.name else path.name


def read_info(path: Path) -> dict | None:
    """Çözünürlük, FPS, kare sayısı. Açılamazsa None."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if w <= 0 or h <= 0:
            return None
        return {
            "width": w,
            "height": h,
            "fps": round(fps, 3) if fps > 0 else None,
            "frame_count": count,
            "duration_sec": round(count / fps, 1) if fps > 0 and count > 0 else None,
        }
    finally:
        cap.release()


def read_first_frame(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def suggest_frame_step(fps: float) -> int:
    """Hızlı analiz için önerilen kare atlama: etkin analiz hızı ≈ 10 kare/sn (en çok her 6. kare)."""
    try:
        return max(1, min(6, round(float(fps) / 10.0)))
    except (TypeError, ValueError):
        return 1


def describe(path: Path, with_info: bool = True) -> dict:
    vid = path.name.split("__", 1)[0]
    d = {
        "id": vid,
        "name": display_name(path),
        "size_bytes": path.stat().st_size,
        "uploaded_at": path.stat().st_mtime,
    }
    if with_info:
        info = read_info(path) or {}
        d.update(info)
        if info:
            d["suggested_frame_step"] = suggest_frame_step(info.get("fps", 30))
    return d


def list_videos() -> list[dict]:
    if not UPLOAD_DIR.exists():
        return []
    items = [p for p in UPLOAD_DIR.iterdir()
             if p.is_file() and "__" in p.name and not p.name.endswith(".part")
             and valid_id(p.name.split("__", 1)[0])]
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [describe(p) for p in items]


def areas_file(video_id: str) -> Path:
    return AREAS_DIR / f"video_{video_id}.json"


def delete_video(video_id: str) -> bool:
    p = find_video(video_id)
    if p is None:
        return False
    p.unlink(missing_ok=True)
    areas_file(video_id).unlink(missing_ok=True)
    return True
