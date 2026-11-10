"""
api/api_settings.py
───────────────────
Yalnızca web akışına ait ayarlar (core.config.Settings'te olmayanlar).
Aynı settings tablosunda "api." önekiyle saklanır.
"""

from core.database import get_setting, set_setting

DEFAULT_STREAM_MAX_WIDTH = 960
DEFAULT_JPEG_QUALITY = 80


def _int(key: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(float(get_setting(key) or default))
    except ValueError:
        v = default
    return min(max(v, lo), hi)


def stream_max_width() -> int:
    return _int("api.stream_max_width", DEFAULT_STREAM_MAX_WIDTH, 320, 3840)


def jpeg_quality() -> int:
    return _int("api.jpeg_quality", DEFAULT_JPEG_QUALITY, 30, 95)


def get_all() -> dict:
    return {"stream_max_width": stream_max_width(), "jpeg_quality": jpeg_quality()}


def set_stream_max_width(v: int) -> None:
    set_setting("api.stream_max_width", str(int(v)))


def set_jpeg_quality(v: int) -> None:
    set_setting("api.jpeg_quality", str(int(v)))
