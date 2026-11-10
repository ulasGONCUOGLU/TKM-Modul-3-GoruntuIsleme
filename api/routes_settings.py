"""
api/routes_settings.py
──────────────────────
Analiz ayarları (core.config.Settings) + web akışı ayarları.
"""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from api import api_settings
from api.schemas import SettingsPatch
from api.state import state
from core.config import MODE_IMGSZ, TRACKERS, Settings

router = APIRouter(tags=["ayarlar"])


def _payload() -> dict:
    s = Settings.load()
    d = asdict(s)
    d["allowed_classes"] = list(s.allowed_classes)
    d.update(api_settings.get_all())
    return {
        "settings": d,
        "choices": {"tracker": list(TRACKERS), "mode": list(MODE_IMGSZ),
                    "device": ["auto", "cpu", "cuda:0", "mps"]},
    }


@router.get("/settings")
def get_settings():
    return _payload()


@router.put("/settings")
def put_settings(body: SettingsPatch):
    cur = state.sessions.current
    if cur is not None and cur.is_active:
        raise HTTPException(409, "Analiz sürerken ayar değiştirilemez.")
    data = body.model_dump(exclude_none=True)
    s = Settings.load()
    for key in ("stream_max_width", "jpeg_quality"):
        v = data.pop(key, None)
        if v is not None:
            (api_settings.set_stream_max_width if key == "stream_max_width"
             else api_settings.set_jpeg_quality)(v)
    if "allowed_classes" in data:
        data["allowed_classes"] = tuple(data["allowed_classes"])
    for k, v in data.items():
        setattr(s, k, v)
    s.save()
    return _payload()
