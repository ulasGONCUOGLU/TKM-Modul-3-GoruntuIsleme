"""
api/schemas.py
──────────────
İstek gövdesi şemaları (pydantic). Doğrulama hataları FastAPI tarafından 422 olarak döner.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from core.config import MODE_IMGSZ, TRACKERS


class AreaIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    points: list[tuple[int, int]] = Field(min_length=3, max_length=500)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Alan adı boş olamaz")
        return v


class AreasPayload(BaseModel):
    areas: list[AreaIn] = Field(max_length=50)
    # Noktaların ait olduğu çözünürlük. Verilmezse videonun çözünürlüğü kabul edilir.
    frame_width: int | None = Field(default=None, ge=1)
    frame_height: int | None = Field(default=None, ge=1)

    @field_validator("areas")
    @classmethod
    def _unique_names(cls, areas: list[AreaIn]) -> list[AreaIn]:
        seen = set()
        for a in areas:
            key = a.name.casefold()
            if key in seen:
                raise ValueError(f"Alan adı tekrar ediyor: {a.name}")
            seen.add(key)
        return areas


class TemplatePayload(AreasPayload):
    name: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"[\w\- ]{1,60}", v, flags=re.UNICODE):
            raise ValueError("Şablon adı yalnızca harf, rakam, boşluk, - ve _ içerebilir (en fazla 60)")
        if v.casefold().startswith("video_"):
            raise ValueError("Şablon adı 'video_' ile başlayamaz")
        return v


class ModelRegister(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    name: str | None = Field(default=None, max_length=200)


class ModelRename(BaseModel):
    id: str = Field(min_length=1, max_length=1024)
    name: str = Field(default="", max_length=200)      # boş = özel adı kaldır


class ModelId(BaseModel):
    id: str = Field(min_length=1, max_length=1024)


class ExportRequest(BaseModel):
    id: str = Field(min_length=1, max_length=1024)
    format: Literal["openvino", "onnx"] = "openvino"
    imgsz: int | None = Field(default=None, ge=320, le=1920)


class ClassesPayload(BaseModel):
    classes: list[str] = Field(default_factory=list, max_length=500)


class SettingsPatch(BaseModel):
    """Kısmi güncelleme: yalnızca gönderilen alanlar değişir."""
    count_conf: float | None = Field(default=None, ge=0.05, le=0.99)
    iou: float | None = Field(default=None, ge=0.1, le=0.95)
    tracker: Literal[tuple(TRACKERS)] | None = None          # type: ignore[valid-type]
    mode: Literal[tuple(MODE_IMGSZ)] | None = None            # type: ignore[valid-type]
    fps_limit: int | None = Field(default=None, ge=1, le=240)
    realtime_sync: bool | None = None
    brightness: int | None = Field(default=None, ge=-100, le=100)
    min_frames_in_zone: int | None = Field(default=None, ge=1, le=30)
    trace_length: int | None = Field(default=None, ge=0, le=200)
    device: str | None = Field(default=None, pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$")
    half: bool | None = None
    allowed_classes: list[str] | None = Field(default=None, max_length=500)
    # API'ye özgü
    stream_max_width: int | None = Field(default=None, ge=320, le=3840)
    jpeg_quality: int | None = Field(default=None, ge=30, le=95)


class SessionCreate(BaseModel):
    video_id: str = Field(min_length=1, max_length=64)
    mode: Literal["live", "fast"] = "live"
    # Yalnızca hızlı analizde: her N. kareyi işle. Boş (null) = otomatik (≈10 kare/sn analiz hızı)
    frame_step: int | None = Field(default=None, ge=1, le=60)


class SaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Kayıt adı boş olamaz")
        return v
