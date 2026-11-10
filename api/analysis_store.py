"""
api/analysis_store.py
─────────────────────
Biten analizlerin KAYDEDİLDİĞİ yer: Supabase, tablo `gi_analyses`.

Bu API yalnızca YAZAR. Kayıtları listeleme / ayrıntı / silme işini Nuxt sunucusu doğrudan
Supabase'ten yapar (bkz. Nuxt projesi, server/api/goruntuisleme/analizler). Tablo şeması bu projede:
sql/kurulum.sql.

Kayıt (record) alanları:
  name, video_name, model, mode ('live'|'fast'), frame_step, frame_count, processed_frames,
  duration_sec (işlenen video süresi), processing_sec (analizin sürdüğü süre), source_fps,
  width, height, total_transitions, totals {sınıf: adet}, transitions [{from,to,count,by_class}],
  zone_visits [{area,count,by_class}], areas [{name,points}], settings {…}
"""

from api import config
from core.logger import get_logger
from core.supabase_client import StorageError, get_client, host

__all__ = ["StorageError", "save", "describe"]

log = get_logger("analyses")

_JSON_FIELDS = ("totals", "transitions", "zone_visits", "areas", "settings")


def _clean(record: dict) -> dict:
    rec = {k: record.get(k) for k in (
        "name", "video_name", "model", "mode", "frame_step", "frame_count", "processed_frames",
        "duration_sec", "processing_sec", "source_fps", "width", "height", "total_transitions",
        *_JSON_FIELDS)}
    for k in _JSON_FIELDS:
        if rec[k] is None:
            rec[k] = {} if k in ("totals", "settings") else []
    return rec


def save(record: dict) -> int:
    """Kaydı Supabase'e yazar, yeni kaydın id'sini döndürür."""
    r = get_client().request("POST", config.SUPABASE_TABLE, json_body=_clean(record),
                             prefer="return=representation")
    rows = r.json()
    if not rows:
        raise StorageError("Supabase kayıt oluşturdu ama kimlik döndürmedi.")
    return int(rows[0]["id"])


def describe() -> dict:
    return {"kind": "supabase", "label": "Supabase", "table": config.SUPABASE_TABLE, "host": host()}
