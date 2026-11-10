"""
core/database.py
────────────────
Ayarlar (anahtar/değer) — Supabase `gi_settings` tablosunda saklanır.

Açılışta tablo bir kerede okunup belleğe alınır; okumalar bellekten yapılır (ağ yok), yazmalar
Supabase'e gider ve başarılı olunca belleğe işlenir. Yazma başarısızsa StorageError fırlatılır
(API bunu 502 olarak döndürür), ayar eski değerinde kalır.
"""

import threading

from core.logger import get_logger
from core.supabase_client import get_client

TABLE = "gi_settings"
log = get_logger("db")

_cache: dict[str, str] = {}
_lock = threading.Lock()
_loaded = False


def init_db() -> None:
    """Ayarları Supabase'ten yükle (açılışta). Ulaşılamazsa StorageError."""
    global _loaded
    rows = get_client().request("GET", TABLE, params={"select": "key,value"}).json()
    with _lock:
        _cache.clear()
        _cache.update({r["key"]: r["value"] for r in rows if r.get("value") is not None})
        _loaded = True
    log.info("Ayarlar Supabase'ten yüklendi (%d kayıt).", len(rows))


def get_setting(key: str, default: str | None = None) -> str | None:
    if not _loaded:
        init_db()
    with _lock:
        return _cache.get(key, default)


def set_settings(values: dict[str, str]) -> None:
    """Birden çok ayarı TEK istekte yaz (upsert)."""
    if not values:
        return
    rows = [{"key": k, "value": str(v)} for k, v in values.items()]
    get_client().request("POST", TABLE, params={"on_conflict": "key"}, json_body=rows,
                         prefer="resolution=merge-duplicates,return=minimal")
    with _lock:
        _cache.update({k: str(v) for k, v in values.items()})


def set_setting(key: str, value: str) -> None:
    set_settings({key: value})
