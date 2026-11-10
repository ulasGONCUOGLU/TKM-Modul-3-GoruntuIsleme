"""
api/config.py
─────────────
API'nin ortam değişkenleriyle ayarlanan değerleri.

  GORUNTU_HOST                 dinlenecek adres (varsayılan 127.0.0.1: yalnızca bu bilgisayar)
  GORUNTU_PORT                 port (varsayılan 5001)
  GORUNTU_CORS_ORIGINS         virgülle ayrılmış izinli tarayıcı adresleri
                               (varsayılan: http://localhost:3000,http://127.0.0.1:3000)
  GORUNTU_MAX_UPLOAD_MB        tek video yüklemesi üst sınırı (varsayılan 16384 = 16 GB)
  GORUNTU_MAX_MODEL_UPLOAD_MB  model yüklemesi üst sınırı (varsayılan 2048)
  GORUNTU_ALLOW_PATH_REGISTER  "1": modeli elle girilen disk yolundan kaydetmeye izin ver
                               (yerelde açık; sunucuya geçerken kimlik doğrulama gelene kadar "0" yapın)
  GORUNTU_DATA_DIR             (core/paths.py) veri klasörünü taşır
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
                               ZORUNLU. Ayarlar (gi_settings) ve analiz kayıtları (gi_analyses) burada tutulur.

Değerler proje klasöründeki `.env` dosyasından da okunur (bkz. .env.example).
"""

import os

from core.env import load_env

load_env()                                  # core.paths'ten ÖNCE: GORUNTU_DATA_DIR .env'de olabilir

from core.paths import DOSYALAR_DIR  # noqa: E402

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_TABLE = "gi_analyses"            # tablo şeması: sql/kurulum.sql

UPLOAD_DIR = DOSYALAR_DIR / "yuklenen"

HOST = os.environ.get("GORUNTU_HOST", "127.0.0.1")
PORT = int(os.environ.get("GORUNTU_PORT", "5001"))


def _origins() -> list[str]:
    raw = os.environ.get(
        "GORUNTU_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


CORS_ORIGINS = _origins()


def _mb(name: str, default: float) -> int:
    try:
        return int(float(os.environ.get(name, default)) * 1024 * 1024)
    except ValueError:
        return int(default * 1024 * 1024)


MAX_UPLOAD_BYTES = _mb("GORUNTU_MAX_UPLOAD_MB", 16384)
MAX_MODEL_UPLOAD_BYTES = _mb("GORUNTU_MAX_MODEL_UPLOAD_MB", 2048)

ALLOW_PATH_REGISTER = os.environ.get("GORUNTU_ALLOW_PATH_REGISTER", "1").strip().lower() \
    in ("1", "true", "yes", "evet")

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".wmv", ".webm"}
MODEL_EXTS = {".pt", ".onnx", ".engine"}

API_VERSION = "1.1.0"
