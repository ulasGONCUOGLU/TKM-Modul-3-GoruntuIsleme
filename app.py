"""
Görüntü İşleme API'sini başlatır:

    python app.py

Adres, port ve Supabase ayarları proje klasöründeki `.env` dosyasından okunur (bkz. .env.example).
Belgeler (tarayıcı): http://127.0.0.1:5001/docs
"""

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 veya üstü gerekli.")

if sys.prefix == sys.base_prefix:
    print("UYARI: Sanal ortam (venv) etkin değil. Önerilen kurulum için README.md'ye bakın.\n")

try:
    import fastapi  # noqa: F401
    import uvicorn
except ImportError:
    sys.exit("Gerekli paketler kurulu değil. Sanal ortam etkinken çalıştırın:\n"
             "    pip install -r requirements.txt")

from api import config  # noqa: E402  (.env burada okunur)
from core.database import init_db  # noqa: E402
from core.detection import YOLO_AVAILABLE  # noqa: E402
from core.supabase_client import StorageError, is_configured  # noqa: E402


def main() -> None:
    if not is_configured():
        sys.exit("HATA: Supabase ayarlı değil.\n"
                 "  .env dosyasına SUPABASE_URL ve SUPABASE_SERVICE_ROLE_KEY yazın (bkz. .env.example).\n"
                 "  Veritabanı tablolarını sql/kurulum.sql ile oluşturun (README.md → Kurulum).")
    try:
        init_db()                       # bağlantıyı ve tabloyu açılışta sına
    except StorageError as e:
        sys.exit(f"HATA: Supabase'e erişilemedi.\n  {e}")
    print("─" * 60)
    print(f" Görüntü İşleme API'si   http://{config.HOST}:{config.PORT}")
    print(f" Veritabanı              Supabase ({config.SUPABASE_URL})")
    print(f" YOLO (ultralytics)      {'hazır' if YOLO_AVAILABLE else 'KURULU DEĞİL → pip install ultralytics'}")
    print(f" Belgeler                http://{config.HOST}:{config.PORT}/docs")
    print("─" * 60)
    uvicorn.run("api.main:app", host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
