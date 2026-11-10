"""
core/paths.py
─────────────
Tüm yollar burada. Yollar bu dosyanın konumuna göre hesaplanır; uygulama
hangi klasörden başlatılırsa başlatılsın model ve veritabanı bulunur.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# GORUNTU_DATA_DIR ortam değişkeni ile veri klasörü başka yere taşınabilir.
DOSYALAR_DIR = Path(os.environ.get("GORUNTU_DATA_DIR") or (BASE_DIR / "dosyalar"))
MODEL_TRAIN_DIR = DOSYALAR_DIR / "Model"    # yüklenen modeller (Model/<ad>/weights/best.pt)
AREAS_DIR = DOSYALAR_DIR / "alanlar"        # alan JSON dosyaları (video ve şablon)
LOG_DIR = DOSYALAR_DIR / "logs"


def ensure_dirs() -> None:
    for d in (DOSYALAR_DIR, MODEL_TRAIN_DIR, AREAS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
