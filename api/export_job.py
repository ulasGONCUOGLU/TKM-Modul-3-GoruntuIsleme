"""
api/export_job.py
─────────────────
Modeli .pt → OpenVINO / ONNX dönüştürme işi (arka planda, tek seferde bir tane).

OpenVINO, özellikle Intel işlemcilerde CPU'da analizi belirgin hızlandırabilir. Kazanç donanıma
göre değişir; dönüştürdükten sonra analiz ekranındaki "kare/sn" ile karşılaştırın.
Dönüştürülen model .pt'nin yanına yazılır ve Modeller listesinde ayrıca görünür.
"""

import threading
import time

from api import models_store
from core.exporter import FORMATS, export_model
from core.logger import get_logger

log = get_logger("export-job")

_lock = threading.Lock()
_job: dict = {"state": "idle"}      # idle | running | done | error


def status() -> dict:
    with _lock:
        return dict(_job)


def is_running() -> bool:
    return status()["state"] == "running"


def start(model_id: str, src_path, fmt: str, imgsz: int) -> None:
    """Başlatır; zaten bir iş çalışıyorsa RuntimeError."""
    if fmt not in FORMATS or fmt == "engine":
        raise ValueError("Desteklenen formatlar: openvino, onnx")
    with _lock:
        if _job.get("state") == "running":
            raise RuntimeError("Başka bir dönüştürme sürüyor.")
        _job.clear()
        _job.update(state="running", model=model_id, format=fmt, imgsz=imgsz,
                    started_at=time.time(), message="Dönüştürülüyor… (birkaç dakika sürebilir)")

    def work():
        try:
            out = export_model(str(src_path), fmt, imgsz, False)
            if models_store.is_abs(model_id):           # elle kaydedilmiş modelin çıktısı da kaydedilsin
                models_store.add_extra(out)
            with _lock:
                _job.update(state="done", output=str(out), finished_at=time.time(),
                            message="Dönüştürme tamam. Modeller listesinden aktif yapabilirsiniz.")
        except Exception as e:                           # noqa: BLE001
            log.exception("Dışa aktarma başarısız")
            with _lock:
                _job.update(state="error", finished_at=time.time(),
                            message=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True, name="model-export").start()
