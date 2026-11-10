"""
core/exporter.py
────────────────
.pt → ONNX / OpenVINO / TensorRT dönüştürme (Ultralytics export).

  onnx      : taşınabilir format (her yerde çalışır)
  openvino  : Intel CPU / iGPU için tasarlanmıştır
  engine    : NVIDIA GPU + TensorRT kurulu olmalı

HIZ GARANTİSİ YOK: kazanç donanıma göre değişir (tek çekirdekli bir CPU'da ONNX,
.pt'den yavaş çıkabilir). Dönüştürdükten sonra analiz ekranındaki "kare/sn" değeriyle
kendi makinende karşılaştır.

Dönüştürülen dosya .pt'nin yanına yazılır; Modeller sayfasında ayrıca görünür. Not: dışa aktarılan model sabit imgsz ile
üretilir, çıkarımda aynı imgsz kullanılır.
"""

from core.detection import precision_kwargs
from core.logger import get_logger

log = get_logger("export")

FORMATS = {
    "onnx": "ONNX",
    "openvino": "OpenVINO (Intel CPU)",
    "engine": "TensorRT (NVIDIA)",
}


def export_model(pt_path: str, fmt: str = "onnx", imgsz: int = 640,
                 half: bool = False) -> str:
    """Dönüştür ve üretilen yolu döndür. Hata olursa istisna fırlatır."""
    if fmt not in FORMATS:
        raise ValueError(f"Desteklenmeyen format: {fmt}")
    from ultralytics import YOLO

    log.info("Dışa aktarılıyor: %s → %s (imgsz=%d, half=%s)", pt_path, fmt, imgsz, half)
    out = YOLO(pt_path).export(format=fmt, imgsz=imgsz, dynamic=False, **precision_kwargs(half))
    log.info("Dışa aktarma tamam: %s", out)
    return str(out)
