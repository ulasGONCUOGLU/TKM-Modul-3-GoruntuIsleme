"""
core/detection.py
─────────────────
YOLO tespiti + çok nesne takibi → supervision.Detections.

  * Cihaz (cuda/mps/cpu) ve FP16 (half) artık açıkça seçilir.
  * Tracker: 'bytetrack' veya 'botsort' (Ultralytics), ayarlardan seçilir.
  * Model: .pt, .onnx, .engine (TensorRT) veya OpenVINO klasörü.
  * Çıktı supervision.Detections: kutu, güven, sınıf, tracker_id ve
    data['class_name'] alanı hazırdır.
  * Güven eşiği tek yerde (Settings.count_conf); ikinci bir eleme yoktur.
"""

import os
from pathlib import Path

import numpy as np
import supervision as sv

from core.config import Settings
from core.logger import get_logger

log = get_logger("detection")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:  # ultralytics yoksa uygulama yine açılır, tespit devre dışı kalır
    YOLO = None
    YOLO_AVAILABLE = False
    log.warning("ultralytics bulunamadı; tespit özellikleri devre dışı.")


def resolve_device(preference: str = "auto") -> str:
    """'auto' → cuda:0 > mps > cpu. Torch yoksa 'cpu'."""
    if preference and preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def precision_kwargs(half: bool) -> dict:
    """
    FP16 isteği için sürümden bağımsız argüman.
    Yeni Ultralytics `half` argümanını kaldırıyor (yerine `quantize=16`);
    eski sürümler ise `quantize` bilmez. FP32 her sürümde varsayılandır,
    bu yüzden half=False için hiçbir şey geçilmez (uyarı da çıkmaz).
    """
    if not half:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


def _is_torch_weights(path: str) -> bool:
    return str(path).lower().endswith(".pt")


class DetectionEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._model = None
        self._model_path: str | None = None
        self._class_ids: list[int] | None = None
        self.names: dict[int, str] = {}
        self.device = "cpu"
        self._half = False
        self._tracker_in_use: str | None = None

    # ── model ──────────────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str | None:
        return self._model_path

    def configure(self, settings: Settings) -> None:
        """Video başlarken çağrılır; oynatma sırasında ayar değişmez."""
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._half = bool(settings.half) and self.device.startswith("cuda")
        if self._model is not None:
            self._compute_class_filter()

    def load(self, model_path: str) -> None:
        """Modeli yükle. Hata olursa istisna fırlatır (çağıran gösterir)."""
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics kurulu değil: pip install ultralytics")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")

        self.device = resolve_device(self.settings.device)
        self._half = bool(self.settings.half) and self.device.startswith("cuda")

        model = YOLO(str(model_path), task="detect")
        # .pt dosyasını bir kez cihaza taşı (ilk kare gecikmesini azaltır)
        if _is_torch_weights(model_path):
            try:
                model.to(self.device)
            except Exception:
                log.exception("Model %s cihazına taşınamadı; CPU'da devam.", self.device)
                self.device, self._half = "cpu", False

        self._model = model
        self._model_path = str(model_path)
        self.names = dict(model.names)
        self._compute_class_filter()
        log.info("Model yüklendi: %s | cihaz=%s half=%s | sınıflar=%s",
                 Path(model_path).name, self.device, self._half, list(self.names.values()))

    def _compute_class_filter(self) -> None:
        """Sayılacak sınıf id'leri. Hiçbiri eşleşmezse tüm sınıflar sayılır."""
        allowed = set(self.settings.allowed_classes)
        ids = [i for i, n in self.names.items() if n in allowed]
        if ids:
            self._class_ids = ids
        else:
            self._class_ids = None
            log.warning("Modelde izin verilen sınıf adı yok (%s); tüm sınıflar sayılacak.",
                        sorted(allowed))

    def unload(self) -> None:
        self._model = None
        self._model_path = None
        self.names = {}

    # ── takip durumu ───────────────────────────────────────────────
    def reset_tracker(self) -> None:
        """Yeni video / yeni analiz: tracker ID'leri 1'den başlasın."""
        if self._model is None:
            return
        # Tracker türü değiştiyse (bytetrack ↔ botsort) reset yetmez; yeniden kurulmalı.
        if self._tracker_in_use not in (None, self.settings.tracker):
            self._model.predictor = None
            self._tracker_in_use = None
            return
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            try:
                for t in trackers:
                    t.reset()
                return
            except Exception:
                log.exception("tracker.reset() başarısız; predictor sıfırlanıyor.")
        # Yedek: predictor'ı at; sonraki track() çağrısı yeniden kurar.
        self._model.predictor = None

    # ── çıkarım ────────────────────────────────────────────────────
    def infer(self, frame: np.ndarray) -> sv.Detections:
        """Tek kare: tespit + takip. Takip ID'si olmayan kutular atılır."""
        if self._model is None:
            return sv.Detections.empty()

        s = self.settings
        kwargs = dict(
            conf=s.detect_conf,
            iou=s.iou,
            tracker=f"{s.tracker}.yaml",
            persist=True,
            verbose=False,
        )
        if self._class_ids is not None:
            kwargs["classes"] = self._class_ids
        if _is_torch_weights(self._model_path):
            kwargs.update(device=self.device, imgsz=s.imgsz, **precision_kwargs(self._half))

        result = self._model.track(frame, **kwargs)[0]
        self._tracker_in_use = s.tracker
        det = sv.Detections.from_ultralytics(result)

        if det.tracker_id is None or len(det) == 0:
            return sv.Detections.empty()

        det = det[det.confidence >= s.count_conf]  # TEK güven eşiği
        return det
