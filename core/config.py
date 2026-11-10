"""
core/config.py
──────────────
TÜM EŞİKLER VE AYARLAR TEK YERDE.

Güven eşikleri ve diğer ayarlar `Settings` içinde, adı ve anlamı belli:

  count_conf          Bir aracın GÖSTERİLMESİ ve SAYILMASI için min. güven
  detect_conf         Dedektörün izleyiciye verdiği min. güven  (türetilir)
  iou                 NMS IoU eşiği
  min_frames_in_zone  Yeni alana geçişin onaylanması için gereken ardışık kare
  tracker             'bytetrack' | 'botsort'
  mode                'fast' | 'balanced' | 'quality'  → imgsz
  fps_limit           İşleme hızı üst sınırı
  realtime_sync       Yavaş kalınca kare atla (gerçek zamanlı akış için)
  brightness          Parlaklık düzeltmesi (-50..50)

Değerler web arayüzündeki Ayarlar sayfasından değiştirilir ve yerel veritabanında
saklanır; analiz her başlatıldığında `Settings.load()` ile okunur.
"""

from dataclasses import dataclass, fields

from core.database import get_setting, set_settings

# Sayılacak sınıflar: Türkçe etiketli kendi modelin + COCO İngilizce adları.
# (Modelinde bunlardan hiçbiri yoksa tüm sınıflar sayılır.)
DEFAULT_ALLOWED_CLASSES = ("Araba", "Kamyon", "Otobus", "car", "truck", "bus")

# Sınıf renkleri (RGB). Listede olmayan sınıflar için FALLBACK_COLORS kullanılır.
CLASS_COLORS_RGB = {
    "Araba": (0, 220, 0),
    "car": (0, 220, 0),
    "Kamyon": (255, 165, 0),
    "truck": (255, 165, 0),
    "Otobus": (30, 144, 255),
    "bus": (30, 144, 255),
}
FALLBACK_COLORS_RGB = [
    (255, 64, 64), (163, 81, 251), (0, 214, 193), (255, 214, 10),
    (255, 118, 51), (76, 251, 18), (255, 57, 201),
]

MODE_IMGSZ = {"fast": 480, "balanced": 640, "quality": 960}
TRACKERS = ("bytetrack", "botsort")

_PREFIX = "cfg."


@dataclass
class Settings:
    count_conf: float = 0.50
    iou: float = 0.70
    tracker: str = "bytetrack"
    mode: str = "balanced"
    fps_limit: int = 60
    realtime_sync: bool = False
    brightness: int = 0
    min_frames_in_zone: int = 2
    trace_length: int = 20
    device: str = "auto"       # 'auto' | 'cpu' | 'cuda:0' | 'mps'
    half: bool = True          # yalnızca CUDA'da etkili (FP16)
    allowed_classes: tuple = DEFAULT_ALLOWED_CLASSES

    # ── türetilmiş değerler ────────────────────────────────────────
    @property
    def detect_conf(self) -> float:
        """İzleyiciye verilen düşük eşik: sayım eşiğinden hiçbir zaman yüksek olmaz."""
        return min(0.30, self.count_conf)

    @property
    def imgsz(self) -> int:
        return MODE_IMGSZ.get(self.mode, 640)

    # ── DB ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        for f in fields(cls):
            if f.name == "allowed_classes":
                raw = get_setting(_PREFIX + f.name)
                if raw:
                    s.allowed_classes = tuple(x for x in raw.split(",") if x)
                continue
            raw = get_setting(_PREFIX + f.name)
            if raw is None:
                continue
            try:
                cur = getattr(s, f.name)
                if isinstance(cur, bool):
                    val = raw.lower() in ("1", "true", "yes")
                elif isinstance(cur, int):
                    val = int(float(raw))
                elif isinstance(cur, float):
                    val = float(raw)
                else:
                    val = raw
                setattr(s, f.name, val)
            except ValueError:
                pass  # bozuk değer → varsayılan kalır
        s.validate()
        return s

    def save(self) -> None:
        self.validate()
        out = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                val = ",".join(val)
            out[_PREFIX + f.name] = str(val)
        set_settings(out)                       # tek istekte yaz

    def validate(self) -> None:
        self.count_conf = min(max(self.count_conf, 0.05), 0.99)
        self.iou = min(max(self.iou, 0.1), 0.95)
        self.fps_limit = min(max(int(self.fps_limit), 1), 240)
        self.brightness = min(max(int(self.brightness), -100), 100)
        self.min_frames_in_zone = min(max(int(self.min_frames_in_zone), 1), 30)
        self.trace_length = min(max(int(self.trace_length), 0), 200)
        if self.tracker not in TRACKERS:
            self.tracker = "bytetrack"
        if self.mode not in MODE_IMGSZ:
            self.mode = "balanced"
