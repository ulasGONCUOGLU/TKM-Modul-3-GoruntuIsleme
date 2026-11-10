"""
core/areas.py
─────────────
Alanları (poligonları) JSON olarak kaydet / yükle.

Alanlar JSON dosyasına kaydedilir; uygulama kapanınca kaybolmaz.
Dosya, çizildiği videonun çözünürlüğünü de saklar: farklı çözünürlükte
bir videoya yüklenirse noktalar otomatik ölçeklenir.
"""

import json
from pathlib import Path


def save_areas(path, areas: list[dict], frame_size: tuple[int, int]) -> None:
    data = {
        "version": 1,
        "frame_width": int(frame_size[0]),
        "frame_height": int(frame_size[1]),
        "areas": [
            {"name": a["name"], "points": [[int(x), int(y)] for x, y in a["points"]]}
            for a in areas
        ],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_areas(path, target_size: tuple[int, int] | None = None) -> list[dict]:
    """[{'id','name','points'}] döndürür; target_size verilirse noktalar ölçeklenir."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sw, sh = data.get("frame_width") or 0, data.get("frame_height") or 0
    sx = sy = 1.0
    if target_size and sw and sh:
        sx, sy = target_size[0] / sw, target_size[1] / sh
    areas = []
    for i, a in enumerate(data.get("areas", []), start=1):
        pts = [(int(round(x * sx)), int(round(y * sy))) for x, y in a["points"]]
        if len(pts) >= 3:
            areas.append({"id": i, "name": str(a["name"]), "points": pts})
    return areas
