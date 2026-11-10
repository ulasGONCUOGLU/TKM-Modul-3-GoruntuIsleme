"""
core/annotate.py
────────────────
Kare üzerine çizimler (supervision annotator'ları + alan çizimi).

  * Kutu + etiket        : sv.BoxAnnotator / sv.LabelAnnotator (sınıfa göre renk)
  * Sayım noktası        : kutunun alt-orta noktası (alanın tetiklendiği nokta)
  * İz çizgisi           : takip ID'si başına son N konum
  * Alanlar              : yarı saydam dolgu + kenar + isim + o an içindeki araç sayısı

Alan isimleri Türkçe karakter içerebildiği için core.text.put_text ile yazılır.
"""

from collections import deque

import cv2
import numpy as np
import supervision as sv

from core.config import CLASS_COLORS_RGB, FALLBACK_COLORS_RGB
from core.text import put_text

ZONE_COLORS = [
    sv.Color(0, 200, 0), sv.Color(255, 140, 0), sv.Color(30, 144, 255),
    sv.Color(220, 20, 60), sv.Color(148, 0, 211), sv.Color(0, 206, 209),
]


class FrameAnnotator:
    HISTORY_PURGE_EVERY = 120

    def __init__(self, trace_length: int = 20):
        self.trace_length = trace_length
        self._palette = sv.ColorPalette([sv.Color(*c) for c in FALLBACK_COLORS_RGB])
        self._box = sv.BoxAnnotator(color=self._palette, thickness=2,
                                    color_lookup=sv.ColorLookup.CLASS)
        self._label = sv.LabelAnnotator(color=self._palette, text_scale=0.5,
                                        text_thickness=1, text_padding=4,
                                        color_lookup=sv.ColorLookup.CLASS)
        self._dot = sv.DotAnnotator(color=self._palette, radius=4,
                                    position=sv.Position.BOTTOM_CENTER,
                                    color_lookup=sv.ColorLookup.CLASS)
        self._history: dict[int, deque] = {}
        self._seen: dict[int, int] = {}
        self._frame_no = 0

    def set_classes(self, names: dict[int, str]) -> None:
        """Model sınıf adlarına göre renk paletini kur (sınıf id → renk)."""
        if not names:
            return
        colors = []
        for i in range(max(names) + 1):
            rgb = CLASS_COLORS_RGB.get(names.get(i, ""))
            if rgb is None:
                rgb = FALLBACK_COLORS_RGB[i % len(FALLBACK_COLORS_RGB)]
            colors.append(sv.Color(*rgb))
        self._palette = sv.ColorPalette(colors)
        self._box.color = self._palette
        self._label.color = self._palette
        self._dot.color = self._palette

    def reset_history(self) -> None:
        self._history.clear()
        self._seen.clear()

    # ── çizim ──────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, detections: sv.Detections | None,
             areas: list[dict], zone_current: dict | None = None,
             update_history: bool = True) -> np.ndarray:
        """frame yerinde değiştirilir ve döndürülür."""
        self._draw_zones(frame, areas, zone_current or {})

        if detections is not None and len(detections) > 0:
            if update_history:
                self._update_history(detections)
            self._draw_traces(frame, detections)
            frame = self._box.annotate(frame, detections)
            labels = self._labels(detections)
            frame = self._label.annotate(frame, detections, labels=labels)
            frame = self._dot.annotate(frame, detections)
        elif update_history:
            self._frame_no += 1
        return frame

    @staticmethod
    def _labels(det: sv.Detections) -> list[str]:
        names = det.data.get("class_name") if det.data else None
        tids = det.tracker_id
        out = []
        for i in range(len(det)):
            n = str(names[i]) if names is not None else str(int(det.class_id[i]))
            out.append(f"{n} #{int(tids[i])}" if tids is not None else n)
        return out

    def _update_history(self, det: sv.Detections) -> None:
        self._frame_no += 1
        if det.tracker_id is None or self.trace_length <= 0:
            return
        pts = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for tid, (x, y) in zip(det.tracker_id, pts):
            tid = int(tid)
            h = self._history.get(tid)
            if h is None:
                h = self._history[tid] = deque(maxlen=self.trace_length)
            h.append((int(x), int(y)))
            self._seen[tid] = self._frame_no
        if self._frame_no % self.HISTORY_PURGE_EVERY == 0:
            limit = self._frame_no - 90
            for tid in [t for t, f in self._seen.items() if f < limit]:
                self._seen.pop(tid, None)
                self._history.pop(tid, None)

    def _draw_traces(self, frame: np.ndarray, det: sv.Detections) -> None:
        if det.tracker_id is None or self.trace_length <= 0:
            return
        for tid, cid in zip(det.tracker_id, det.class_id):
            h = self._history.get(int(tid))
            if h is None or len(h) < 2:
                continue
            c = self._palette.by_idx(int(cid)).as_bgr()
            cv2.polylines(frame, [np.array(h, dtype=np.int32)], False, c, 2, cv2.LINE_AA)

    def _draw_zones(self, frame: np.ndarray, areas: list[dict], zone_current: dict) -> None:
        for i, area in enumerate(areas):
            pts = np.array(area["points"], dtype=np.int32)
            if len(pts) < 3:
                continue
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            sv.draw_filled_polygon(frame, pts, color=color, opacity=0.12)
            sv.draw_polygon(frame, pts, color=color, thickness=2)
            n = zone_current.get(area["name"])
            label = f"{area['name']}" + (f"  ({n})" if n is not None else "")
            x = int(pts[:, 0].min())
            y = max(int(pts[:, 1].min()) - 26, 0)
            put_text(frame, label, (x, y), size=20, color_bgr=(255, 255, 255),
                     bg_bgr=color.as_bgr())
