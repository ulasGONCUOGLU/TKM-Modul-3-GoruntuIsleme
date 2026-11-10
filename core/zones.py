"""
core/zones.py
─────────────
ALAN BAZLI geçiş sayımı (supervision.PolygonZone ile).

Neden çizgi değil alan?
  Çizgi sayımı "iki ardışık karede çizginin iki yanında görülme" ister.
  Video kare atlarsa (yavaş işlemci, gerçek zamanlı senkron, kayıp kareler)
  araç çizgiyi iki kare arasında geçer ve hiç sayılmaz. Alan bazlı
  yaklaşım DURUM tutar: araç en son hangi alanda onaylandıysa onu hatırlar;
  başka bir alanda göründüğü ilk anda (kaç kare geçmiş olursa olsun) geçiş
  sayılır. Alanlar arasındaki boşluktan geçerken de durum korunur.

Algoritma (her takip ID'si için):
  1. Aracın "ayak noktası" (kutunun alt-orta noktası) hangi alanda?  (supervision)
  2. Alan dışındaysa hiçbir şey değişmez; son onaylı alan korunur.
  3. Son onaylı alandan FARKLI bir alandaysa aday olur; `min_frames_in_zone`
     ardışık VIDEO karesi boyunca (atlanan kareler dahil) aynı aday alanda
     görülürse onaylanır:
       - önceki onaylı alan varsa  → GEÇİŞ (önceki → yeni) sayılır
       - yoksa                     → yalnızca ilk giriş olarak kaydedilir
  4. Sınır titremesi (kutu iki alanın kenarında gidip gelirse) debounce ile
     çift sayıma yol açmaz.
  5. TARANAN YOL: araç iki işlenen kare arasında dar bir alanı tamamen atlayıp
     geçtiyse (iki konum da alanın dışında, aralarındaki doğru alanı kesiyor)
     o alan da ziyaret edilmiş sayılır. Yalnızca aracın daha önce görüldüğü ve
     MAX_BRIDGE video karesinden uzun süre kaybolmadığı durumlarda uygulanır.

Ek olarak alan başına "benzersiz araç girişi" sayılır: takip ID'si alanlar
arasında kaybolsa bile her alanın kendi sayımı doğru kalır.
"""

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

from core.logger import get_logger

log = get_logger("zones")

UNKNOWN_CLASS = "?"


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0


def segment_entry_t(p, q, poly: np.ndarray) -> float | None:
    """
    p→q doğrusunun poligonun KENARINA ilk değdiği nokta (0..1 parametre).
    Uçlardan biri poligon içindeyse None döner (o durum normal onay yoluyla işlenir);
    yalnızca "iki ucu da dışarıda ama arası içinden geçiyor" durumu için değer verir.
    """
    if _inside(poly, *p) or _inside(poly, *q):
        return None
    r = (q[0] - p[0], q[1] - p[1])
    best = None
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        s = (b[0] - a[0], b[1] - a[1])
        denom = r[0] * s[1] - r[1] * s[0]
        if denom == 0:
            continue
        ap = (a[0] - p[0], a[1] - p[1])
        t = (ap[0] * s[1] - ap[1] * s[0]) / denom
        u = (ap[0] * r[1] - ap[1] * r[0]) / denom
        if 0 <= t <= 1 and 0 <= u <= 1 and (best is None or t < best):
            best = t
    return best


@dataclass(frozen=True)
class TransitionEvent:
    from_area: str
    to_area: str
    class_name: str
    tracker_id: int


@dataclass
class CountSnapshot:
    """Sayım durumunun değişmez bir kopyası (thread'ler arası güvenle taşınır)."""
    transitions: dict = field(default_factory=dict)            # {(a,b): n}
    transitions_by_class: dict = field(default_factory=dict)   # {(a,b): {cls: n}}
    zone_visits: dict = field(default_factory=dict)            # {alan: n}
    zone_visits_by_class: dict = field(default_factory=dict)   # {alan: {cls: n}}
    zone_current: dict = field(default_factory=dict)           # {alan: o an içindeki araç}
    version: int = 0

    @property
    def total_transitions(self) -> int:
        return sum(self.transitions.values())


class ZoneCounter:
    PURGE_EVERY = 500      # işlenen kare
    STALE_AFTER = 900      # video karesi: bu kadar süre görülmeyen ID'nin durumu silinir
    MAX_BRIDGE = 90        # video karesi: "taranan yol" kontrolü için en uzun görülmeme süresi

    def __init__(self, min_frames_in_zone: int = 2,
                 anchor: sv.Position = sv.Position.BOTTOM_CENTER):
        self._lock = threading.RLock()
        self.min_frames = max(1, int(min_frames_in_zone))
        self.anchor = anchor

        self._areas: list[dict] = []
        self._zones: list[tuple[str, sv.PolygonZone]] = []
        self._polys: list[np.ndarray] = []

        self._transitions: dict[tuple, Counter] = {}
        self._visits: dict[str, Counter] = {}
        self._reset_tracking_state_unlocked()
        self._version = 0
        self._frame_no = 0
        self._clock = 0        # video karesi sayacı (atlanan kareler dahil)

    # ── yapılandırma ───────────────────────────────────────────────
    def configure(self, min_frames_in_zone: int) -> None:
        with self._lock:
            self.min_frames = max(1, int(min_frames_in_zone))

    @property
    def areas(self) -> list[dict]:
        with self._lock:
            return [dict(a, points=list(a["points"])) for a in self._areas]

    @property
    def version(self) -> int:
        return self._version

    def set_areas(self, areas: list[dict]) -> None:
        """
        Alan listesini değiştir (ekle/sil/düzenle sonrası).
        Hâlâ var olan alan çiftlerinin sayımları korunur; takip durumu
        (hangi araç hangi alanda) alan adları değişmediyse korunur.
        """
        with self._lock:
            self._areas = [dict(a, points=list(a["points"])) for a in areas]
            self._zones = []
            self._polys = []
            for a in self._areas:
                poly = np.array(a["points"], dtype=np.int32).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                self._zones.append((a["name"], sv.PolygonZone(
                    polygon=poly, triggering_anchors=(self.anchor,))))
                self._polys.append(poly)

            names = [n for n, _ in self._zones]
            valid_pairs = {(a, b) for a in names for b in names if a != b}
            self._transitions = {k: v for k, v in self._transitions.items() if k in valid_pairs}
            self._visits = {k: v for k, v in self._visits.items() if k in names}

            self._last_zone = {t: z for t, z in self._last_zone.items() if z in names}
            kept = defaultdict(set)
            for t, zs in self._visited.items():
                kept[t] = {z for z in zs if z in names}
            self._visited = kept
            self._cand.clear()
            self._version += 1

    # ── sıfırlama ──────────────────────────────────────────────────
    def reset_counts(self) -> None:
        with self._lock:
            self._transitions.clear()
            self._visits.clear()
            self._version += 1

    def reset_tracking_state(self) -> None:
        """Takip ID'leri yeniden başlayınca (yeni video / tracker reset) çağrılmalı."""
        with self._lock:
            self._reset_tracking_state_unlocked()

    def _reset_tracking_state_unlocked(self) -> None:
        self._last_zone: dict[int, str] = {}
        self._cand: dict[int, tuple[str, int]] = {}
        self._last_seen: dict[int, int] = {}
        self._last_pos: dict[int, tuple[float, float]] = {}
        self._visited: dict[int, set] = defaultdict(set)

    # ── ana işlem ──────────────────────────────────────────────────
    def update(self, detections: sv.Detections, frame_step: int = 1) -> list[TransitionEvent]:
        """
        Bir karenin takipli tespitlerini işle; onaylanan geçişleri döndür.

        frame_step: bu işlenen kare, bir önceki işlenen kareden kaç VIDEO karesi
        sonra geldi (atlanan kareler dahil; normal oynatmada 1). Onay süresi
        (min_frames_in_zone) video karesi cinsinden sayıldığı için, kare atlansa
        bile "iki kare bekle" kuralı geçerli geçişi kaçırmaz.
        """
        step = max(1, int(frame_step))
        events: list[TransitionEvent] = []
        with self._lock:
            self._frame_no += 1
            self._clock += step
            n = len(detections)
            if not self._zones:
                return events

            tids = detections.tracker_id
            if n == 0 or tids is None:
                for _, z in self._zones:
                    z.current_count = 0
                return events

            # Her tespit için ilk eşleşen alanın indeksi (-1: hiçbiri)
            zone_idx = np.full(n, -1, dtype=int)
            for i, (_, zone) in enumerate(self._zones):
                inside = zone.trigger(detections)
                zone_idx[(zone_idx == -1) & inside] = i

            cls_names = detections.data.get("class_name") if detections.data else None
            pts = detections.get_anchors_coordinates(self.anchor)

            for k in range(n):
                tid = int(tids[k])
                cur = (float(pts[k][0]), float(pts[k][1]))
                prev_pos = self._last_pos.get(tid)
                prev_seen = self._last_seen.get(tid)
                self._last_seen[tid] = self._clock
                self._last_pos[tid] = cur
                zi = int(zone_idx[k])
                cls = str(cls_names[k]) if cls_names is not None else UNKNOWN_CLASS

                # 1) TARANAN YOL: iki işlenen kare arasında, hiçbirinde içinde görülmediği
                #    bir alanın içinden geçtiyse o alan da ziyaret edilmiş sayılır.
                if (prev_pos is not None and prev_seen is not None
                        and self._clock - prev_seen <= self.MAX_BRIDGE):
                    crossed = []
                    for j, poly in enumerate(self._polys):
                        t = segment_entry_t(prev_pos, cur, poly)
                        if t is not None:
                            crossed.append((t, j))
                    for _, j in sorted(crossed):
                        self._confirm(tid, self._zones[j][0], cls, events)

                # 2) Normal yol: şu an bir alanın içinde (debounce'lu onay)
                if zi < 0:
                    self._cand.pop(tid, None)  # alan dışı: son onaylı alan korunur
                    continue

                zname = self._zones[zi][0]
                if zname == self._last_zone.get(tid):
                    self._cand.pop(tid, None)
                    continue

                cand_zone, cnt = self._cand.get(tid, (None, 0))
                cnt = cnt + step if cand_zone == zname else step
                if cnt < self.min_frames:
                    self._cand[tid] = (zname, cnt)
                    continue
                self._cand.pop(tid, None)
                self._confirm(tid, zname, cls, events)

            if events:
                self._version += 1
            if self._frame_no % self.PURGE_EVERY == 0:
                self._purge_stale()
        return events

    def _confirm(self, tid: int, zname: str, cls: str, events: list) -> None:
        """Bir aracın bu alana girişini kesinleştir; önceki onaylı alandan geçiş varsa say."""
        prev = self._last_zone.get(tid)
        if zname == prev:
            return
        self._last_zone[tid] = zname
        visited = self._visited.setdefault(tid, set())
        if zname not in visited:
            visited.add(zname)
            self._visits.setdefault(zname, Counter())[cls] += 1
        if prev is not None:
            self._transitions.setdefault((prev, zname), Counter())[cls] += 1
            events.append(TransitionEvent(prev, zname, cls, tid))

    def _purge_stale(self) -> None:
        limit = self._clock - self.STALE_AFTER
        stale = [t for t, f in self._last_seen.items() if f < limit]
        for t in stale:
            self._last_seen.pop(t, None)
            self._last_pos.pop(t, None)
            self._last_zone.pop(t, None)
            self._cand.pop(t, None)
            self._visited.pop(t, None)

    # ── okuma ──────────────────────────────────────────────────────
    def snapshot(self) -> CountSnapshot:
        with self._lock:
            names = [n for n, _ in self._zones]
            trans, by_cls = {}, {}
            for a in names:
                for b in names:
                    if a == b:
                        continue
                    c = self._transitions.get((a, b), Counter())
                    trans[(a, b)] = sum(c.values())
                    by_cls[(a, b)] = dict(c)
            visits = {n: sum(self._visits.get(n, Counter()).values()) for n in names}
            visits_cls = {n: dict(self._visits.get(n, Counter())) for n in names}
            current = {n: int(getattr(z, "current_count", 0)) for n, z in self._zones}
            return CountSnapshot(trans, by_cls, visits, visits_cls, current, self._version)

    def zone_current(self) -> dict:
        with self._lock:
            return {n: int(getattr(z, "current_count", 0)) for n, z in self._zones}
