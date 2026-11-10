"""
core/player.py
──────────────
Video okuma + işleme thread'i.

İki çalışma biçimi vardır:

  izleyerek (headless=False)
      Video kendi gerçek FPS'ine göre oynar; her kare için tespit + takip + sayım yapılır,
      kutular/alanlar çizilir, ölçeklenir ve kuyruğa konur (web arayüzü MJPEG olarak yayınlar).
      `realtime_sync` açıksa ve işlem yetişemiyorsa kare atlanır; alan bazlı sayım kare
      atlamalarına dayanıklıdır.

  hızlı analiz (headless=True)
      Çizim, ölçekleme ve bekleme YOK. Yalnızca tespit + takip + sayım yapılır.
      `frame_step=N` ile her N. kare işlenir.
      Kare OKUMA ayrı bir thread'de yapılır ve çıkarımla paralel ilerler (okuma/çözme
      süresi çıkarım süresinin içinde gizlenir). Atlanan kareler de bu thread'de okunur.

İşçi thread'i web katmanıyla yalnızca `queue` üzerinden konuşur.
"""

import queue
import threading
import time
from collections import namedtuple
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from core.annotate import FrameAnnotator
from core.config import Settings
from core.detection import DetectionEngine
from core.logger import get_logger
from core.zones import CountSnapshot, ZoneCounter

log = get_logger("player")

PREFETCH_DEPTH = 4          # hızlı analizde önceden okunan kare sayısı


@dataclass
class Packet:
    kind: str                                  # 'frame' | 'eof' | 'error'
    display_rgb: np.ndarray | None = None      # gösterilecek kare (RGB, ölçekli); hızlı analizde None
    snapshot: CountSnapshot | None = None      # sayımlar değiştiyse
    frame_idx: int = 0
    fps: float = 0.0
    message: str = ""


# frame: işlenecek kare (EOF'ta None) · pos: bu kareye ve ardından atlananlara kadar okunan
# toplam video karesi · step: önceki işlenen kareden bu yana geçen video karesi · eof: bitti
_Item = namedtuple("_Item", "frame pos step eof")


class VideoPlayer:
    def __init__(self, engine: DetectionEngine, counter: ZoneCounter, annotator: FrameAnnotator):
        self.engine = engine
        self.counter = counter
        self.annotator = annotator

        self.queue: "queue.Queue[Packet]" = queue.Queue(maxsize=3)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._display_size = (0, 0)
        self._last_version = -1

        # Hızlı analizde duraklatınca önceden okunmuş (ama işlenmemiş) kareler burada saklanır
        # ve devam edince ilk onlar işlenir; böylece duraklatma kare kaybettirmez.
        self._leftover: list[_Item] = []
        self._rd_pos = 0          # okuyucunun okuduğu toplam kare
        self._rd_step = 1         # okuyucunun bir sonraki kare için step değeri

        self.frame_width = 0
        self.frame_height = 0
        self.source_fps = 30.0
        self.frame_count = 0
        self.frame_idx = 0
        self.last_det: sv.Detections | None = None

    # ── video ──────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def open(self, path: str) -> bool:
        self.stop()
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self._cap = cap
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_idx = 0
        self._leftover, self._rd_pos, self._rd_step = [], 0, 1
        self.last_det = None
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._leftover = []
        self.last_det = None

    def set_display_size(self, w: int, h: int) -> None:
        self._display_size = (int(w), int(h))

    # ── kontrol ────────────────────────────────────────────────────
    def start(self, settings: Settings, headless: bool = False, frame_step: int = 1) -> bool:
        """
        headless=True : "hızlı analiz" (bkz. modül açıklaması). Kuyruğa `display_rgb=None`
                        olan ilerleme paketleri konur.
        frame_step=N  : yalnızca headless'ta; her N. kareyi işler (arada N-1 kare atlanır).
                        Sayım, atlanan kareleri bilir (ZoneCounter.update frame_step).
        """
        if not self.is_open:
            return False
        if self._thread is not None and self._thread.is_alive():
            log.warning("Önceki işleme thread'i hâlâ çalışıyor; başlatılamadı.")
            return False
        self._stop.clear()
        self._last_version = -1
        self._thread = threading.Thread(
            target=self._run, args=(settings, bool(headless), max(1, int(frame_step))),
            daemon=True, name="video-worker")
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Thread'i durdur ve bitmesini bekle (en fazla `timeout` sn)."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)
            if t.is_alive():
                log.warning("İşleme thread'i %.1f sn içinde durmadı.", timeout)
                return
        self._thread = None

    def drain(self) -> None:
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

    # ── işçi thread ────────────────────────────────────────────────
    def _put(self, pkt: Packet) -> None:
        """Kuyruk doluysa en eski paketi at (tüketici yetişemiyorsa gecikme birikmesin)."""
        while True:
            try:
                self.queue.put_nowait(pkt)
                return
            except queue.Full:
                try:
                    old = self.queue.get_nowait()
                    # Kontrol paketini (eof/error) ve sayım güncellemesini asla kaybetme
                    if old.kind != "frame":
                        self.queue.put_nowait(old)
                        time.sleep(0.005)
                    elif old.snapshot is not None and pkt.snapshot is None:
                        pkt.snapshot = old.snapshot
                except queue.Empty:
                    pass

    def _reader(self, cap, frame_step: int, out: "queue.Queue[_Item]",
                stop: threading.Event, carry: list) -> None:
        """Hızlı analiz okuyucusu: kareleri oku, N-1 tanesini atla, kuyruğa koy."""
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    item = _Item(None, self._rd_pos, self._rd_step, True)
                else:
                    self._rd_pos += 1
                    pos_step = self._rd_step
                    self._rd_step = 1
                    for _ in range(frame_step - 1):
                        if not cap.grab():
                            break
                        self._rd_pos += 1
                        self._rd_step += 1
                    item = _Item(frame, self._rd_pos, pos_step, False)
                while True:
                    if stop.is_set():
                        carry.append(item)          # kuyruğa konamadı; devam edince işlenecek
                        return
                    try:
                        out.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if item.eof:
                    return
        except Exception:                           # noqa: BLE001
            log.exception("Kare okuyucu thread'inde hata")
            try:
                out.put(_Item(None, self._rd_pos, self._rd_step, True), timeout=1)
            except queue.Full:
                pass

    def _run_headless(self, settings: Settings, frame_step: int) -> None:
        cap = self._cap
        out: "queue.Queue[_Item]" = queue.Queue(maxsize=PREFETCH_DEPTH)
        stop_rd = threading.Event()
        carry: list[_Item] = []
        pending = self._leftover
        self._leftover = []
        eof_known = any(i.eof for i in pending)
        reader = None
        if not eof_known:
            reader = threading.Thread(
                target=self._reader, args=(cap, frame_step, out, stop_rd, carry),
                daemon=True, name="video-reader")
            reader.start()

        have_model = self.engine.loaded
        ema_fps = 0.0
        last_push = 0.0
        prev_pos, prev_t = self.frame_idx, time.perf_counter()
        finished = False
        try:
            while not self._stop.is_set():
                if pending:
                    item = pending.pop(0)
                else:
                    try:
                        item = out.get(timeout=0.2)
                    except queue.Empty:
                        continue
                if item.eof:
                    self.frame_idx = max(self.frame_idx, item.pos)
                    finished = True
                    self._put(Packet("eof", frame_idx=self.frame_idx))
                    return

                frame = item.frame
                if settings.brightness:
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=settings.brightness)
                det = None
                if have_model:
                    det = self.engine.infer(frame)
                    self.counter.update(det, frame_step=item.step)
                self.last_det = det
                self.frame_idx = item.pos

                now = time.perf_counter()
                dt = now - prev_t
                inst = (item.pos - prev_pos) / dt if dt > 0 else 0.0   # video karesi / gerçek sn
                prev_pos, prev_t = item.pos, now
                ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
                if now - last_push >= 0.2:
                    last_push = now
                    snap = None
                    v = self.counter.version
                    if v != self._last_version:
                        self._last_version = v
                        snap = self.counter.snapshot()
                    self._put(Packet("frame", None, snap, self.frame_idx, ema_fps))
        finally:
            stop_rd.set()
            if reader is not None:
                reader.join(2.0)
            if not finished:
                # Duraklatma/durdurma: işlenmemiş kareleri sakla (sıra: eski → yeni)
                rest = list(pending)
                try:
                    while True:
                        rest.append(out.get_nowait())
                except queue.Empty:
                    pass
                rest.extend(carry)
                self._leftover = rest

    def _run(self, settings: Settings, headless: bool = False, frame_step: int = 1) -> None:
        cap = self._cap
        try:
            if headless:
                self._run_headless(settings, frame_step)
                return
            self.annotator.trace_length = settings.trace_length
            target_fps = max(1.0, min(self.source_fps, float(settings.fps_limit)))
            interval = 1.0 / target_fps
            can_drop = settings.realtime_sync
            next_t = time.perf_counter()
            step = 1                      # önceki işlenen kareden bu yana geçen video karesi
            ema_fps = 0.0
            have_model = self.engine.loaded

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._put(Packet("eof", frame_idx=self.frame_idx))
                    return
                self.frame_idx += 1
                t0 = time.perf_counter()

                if settings.brightness:
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=settings.brightness)

                det = None
                if have_model:
                    det = self.engine.infer(frame)
                    self.counter.update(det, frame_step=step)

                annotated = self.annotator.draw(
                    frame, det, self.counter.areas, self.counter.zone_current())
                self.last_det = det

                dw, dh = self._display_size
                if dw > 1 and dh > 1:
                    interp = cv2.INTER_AREA if dw < annotated.shape[1] else cv2.INTER_LINEAR
                    shown = cv2.resize(annotated, (dw, dh), interpolation=interp)
                else:
                    shown = annotated
                rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)

                snap = None
                v = self.counter.version
                if v != self._last_version:
                    self._last_version = v
                    snap = self.counter.snapshot()

                dt = time.perf_counter() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
                self._put(Packet("frame", rgb, snap, self.frame_idx, ema_fps))

                # ── zamanlama ──
                next_t += interval
                step = 1
                slack = next_t - time.perf_counter()
                if slack > 0:
                    self._stop.wait(slack)
                else:
                    if can_drop:
                        n_skip = int(-slack / interval)
                        skipped = 0
                        for _ in range(n_skip):
                            if not cap.grab():
                                break
                            self.frame_idx += 1
                            skipped += 1
                        next_t += n_skip * interval
                        step = 1 + skipped
                    else:
                        next_t = time.perf_counter()  # borç biriktirme
        except Exception as e:  # işçi thread'i sessizce ölmesin
            log.exception("İşleme thread'inde hata")
            self._put(Packet("error", message=f"{type(e).__name__}: {e}"))
