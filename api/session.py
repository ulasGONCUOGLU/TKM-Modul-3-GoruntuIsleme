"""
api/session.py
──────────────
Analiz oturumu: sunucuda çalışır, tarayıcıdan bağımsızdır.

  * Tek aktif oturum (DetectionEngine tek; ikinci istek "meşgul" döner).
  * `core.VideoPlayer` kendi işçi thread'ini çalıştırır; burada bir "pump" thread'i
    player.queue'yu boşaltır, JPEG'e çevirir ve son durumu bellekte tutar.
  * MJPEG ve SSE uçları bu durumu okur; tarayıcı kapansa da analiz sürer.

Durumlar: starting → running ⇄ paused → finished | stopped | error
"""

import threading
import time
import uuid
from dataclasses import dataclass, field


from api import analysis_store, api_settings, export_job, models_store
from api import videos_store as vs
from api.util import rgb_to_jpeg
from core.annotate import FrameAnnotator
from core.config import Settings
from core.logger import get_logger
from core.player import VideoPlayer
from core.zones import CountSnapshot, ZoneCounter

log = get_logger("api.session")

ACTIVE_STATES = ("starting", "running", "paused")


class SessionError(Exception):
    """İstemciye gösterilecek anlaşılır hata. `status` HTTP kodudur."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def snapshot_to_dict(snap: CountSnapshot | None) -> dict:
    if snap is None:
        return {"version": -1, "transitions": [], "zone_visits": [], "totals": {}}
    transitions = []
    for (a, b), n in snap.transitions.items():
        by_class = snap.transitions_by_class.get((a, b), {})
        transitions.append({"from": a, "to": b, "count": n, "by_class": by_class})
    visits = []
    for area, n in snap.zone_visits.items():
        visits.append({
            "area": area, "count": n,
            "by_class": snap.zone_visits_by_class.get(area, {}),
            "current": snap.zone_current.get(area, 0),
        })
    totals: dict[str, int] = {}
    for per in snap.transitions_by_class.values():
        for cls, n in per.items():
            totals[cls] = totals.get(cls, 0) + n
    return {"version": snap.version, "total": snap.total_transitions,
            "transitions": transitions, "zone_visits": visits, "totals": totals}


@dataclass
class Session:
    id: str
    video_id: str
    video_name: str
    mode: str                       # 'live' | 'fast'
    frame_step: int
    player: VideoPlayer
    counter: ZoneCounter
    frame_count: int
    source_fps: float
    width: int
    height: int
    state: str = "starting"
    message: str = ""
    warning: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    frame_idx: int = 0
    proc_fps: float = 0.0
    saved_record_id: int | None = None
    model_id: str = ""
    model_name: str = ""          # başlangıçtaki görünen ad (kayıtta bu yazılır)
    areas: list = field(default_factory=list)
    paused_total: float = 0.0
    paused_at: float | None = None
    jpeg: bytes | None = None
    jpeg_seq: int = 0
    snapshot: CountSnapshot | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(default_factory=threading.Condition)
    pump: threading.Thread | None = None
    settings: Settings | None = None

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def progress(self) -> float | None:
        if self.frame_count > 0:
            return min(1.0, self.frame_idx / self.frame_count)
        return None

    def active_sec(self) -> float:
        """Çalışma süresi (duraklatılan süre hariç)."""
        now = self.finished_at or time.time()
        paused = self.paused_total + ((now - self.paused_at) if self.paused_at else 0.0)
        return max(0.0, now - self.created_at - paused)

    def status(self) -> dict:
        elapsed = self.active_sec()
        prog = self.progress()
        eta = None
        if self.state == "running" and prog and 0.01 < prog < 1 and elapsed > 1:
            eta = elapsed * (1 - prog) / prog
        return {
            "id": self.id, "video_id": self.video_id, "video_name": self.video_name,
            "mode": self.mode, "frame_step": self.frame_step,
            "state": self.state, "message": self.message, "warning": self.warning,
            "frame_idx": self.frame_idx, "frame_count": self.frame_count,
            "progress": prog, "fps": round(self.proc_fps, 1),
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 0) if eta is not None else None,
            "source_fps": self.source_fps, "width": self.width, "height": self.height,
            "saved_record_id": self.saved_record_id,
        }

    def counts(self) -> dict:
        with self.lock:
            snap = self.snapshot
        return snapshot_to_dict(snap)


class SessionManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.current: Session | None = None
        self._by_id: dict[str, Session] = {}

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._by_id.get(session_id)

    # ── başlatma ──────────────────────────────────────────────────
    def create(self, engine, video_id: str, mode: str, frame_step: int | None = None) -> Session:
        with self._lock:
            if self.current is not None and self.current.is_active:
                raise SessionError(
                    "Başka bir analiz sürüyor. Önce onu durdurun ya da bitmesini bekleyin.", 409)

            if export_job.is_running():
                raise SessionError("Model dönüştürme sürerken analiz başlatılamaz.", 409)
            path = vs.find_video(video_id)
            if path is None:
                raise SessionError("Video bulunamadı.", 404)

            areas_f = vs.areas_file(video_id)
            if not areas_f.exists():
                raise SessionError("Bu video için alan çizilmemiş. Önce en az 1 alan çizip kaydedin.", 422)

            model_id = models_store.active_id()
            model_path = models_store.resolve(model_id) if model_id else None
            if model_path is None:
                raise SessionError("Aktif model seçili değil ya da dosyası bulunamadı. Modeller sayfasından seçin.", 422)

            settings = Settings.load()
            allowed = models_store.saved_classes(model_id)
            if allowed:
                settings.allowed_classes = tuple(allowed)

            player_counter = ZoneCounter(settings.min_frames_in_zone)
            annotator = FrameAnnotator(settings.trace_length)
            player = VideoPlayer(engine, player_counter, annotator)
            if not player.open(str(path)):
                raise SessionError("Video açılamadı.", 422)

            from core.areas import load_areas
            areas = load_areas(areas_f, (player.frame_width, player.frame_height))
            if not areas:
                player.release()
                raise SessionError("Kayıtlı alan bulunamadı. En az 1 alan çizip kaydedin.", 422)
            player_counter.set_areas(areas)

            try:
                engine.configure(settings)
                if not engine.loaded or models_store.canon(engine.model_path or "") != models_store.canon(str(model_path)):
                    engine.load(str(model_path))
                engine.configure(settings)          # sınıf filtresini yüklü modele göre yeniden hesapla
            except Exception as e:                    # noqa: BLE001
                player.release()
                log.exception("Model yüklenemedi")
                raise SessionError(f"Model yüklenemedi: {e}", 500) from e

            annotator.set_classes(engine.names)
            engine.reset_tracker()

            names = [engine.names[i] for i in sorted(engine.names)]
            warn = None
            if not allowed:
                from core.config import DEFAULT_ALLOWED_CLASSES
                warn = models_store.class_warning(names, DEFAULT_ALLOWED_CLASSES)

            if mode == "fast":
                step = frame_step or vs.suggest_frame_step(player.source_fps)   # None = otomatik
            else:
                step = 1
            sess = Session(
                id=uuid.uuid4().hex[:12], video_id=video_id, video_name=vs.display_name(path),
                mode=mode, frame_step=step,
                player=player, counter=player_counter,
                frame_count=player.frame_count, source_fps=player.source_fps,
                width=player.frame_width, height=player.frame_height, warning=warn)
            sess.settings = settings
            sess.model_id = models_store.canon(model_id)
            sess.model_name = models_store.display_name(model_id)
            sess.areas = [{"name": a["name"], "points": [[int(x), int(y)] for x, y in a["points"]]}
                          for a in areas]
            self._by_id[sess.id] = sess
            self.current = sess

        self._start_player(sess)
        return sess

    def _start_player(self, sess: Session) -> None:
        settings = sess.settings
        headless = sess.mode == "fast"
        target = max(640, int(api_settings.stream_max_width()))
        if not headless:
            w, h = sess.width, sess.height
            if w > target:
                h = int(round(h * target / w))
                w = target
            sess.player.set_display_size(w, h)
        ok = sess.player.start(settings, headless=headless, frame_step=sess.frame_step)
        if not ok:
            sess.state, sess.message = "error", "Analiz başlatılamadı."
            return
        sess.state = "running"
        if sess.pump is None or not sess.pump.is_alive():
            sess.pump = threading.Thread(target=self._pump, args=(sess,), daemon=True,
                                         name=f"pump-{sess.id}")
            sess.pump.start()

    # ── pump: player.queue → oturum durumu ────────────────────────
    def _pump(self, sess: Session) -> None:
        q = sess.player.queue
        quality = api_settings.jpeg_quality()
        while True:
            if sess.state in ("stopped", "error", "finished"):
                return
            try:
                pkt = q.get(timeout=0.25)
            except Exception:                         # queue.Empty
                continue
            if pkt.kind == "frame":
                sess.frame_idx = pkt.frame_idx
                sess.proc_fps = pkt.fps
                if pkt.snapshot is not None:
                    with sess.lock:
                        sess.snapshot = pkt.snapshot
                if pkt.display_rgb is not None:
                    jpg = rgb_to_jpeg(pkt.display_rgb, quality)
                    with sess.cond:
                        sess.jpeg = jpg
                        sess.jpeg_seq += 1
                        sess.cond.notify_all()
            elif pkt.kind == "eof":
                with sess.lock:
                    sess.snapshot = sess.counter.snapshot()
                sess.frame_idx = max(sess.frame_idx, pkt.frame_idx)
                self._finish(sess, "finished")
                return
            elif pkt.kind == "error":
                sess.message = pkt.message or "Bilinmeyen hata"
                self._finish(sess, "error")
                return

    def _finish(self, sess: Session, state: str) -> None:
        sess.state = state
        sess.finished_at = time.time()
        if sess.paused_at:
            sess.paused_total += sess.finished_at - sess.paused_at
            sess.paused_at = None
        with sess.lock:
            sess.snapshot = sess.counter.snapshot()
        with sess.cond:
            sess.cond.notify_all()
        try:
            sess.player.release()
        except Exception:                             # noqa: BLE001
            log.exception("Video serbest bırakılamadı")

    # ── kontrol ───────────────────────────────────────────────────
    def pause(self, sess: Session) -> None:
        if sess.state != "running":
            raise SessionError("Oturum çalışmıyor.", 409)
        sess.player.stop()
        sess.paused_at = time.time()
        sess.state = "paused"

    def resume(self, sess: Session) -> None:
        if sess.state != "paused":
            raise SessionError("Oturum duraklatılmış değil.", 409)
        sess.player.drain()
        if sess.paused_at:
            sess.paused_total += time.time() - sess.paused_at
            sess.paused_at = None
        sess.state = "running"
        if not sess.player.start(sess.settings, headless=sess.mode == "fast",
                                 frame_step=sess.frame_step):
            sess.state, sess.message = "error", "Devam ettirilemedi."
            raise SessionError("Devam ettirilemedi.", 500)

    def stop(self, sess: Session) -> None:
        if not sess.is_active:
            return
        sess.state = "stopped"                        # pump bunu görüp çıkar
        sess.player.stop()
        self._finish(sess, "stopped")

    def shutdown(self) -> None:
        with self._lock:
            cur = self.current
        if cur is not None and cur.is_active:
            self.stop(cur)

    # ── kaydetme ──────────────────────────────────────────────────
    def save(self, sess: Session, name: str) -> int:
        if sess.state in ("starting", "running"):
            raise SessionError("Analiz sürerken kaydedilemez. Durdurun ya da bitmesini bekleyin.", 409)
        if sess.saved_record_id is not None:
            return sess.saved_record_id
        with sess.lock:
            snap = sess.snapshot
        if snap is None or (snap.total_transitions == 0 and not any(snap.zone_visits.values())):
            raise SessionError("Kaydedilecek sayım yok.", 422)
        data = snapshot_to_dict(snap)
        s = sess.settings
        record = {
            "name": name, "video_name": sess.video_name,
            "model": sess.model_name or sess.model_id,
            "mode": sess.mode, "frame_step": sess.frame_step,
            "frame_count": sess.frame_count, "processed_frames": sess.frame_idx,
            "duration_sec": round(sess.frame_idx / sess.source_fps, 1) if sess.source_fps else None,
            "processing_sec": round(sess.active_sec(), 1),
            "source_fps": sess.source_fps, "width": sess.width, "height": sess.height,
            "total_transitions": int(data.get("total") or 0),
            "totals": data["totals"], "transitions": data["transitions"],
            "zone_visits": [{k: v for k, v in z.items() if k != "current"} for z in data["zone_visits"]],
            "areas": sess.areas,
            "settings": {"count_conf": s.count_conf, "iou": s.iou, "tracker": s.tracker,
                         "quality_mode": s.mode, "imgsz": s.imgsz, "model_id": sess.model_id,
                         "min_frames_in_zone": s.min_frames_in_zone,
                         "classes": list(s.allowed_classes)},
        }
        try:
            rid = analysis_store.save(record)
        except analysis_store.StorageError as e:
            raise SessionError(str(e), 502) from e
        sess.saved_record_id = rid
        return rid
