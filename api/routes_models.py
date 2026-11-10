"""
api/routes_models.py
────────────────────
Model listesi, yükleme (tarayıcıdan), elle yol kaydı, aktif model, sayılacak sınıflar.
Eğitim burada YOK: model başka yerde eğitilip best.pt olarak içe aktarılır.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool

from api import export_job
from api import models_store as ms
from api.config import ALLOW_PATH_REGISTER, MAX_MODEL_UPLOAD_BYTES, MODEL_EXTS
from api.schemas import ClassesPayload, ExportRequest, ModelId, ModelRegister, ModelRename
from api.state import state
from api.upload import check_space, declared_length, stream_to_file
from api.util import safe_filename
from core.config import DEFAULT_ALLOWED_CLASSES
from core.paths import DOSYALAR_DIR, MODEL_TRAIN_DIR

router = APIRouter(tags=["modeller"])


def _model_or_404(model_id: str) -> Path:
    p = ms.resolve(model_id)
    if p is None:
        raise HTTPException(404, "Model bulunamadı.")
    return p


def _busy_guard() -> None:
    cur = state.sessions.current
    if cur is not None and cur.is_active:
        raise HTTPException(409, "Analiz sürerken model değiştirilemez.")
    if export_job.is_running():
        raise HTTPException(409, "Model dönüştürme sürerken bu işlem yapılamaz.")


@router.get("/models")
def list_models():
    active = ms.canon(ms.active_id() or "")
    return {"models": ms.scan(), "active": active,
            "active_name": ms.display_name(active) if active else "",
            "path_register_enabled": ALLOW_PATH_REGISTER}


def _check_label(name: str, except_id: str | None = None) -> str:
    label = ms.clean_label(name)
    if ms.label_taken(label, except_id):
        raise HTTPException(409, f"“{label}” adında başka bir model var. Farklı bir ad verin.")
    return label


@router.put("/models/name")
def rename_model(body: ModelRename):
    """Modelin görünen adını değiştirir (dosya ve kimlik değişmez). Boş ad = klasör adına dön."""
    _model_or_404(body.id)
    label = _check_label(body.name, body.id)
    ms.set_label(body.id, label)
    return {"id": ms.canon(body.id), "name": ms.display_name(body.id), "custom_name": bool(label)}


@router.post("/models", status_code=201)
async def upload_model(request: Request, name: str = Query(min_length=1, max_length=255),
                       label: str = Query(default="", max_length=200)):
    """
    Ham gövde olarak model yükler: POST /models?name=best.pt&label=Araç tespiti v6
    `name` yüklenen dosyanın adı, `label` modelin görünen adıdır (verilmezse dosya adından türetilir).
    """
    safe = safe_filename(name, "best.pt")
    ext = Path(safe).suffix.lower()
    if ext not in MODEL_EXTS:
        raise HTTPException(415, f"Desteklenmeyen model türü. İzin verilenler: {', '.join(sorted(MODEL_EXTS))}")

    label = _check_label(label) if label.strip() else ""
    declared = declared_length(request, MAX_MODEL_UPLOAD_BYTES)
    # Klasör adı: verilen ad varsa ondan, yoksa dosya adından. Dosyanın kendisi hep best.<uzantı>.
    folder = safe_filename(label, "model") if label else (Path(safe).stem or "model")

    # Aynı klasör adı varsa numara ekle (üzerine yazma)
    n = 1
    target_dir = MODEL_TRAIN_DIR / folder / "weights"
    while (target_dir / f"best{ext}").exists():
        n += 1
        target_dir = MODEL_TRAIN_DIR / f"{folder}_{n}" / "weights"
    target_dir.mkdir(parents=True, exist_ok=True)
    check_space(target_dir, declared)

    target = target_dir / f"best{ext}"
    tmp = target.with_name(target.name + ".part")
    total = await stream_to_file(request, tmp, MAX_MODEL_UPLOAD_BYTES)
    if total == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "Boş dosya.")
    tmp.replace(target)

    import os
    mid = ms.canon(os.path.relpath(target, DOSYALAR_DIR))
    if label:
        ms.set_label(mid, label)
    return {"id": mid, "name": ms.display_name(mid), "kind": ms.kind_of(target), "size_bytes": total}


@router.post("/models/register", status_code=201)
def register_model(body: ModelRegister):
    """Disk üzerindeki bir modeli yolunu yazarak kaydet (dosya kopyalanmaz)."""
    if not ALLOW_PATH_REGISTER:
        raise HTTPException(403, "Bu sunucuda elle yol kaydı kapalı. Modeli tarayıcıdan yükleyin.")
    raw = body.path.strip().strip('"')
    if not ms.is_abs(raw):
        raise HTTPException(422, "Tam (mutlak) yol girin. Örn: C:\\Modeller\\best.pt")
    p = Path(ms.canon(raw))
    if not p.exists():
        raise HTTPException(404, f"Dosya bulunamadı: {raw}")
    if p.is_file() and p.suffix.lower() not in MODEL_EXTS:
        raise HTTPException(415, f"Desteklenmeyen model türü: {p.suffix}")
    if p.is_dir() and not p.name.endswith("_openvino_model"):
        raise HTTPException(415, "Klasör yalnızca *_openvino_model olabilir.")
    label = _check_label(body.name, raw) if body.name and body.name.strip() else ""
    ms.add_extra(raw)
    if label:
        ms.set_label(raw, label)
    return {"id": ms.canon(raw), "name": ms.display_name(raw), "kind": ms.kind_of(p),
            "size_bytes": ms.size_of(p)}


@router.delete("/models/register", status_code=204)
def unregister_model(id: str = Query(min_length=1)):
    """Yalnızca kaydı siler; diskteki dosyaya dokunmaz."""
    if not ms.remove_extra(id):
        raise HTTPException(404, "Bu yol kayıtlı değil.")
    return Response(status_code=204)


@router.put("/models/active")
def set_active(body: ModelId):
    _busy_guard()
    _model_or_404(body.id)
    ms.set_active(body.id)
    return {"active": ms.canon(body.id)}


@router.get("/models/classes")
async def get_classes(id: str = Query(min_length=1)):
    path = _model_or_404(id)
    engine = state.get_engine()
    try:
        names = await run_in_threadpool(ms.model_names, path, engine)
    except Exception as e:                              # noqa: BLE001
        raise HTTPException(503, f"Model sınıfları okunamadı: {e}") from e
    saved = ms.saved_classes(id)
    selected = saved if saved else [n for n in names if n in DEFAULT_ALLOWED_CLASSES] or names
    return {"id": ms.canon(id), "names": names, "selected": [n for n in selected if n in names],
            "custom": bool(saved),
            "warning": None if saved else ms.class_warning(names, DEFAULT_ALLOWED_CLASSES)}


@router.put("/models/classes")
async def put_classes(body: ClassesPayload, id: str = Query(min_length=1)):
    _busy_guard()
    path = _model_or_404(id)
    engine = state.get_engine()
    try:
        names = await run_in_threadpool(ms.model_names, path, engine)
    except Exception as e:                              # noqa: BLE001
        raise HTTPException(503, f"Model sınıfları okunamadı: {e}") from e
    unknown = [c for c in body.classes if c not in names]
    if unknown:
        raise HTTPException(422, f"Modelde olmayan sınıf: {', '.join(unknown)}")
    ms.save_classes(id, body.classes)
    return {"id": ms.canon(id), "selected": body.classes, "custom": bool(body.classes)}


# ── .pt → OpenVINO / ONNX (CPU'da hızlandırma) ────────────────────────
@router.post("/models/export", status_code=202)
async def export_start(body: ExportRequest):
    _busy_guard()
    p = _model_or_404(body.id)
    if p.suffix.lower() != ".pt":
        raise HTTPException(422, "Yalnızca .pt modeller dönüştürülebilir.")
    from core.detection import YOLO_AVAILABLE
    if not YOLO_AVAILABLE:
        raise HTTPException(503, "ultralytics kurulu değil.")
    from core.config import Settings
    imgsz = body.imgsz or Settings.load().imgsz
    try:
        export_job.start(ms.canon(body.id), p, body.format, imgsz)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return export_job.status()


@router.get("/models/export")
def export_status():
    return export_job.status()
