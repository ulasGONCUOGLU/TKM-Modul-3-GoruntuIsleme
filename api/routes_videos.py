"""
api/routes_videos.py
────────────────────
Video yükleme, ilk kare, alanlar (poligonlar) ve alan şablonları.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool

from api import videos_store as vs
from api.config import MAX_UPLOAD_BYTES, UPLOAD_DIR, VIDEO_EXTS
from api.schemas import AreasPayload, TemplatePayload
from api.state import state
from api.upload import check_space, declared_length, stream_to_file
from api.util import encode_jpeg, safe_filename
from core.areas import load_areas, save_areas
from core.paths import AREAS_DIR

router = APIRouter(tags=["videolar"])


def _video_or_404(video_id: str) -> Path:
    p = vs.find_video(video_id)
    if p is None:
        raise HTTPException(404, "Video bulunamadı.")
    return p


# ── yükleme ───────────────────────────────────────────────────────────
@router.post("/videos", status_code=201)
async def upload_video(request: Request, name: str = Query(min_length=1, max_length=255)):
    """
    Ham gövde olarak video yükler:  POST /videos?name=trafik.mp4  (gövde = dosyanın baytları)
    Dosya belleğe alınmadan diske akıtılır.
    """
    safe = safe_filename(name, "video")
    if Path(safe).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(
            415, f"Desteklenmeyen video türü. İzin verilenler: {', '.join(sorted(VIDEO_EXTS))}")

    declared = declared_length(request, MAX_UPLOAD_BYTES)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    check_space(UPLOAD_DIR, declared)

    vid = vs.new_id()
    target = UPLOAD_DIR / vs.video_filename(vid, safe)
    tmp = target.with_name(target.name + ".part")

    total = await stream_to_file(request, tmp, MAX_UPLOAD_BYTES)
    if total == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "Boş dosya.")
    tmp.replace(target)

    info = await run_in_threadpool(vs.read_info, target)
    if info is None:
        target.unlink(missing_ok=True)
        raise HTTPException(422, "Video açılamadı. Dosya bozuk ya da codec desteklenmiyor olabilir.")
    return {**vs.describe(target, with_info=False), **info,
            "suggested_frame_step": vs.suggest_frame_step(info["fps"])}


@router.get("/videos")
async def list_videos():
    return {"videos": await run_in_threadpool(vs.list_videos)}


@router.get("/videos/{video_id}")
async def video_detail(video_id: str):
    p = _video_or_404(video_id)
    return await run_in_threadpool(vs.describe, p, True)


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: str):
    _video_or_404(video_id)
    cur = state.sessions.current
    if cur is not None and cur.video_id == video_id and cur.is_active:
        raise HTTPException(409, "Bu video şu an analiz ediliyor. Önce oturumu durdurun.")
    vs.delete_video(video_id)
    return Response(status_code=204)


@router.get("/videos/{video_id}/first-frame")
async def first_frame(video_id: str):
    p = _video_or_404(video_id)
    frame = await run_in_threadpool(vs.read_first_frame, p)
    if frame is None:
        raise HTTPException(422, "Videodan kare okunamadı.")
    jpg = await run_in_threadpool(encode_jpeg, frame, 88)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ── alanlar (videoya bağlı) ───────────────────────────────────────────
def _areas_response(areas: list[dict], w: int, h: int) -> dict:
    return {
        "frame_width": w,
        "frame_height": h,
        "areas": [{"name": a["name"], "points": [[int(x), int(y)] for x, y in a["points"]]}
                  for a in areas],
    }


@router.get("/videos/{video_id}/areas")
async def get_areas(video_id: str):
    p = _video_or_404(video_id)
    info = await run_in_threadpool(vs.read_info, p)
    if info is None:
        raise HTTPException(422, "Video açılamadı.")
    size = (info["width"], info["height"])
    f = vs.areas_file(video_id)
    if not f.exists():
        return _areas_response([], *size)
    return _areas_response(load_areas(f, size), *size)       # videonun çözünürlüğüne ölçekler


@router.put("/videos/{video_id}/areas")
async def put_areas(video_id: str, body: AreasPayload):
    p = _video_or_404(video_id)
    info = await run_in_threadpool(vs.read_info, p)
    if info is None:
        raise HTTPException(422, "Video açılamadı.")
    size = (info["width"], info["height"])
    src = (body.frame_width or size[0], body.frame_height or size[1])
    areas = [{"name": a.name, "points": a.points} for a in body.areas]
    AREAS_DIR.mkdir(parents=True, exist_ok=True)
    save_areas(vs.areas_file(video_id), areas, src)
    return _areas_response(load_areas(vs.areas_file(video_id), size), *size)


# ── alan şablonları (videodan bağımsız, yeniden kullanılır) ───────────
def _template_path(name: str) -> Path:
    n = name.strip()
    if not n or "/" in n or "\\" in n or n.casefold().startswith("video_"):
        raise HTTPException(400, "Geçersiz şablon adı.")
    return AREAS_DIR / f"{n}.json"


@router.get("/area-templates")
def list_templates():
    out = []
    if AREAS_DIR.exists():
        for f in sorted(AREAS_DIR.glob("*.json")):
            if f.stem.casefold().startswith("video_"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                out.append({"name": f.stem,
                            "frame_width": data.get("frame_width"),
                            "frame_height": data.get("frame_height"),
                            "area_count": len(data.get("areas", []))})
            except (OSError, ValueError):
                continue
    return {"templates": out}


@router.get("/area-templates/{name}")
def get_template(name: str, width: int | None = Query(default=None, ge=1),
                 height: int | None = Query(default=None, ge=1)):
    """width ve height verilirse noktalar o çözünürlüğe ölçeklenir."""
    f = _template_path(name)
    if not f.exists():
        raise HTTPException(404, "Şablon bulunamadı.")
    target = (width, height) if width and height else None
    areas = load_areas(f, target)
    data = json.loads(f.read_text(encoding="utf-8"))
    w = width or data.get("frame_width") or 0
    h = height or data.get("frame_height") or 0
    return _areas_response(areas, w, h)


@router.post("/area-templates", status_code=201)
def save_template(body: TemplatePayload):
    if not body.frame_width or not body.frame_height:
        raise HTTPException(422, "Şablon için frame_width ve frame_height gerekli.")
    f = _template_path(body.name)
    AREAS_DIR.mkdir(parents=True, exist_ok=True)
    save_areas(f, [{"name": a.name, "points": a.points} for a in body.areas],
               (body.frame_width, body.frame_height))
    return {"name": f.stem, "area_count": len(body.areas)}


@router.delete("/area-templates/{name}", status_code=204)
def delete_template(name: str):
    f = _template_path(name)
    if not f.exists():
        raise HTTPException(404, "Şablon bulunamadı.")
    f.unlink()
    return Response(status_code=204)
