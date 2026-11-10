"""
api/routes_sessions.py
──────────────────────
Analiz oturumları: başlat / duraklat / devam / durdur, MJPEG akışı, SSE olayları, kaydet.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from api import analysis_store
from api.schemas import SaveRequest, SessionCreate
from api.session import Session, SessionError
from api.state import state

router = APIRouter(tags=["oturumlar"])


def _get(session_id: str) -> Session:
    s = state.sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Oturum bulunamadı.")
    return s


def _wrap(fn, *a):
    try:
        return fn(*a)
    except SessionError as e:
        raise HTTPException(e.status, e.message) from e


@router.post("/sessions", status_code=201)
async def create_session(body: SessionCreate):
    engine = state.get_engine()

    def _go():
        return state.sessions.create(engine, body.video_id, body.mode, body.frame_step)
    try:
        sess = await run_in_threadpool(_go)       # model yükleme yavaş olabilir
    except SessionError as e:
        raise HTTPException(e.status, e.message) from e
    return sess.status()


@router.get("/sessions/current")
def current_session():
    s = state.sessions.current
    if s is None:
        return {"session": None}
    return {"session": s.status(), "counts": s.counts()}


@router.get("/sessions/{session_id}")
def session_detail(session_id: str):
    s = _get(session_id)
    return {"session": s.status(), "counts": s.counts()}


@router.post("/sessions/{session_id}/pause")
async def pause(session_id: str):
    s = _get(session_id)
    await run_in_threadpool(_wrap, state.sessions.pause, s)
    return s.status()


@router.post("/sessions/{session_id}/resume")
async def resume(session_id: str):
    s = _get(session_id)
    await run_in_threadpool(_wrap, state.sessions.resume, s)
    return s.status()


@router.post("/sessions/{session_id}/stop")
async def stop(session_id: str):
    s = _get(session_id)
    await run_in_threadpool(state.sessions.stop, s)
    return s.status()


@router.post("/sessions/{session_id}/save", status_code=201)
async def save(session_id: str, body: SaveRequest):
    s = _get(session_id)
    rid = await run_in_threadpool(_wrap, state.sessions.save, s, body.name)
    return {"record_id": rid, "database": analysis_store.describe()["host"]}


# ── MJPEG ─────────────────────────────────────────────────────────────
@router.get("/sessions/{session_id}/stream")
async def stream(session_id: str):
    s = _get(session_id)
    if s.mode != "live":
        raise HTTPException(409, "Hızlı analizde görüntü akışı yoktur.")

    async def gen():
        last = -1
        while True:
            if s.jpeg is not None and s.jpeg_seq != last:
                last = s.jpeg_seq
                jpg = s.jpeg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            elif not s.is_active:
                return
            await asyncio.sleep(0.03)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"})


# ── SSE ───────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/sessions/{session_id}/events")
async def events(session_id: str):
    s = _get(session_id)

    async def gen():
        last_version = None
        last_state = None
        yield _sse("status", s.status())
        while True:
            c = s.counts()
            if c["version"] != last_version:
                last_version = c["version"]
                yield _sse("counts", c)
            st = s.status()
            yield _sse("status", st)
            if not s.is_active:
                if last_state == st["state"]:
                    yield _sse("counts", s.counts())
                    return
            last_state = st["state"]
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
