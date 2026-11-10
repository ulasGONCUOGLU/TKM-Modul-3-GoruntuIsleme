"""
api/main.py
───────────
FastAPI uygulaması.  Çalıştırma:  python -m api
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api import analysis_store
from api.routes_models import router as models_router
from api.routes_sessions import router as sessions_router
from api.routes_settings import router as settings_router
from api.routes_videos import router as videos_router
from api.state import state
from core.database import init_db
from core.supabase_client import StorageError
from core.paths import ensure_dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()                           # ayarları Supabase'ten yükler; ulaşılamazsa açılmaz
    yield
    state.sessions.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Görüntü İşleme API", version=config.API_VERSION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(StorageError)
    async def storage_error(_: Request, e: StorageError):
        return JSONResponse(status_code=502, content={"detail": str(e)})

    @app.get("/health", tags=["genel"])
    def health():
        try:
            from core.detection import YOLO_AVAILABLE
        except Exception:                      # noqa: BLE001
            YOLO_AVAILABLE = False
        return {"status": "ok", "version": config.API_VERSION, "ultralytics": bool(YOLO_AVAILABLE),
                "path_register_enabled": config.ALLOW_PATH_REGISTER,
                "storage": analysis_store.describe()}

    for r in (videos_router, models_router, settings_router, sessions_router):
        app.include_router(r)
    return app


app = create_app()
