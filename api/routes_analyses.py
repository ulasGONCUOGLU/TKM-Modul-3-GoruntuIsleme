"""
api/routes_analyses.py
──────────────────────
Kaydedilmiş analizler: listele / ayrıntı / sil. Depo Supabase ya da yerel SQLite olabilir.
"""

from fastapi import APIRouter, HTTPException, Query, Response

from api import analysis_store
from api.analysis_store import StorageError

router = APIRouter(tags=["analizler"])


def _store():
    return analysis_store.get_store()


def _storage_http(e: StorageError) -> HTTPException:
    return HTTPException(502, str(e))


@router.get("/analyses")
def list_analyses(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)):
    try:
        items, total = _store().list(limit, offset)
    except StorageError as e:
        raise _storage_http(e) from e
    return {"items": items, "total": total, "storage": analysis_store.describe()}


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int):
    try:
        row = _store().get(analysis_id)
    except StorageError as e:
        raise _storage_http(e) from e
    if row is None:
        raise HTTPException(404, "Analiz kaydı bulunamadı.")
    return row


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: int):
    try:
        ok = _store().delete(analysis_id)
    except StorageError as e:
        raise _storage_http(e) from e
    if not ok:
        raise HTTPException(404, "Analiz kaydı bulunamadı.")
    return Response(status_code=204)
