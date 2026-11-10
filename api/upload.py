"""
api/upload.py
─────────────
Büyük dosyaları belleğe almadan diske akıtma.

İstek gövdesi ham (application/octet-stream) okunur ve parça parça yazılır; multipart
ayrıştırması ve ikinci bir geçici kopya yoktur. (Faz 5'te bunun yerine parçalı ve
devam edebilen yükleme eklenecek.)
"""

import shutil
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool


def check_space(directory: Path, needed: int) -> None:
    """Gereken boyut + %5 pay kadar boş yer yoksa 507 döndür."""
    if needed <= 0:
        return
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed * 1.05:
        raise HTTPException(
            507, f"Diskte yeterli yer yok: gereken yaklaşık {needed // (1024 * 1024)} MB, "
                 f"boş {free // (1024 * 1024)} MB.")


def declared_length(request: Request, max_bytes: int) -> int:
    """Content-Length varsa sınırı baştan kontrol et. Yoksa 0 döner."""
    raw = request.headers.get("content-length")
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        return 0
    if n > max_bytes:
        raise HTTPException(
            413, f"Dosya çok büyük (sınır {max_bytes // (1024 * 1024)} MB). "
                 "Sınırı GORUNTU_MAX_UPLOAD_MB ile değiştirebilirsiniz.")
    return n


async def stream_to_file(request: Request, tmp_path: Path, max_bytes: int) -> int:
    """İstek gövdesini tmp_path'e yaz; yazılan bayt sayısını döndür. Hata olursa dosyayı siler."""
    total = 0
    try:
        with open(tmp_path, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        413, f"Dosya çok büyük (sınır {max_bytes // (1024 * 1024)} MB).")
                await run_in_threadpool(f.write, chunk)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return total
