"""
core/logger.py
──────────────
print() yerine logging. Hem konsola hem dosyaya (dosyalar/logs/app.log)
yazar; dosya 1 MB'ı geçince döner (3 yedek).
"""

import logging
import logging.handlers

from core.paths import LOG_DIR

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("goruntu")
    root.setLevel(level)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        root.warning("Log dosyası açılamadı; yalnızca konsola yazılacak.")


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"goruntu.{name}")
