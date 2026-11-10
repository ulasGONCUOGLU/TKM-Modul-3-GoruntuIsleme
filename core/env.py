"""
core/env.py
───────────
Proje klasöründeki `.env` dosyasını okur (varsa) ve ortam değişkenlerine ekler.
Zaten tanımlı ortam değişkenlerini EZMEZ. Ek paket gerektirmez.

Biçim:  ANAHTAR=değer   (# ile başlayan satırlar yorumdur, değer tırnaklı olabilir)
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path | None = None) -> list[str]:
    """Okunan anahtarların listesini döndürür."""
    p = path or ENV_FILE
    loaded: list[str] = []
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError:
        return loaded
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
