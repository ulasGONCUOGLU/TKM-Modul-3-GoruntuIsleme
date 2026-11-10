"""
api/models_store.py
───────────────────
Model dosyalarının bulunması, aktif model, modele özel sayılacak sınıflar.

Kimlik (id) kuralı:
  * dosyalar/ altındaki model  → DOSYALAR_DIR'e göre göreli yol, "/" ile yazılır
                                  (örn. "Model/Car_Detection_v6/weights/best.pt")
  * elle kaydedilen model      → mutlak yol (örn. "C:/Modeller/best.pt")
Karşılaştırmalarda "\\" ve "/" aynı sayılır (Windows yolları).

`active_model`, model adları (`api.model_labels`) ve sınıf seçimleri Supabase `gi_settings` tablosunda saklanır.

Görünen AD ile kimlik (id) ayrıdır: id dosya yoludur ve hiç değişmez; ad kullanıcının verdiği, istediği zaman
değiştirilebilen etikettir. Ad verilmemişse klasör adından türetilir (Model/<klasör>/weights/best.pt → <klasör>).
"""

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from core.database import get_setting, set_setting
from core.paths import DOSYALAR_DIR

# Model klasörlerinde aranan dosya adları
MODEL_FILE_NAMES = ("best.pt", "best.onnx", "best.engine")
EXTRA_KEY = "api.extra_models"
LABELS_KEY = "api.model_labels"
MAX_LABEL = 60


def canon(model_id: str) -> str:
    return str(model_id).replace("\\", "/")


def is_abs(model_id: str) -> bool:
    return PureWindowsPath(model_id).is_absolute() or PurePosixPath(canon(model_id)).is_absolute()


# ── elle kaydedilen modeller ──────────────────────────────────────────
def extras() -> list[str]:
    raw = get_setting(EXTRA_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [canon(x) for x in data if isinstance(x, str)]
    except ValueError:
        return []


def _save_extras(items: list[str]) -> None:
    set_setting(EXTRA_KEY, json.dumps(sorted(set(items)), ensure_ascii=False))


def add_extra(path: str) -> None:
    _save_extras(extras() + [canon(path)])


def remove_extra(path: str) -> bool:
    cur = extras()
    p = canon(path)
    if p not in cur:
        return False
    _save_extras([x for x in cur if x != p])
    set_label(p, "")                              # kayıt silinince adı da gitsin
    return True


# ── model adları (kullanıcının verdiği etiketler) ─────────────────────
def _labels() -> dict[str, str]:
    raw = get_setting(LABELS_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return {canon(k): v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v.strip()}


def clean_label(name: str) -> str:
    return " ".join(str(name).split())[:MAX_LABEL]


def default_label(model_id: str) -> str:
    """Ad verilmemişse klasörden türetilen ad."""
    p = PurePosixPath(canon(model_id))
    suffix = ""
    if p.name.endswith("_openvino_model"):
        suffix = " (OpenVINO)"
    elif p.suffix.lower() == ".onnx":
        suffix = " (ONNX)"
    elif p.suffix.lower() == ".engine":
        suffix = " (TensorRT)"
    parent = p.parent
    if parent.name.lower() == "weights" and parent.parent.name:
        base = parent.parent.name
    elif parent.name and parent.name not in (".", "/"):
        base = parent.name
    else:
        base = p.stem
    return base + suffix


def display_name(model_id: str) -> str:
    return _labels().get(canon(model_id)) or default_label(model_id)


def has_custom_label(model_id: str) -> bool:
    return canon(model_id) in _labels()


def set_label(model_id: str, name: str) -> None:
    """Boş ad = özel adı kaldır (klasörden türetilen ada dön)."""
    labels = _labels()
    cid = canon(model_id)
    name = clean_label(name)
    if name:
        labels[cid] = name
    else:
        labels.pop(cid, None)
    set_setting(LABELS_KEY, json.dumps(labels, ensure_ascii=False))


def label_taken(name: str, except_id: str | None = None) -> bool:
    """Aynı ad (büyük/küçük harf farkı olmadan) başka bir modelde var mı?"""
    want = clean_label(name).casefold()
    if not want:
        return False
    skip = canon(except_id) if except_id else None
    return any(m["name"].casefold() == want for m in scan() if m["id"] != skip)


# ── bulma ─────────────────────────────────────────────────────────────
def kind_of(path: Path) -> str:
    if path.is_dir() and path.name.endswith("_openvino_model"):
        return "openvino"
    return path.suffix.lower().lstrip(".")


def size_of(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def resolve(model_id: str) -> Path | None:
    """id → disk yolu. Yalnızca dosyalar/ altındaki ya da kaydedilmiş (extras) modeller kabul edilir."""
    if not model_id:
        return None
    cid = canon(model_id)
    if is_abs(cid):
        if cid not in extras():
            return None
        p = Path(cid)
        return p if p.exists() else None
    base = DOSYALAR_DIR.resolve()
    p = (DOSYALAR_DIR / cid).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        return None
    return p if p.exists() else None


def scan() -> list[dict]:
    """dosyalar/ altındaki ve elle kaydedilen modellerin listesi (aktif olan işaretli)."""
    active = canon(active_id() or "")
    found: list[tuple[str, Path, str]] = []

    if DOSYALAR_DIR.is_dir():
        for kok, dizinler, dosyalar in os.walk(DOSYALAR_DIR):
            for d in list(dizinler):
                if d.endswith("_openvino_model"):
                    full = Path(kok) / d
                    found.append((canon(os.path.relpath(full, DOSYALAR_DIR)), full, "dosyalar"))
                    dizinler.remove(d)          # içine inme
            for f in dosyalar:
                if f in MODEL_FILE_NAMES:
                    full = Path(kok) / f
                    found.append((canon(os.path.relpath(full, DOSYALAR_DIR)), full, "dosyalar"))

    for e in extras():
        p = Path(e)
        if p.exists():
            found.append((e, p, "kayıtlı yol"))

    out = []
    for mid, path, origin in sorted(found, key=lambda x: x[0]):
        out.append({
            "id": mid,
            "name": display_name(mid),
            "custom_name": has_custom_label(mid),
            "kind": kind_of(path),
            "size_bytes": size_of(path),
            "origin": origin,
            "active": mid == active,
        })
    return out


# ── aktif model ───────────────────────────────────────────────────────
def active_id() -> str | None:
    return get_setting("active_model")


def set_active(model_id: str) -> None:
    cid = canon(model_id)
    # Göreli yollar işletim sisteminin ayırıcısıyla yazılır.
    stored = cid if is_abs(cid) else os.path.normpath(cid)
    set_setting("active_model", stored)


# ── modele özel sayılacak sınıflar ────────────────────────────────────
def _classes_key(model_id: str) -> str:
    return f"api.classes.{canon(model_id)}"


def saved_classes(model_id: str) -> list[str] | None:
    raw = get_setting(_classes_key(model_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    names = [x for x in data if isinstance(x, str)]
    return names or None


def save_classes(model_id: str, names: list[str]) -> None:
    """Boş liste = kayıtlı seçimi sil (varsayılan listeye dön)."""
    set_setting(_classes_key(model_id), json.dumps(names, ensure_ascii=False) if names else "")


_names_cache: dict[tuple[str, float], list[str]] = {}


def model_names(path: Path, engine) -> list[str]:
    """Modelin sınıf adları. Yüklü modelse onun adları; değilse ağırlık dosyası okunur (önbellekli)."""
    if engine.loaded and engine.model_path and canon(engine.model_path) == canon(str(path)):
        return [engine.names[i] for i in sorted(engine.names)]
    key = (str(path), path.stat().st_mtime)
    if key in _names_cache:
        return _names_cache[key]
    from core.detection import YOLO, YOLO_AVAILABLE
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics kurulu değil: pip install ultralytics")
    m = YOLO(str(path), task="detect")
    names = [m.names[i] for i in sorted(m.names)]
    _names_cache[key] = names
    return names


def class_warning(names: list[str], allowed: tuple | list) -> str | None:
    """Varsayılan izinli listeyle yalnızca bazı sınıflar eşleşiyorsa kullanıcıyı uyar."""
    allowed_set = set(allowed)
    matched = [n for n in names if n in allowed_set]
    others = [n for n in names if n not in allowed_set]
    if matched and others:
        return (f"Varsayılan sınıf listesiyle yalnızca şunlar eşleşiyor: {', '.join(matched)}. "
                f"Şunlar sayılmayacak: {', '.join(others)}. "
                "Sayılacak sınıfları Modeller sayfasından seçebilirsiniz.")
    return None
