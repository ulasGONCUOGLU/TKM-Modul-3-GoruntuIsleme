"""
core/supabase_client.py
───────────────────────
Supabase'e REST (PostgREST) ile bağlanan küçük istemci. Ek paket gerekmez (httpx).

Yapılandırma `.env` / ortam değişkenlerinden:  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
service_role anahtarı yalnızca bu sunucuda durur; tarayıcıya hiç gitmez.
"""

import base64
import json
import os
from urllib.parse import urlparse

import httpx

from core.env import load_env
from core.logger import get_logger

load_env()
log = get_logger("supabase")


class StorageError(Exception):
    """Veritabanı (Supabase) hatası; mesaj kullanıcıya gösterilebilir."""


class SupabaseClient:
    def __init__(self, url: str, key: str, transport=None):
        self.url = url.rstrip("/")
        self._http = httpx.Client(
            timeout=20.0, transport=transport,
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def request(self, method: str, table: str, params: dict | None = None, json_body=None,
                prefer: str = "") -> httpx.Response:
        headers = {"Prefer": prefer} if prefer else {}
        try:
            r = self._http.request(method, f"{self.url}/rest/v1/{table}", params=params,
                                   json=json_body, headers=headers)
        except httpx.HTTPError as e:
            raise StorageError(f"Supabase'e bağlanılamadı: {e}") from e
        if r.status_code >= 400:
            hint = ""
            try:
                body = r.json()
                msg = body.get("message") or body.get("hint") or r.text
                if body.get("code") in ("42P01", "PGRST205") or "does not exist" in str(msg) \
                        or "schema cache" in str(msg):
                    hint = " (Tablo yok: sql/kurulum.sql dosyasını Supabase SQL Editor'de çalıştırın.)"
                elif r.status_code in (401, 403):
                    hint = " (Anahtar geçersiz: SUPABASE_SERVICE_ROLE_KEY'i kontrol edin.)"
            except ValueError:
                msg = r.text
            raise StorageError(f"Supabase hatası {r.status_code}: {msg}{hint}")
        return r


_client: SupabaseClient | None = None


def key_kind(key: str) -> str:
    """Anahtarın türü: 'service_role' | 'secret' (sb_secret_…) | 'anon' | 'publishable' | 'unknown'."""
    key = (key or "").strip()
    if key.startswith("sb_secret_"):
        return "secret"
    if key.startswith("sb_publishable_"):
        return "publishable"
    try:
        payload = key.split(".")[1]
        role = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("role")
        return role if role in ("service_role", "anon") else "unknown"
    except (IndexError, ValueError):
        return "unknown"


def host() -> str:
    """Bağlı Supabase projesinin adresi (gizli değil) — arayüzde hangi projeye yazıldığını göstermek için."""
    return urlparse(os.environ.get("SUPABASE_URL", "")).netloc


def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL", "").strip() and os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip())


def get_client() -> SupabaseClient:
    global _client
    if _client is None:
        if not is_configured():
            raise StorageError(
                "Supabase ayarlı değil: .env dosyasına SUPABASE_URL ve SUPABASE_SERVICE_ROLE_KEY yazın.")
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()
        if key_kind(key) in ("anon", "publishable"):
            raise StorageError(
                "SUPABASE_SERVICE_ROLE_KEY olarak anon/publishable anahtar girilmiş. Tablolar yalnızca "
                "service_role (ya da sb_secret_…) anahtarıyla okunup yazılabilir: Dashboard → Project Settings → API.")
        _client = SupabaseClient(os.environ["SUPABASE_URL"].strip(), key)
    return _client


def set_client(client: SupabaseClient | None) -> None:
    """Testler için."""
    global _client
    _client = client
