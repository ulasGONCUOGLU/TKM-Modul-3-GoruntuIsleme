-- ============================================================
-- Görüntü İşleme — Supabase kurulum (TEK DOSYA)
-- ============================================================
-- Bu projenin KENDİ Supabase projesinde: Dashboard → SQL Editor → bu dosyanın tamamını yapıştırıp Run.
-- Tekrar çalıştırılabilir (idempotent): hiçbir tabloyu silmez, veriyi bozmaz.
--
-- Tablolar
--   gi_settings   API ayarları (anahtar/değer): eşikler, aktif model, model sınıf seçimleri…
--   gi_analyses   Kaydedilen analiz sonuçları (Nuxt "Analizler" sayfası bunu okur)
--
-- Güvenlik: iki tabloda da RLS AÇIK ve HİÇBİR politika YOK → tarayıcıdaki anon/authenticated anahtar
-- bu tablolara erişemez. Yalnızca service_role (RLS'i atlar) erişir:
--   • Python API   → .env  (yazar: ayarlar + analiz kaydı)
--   • Nuxt sunucusu → .env (okur/siler: analiz kayıtları) — anahtar tarayıcıya gitmez.
-- ============================================================

-- ── Ayarlar ───────────────────────────────────────────────────
create table if not exists public.gi_settings (
  key text primary key,
  value text not null,
  updated_at timestamptz not null default now()
);

create or replace function public.gi_touch_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists gi_settings_touch on public.gi_settings;
create trigger gi_settings_touch before update on public.gi_settings
  for each row execute function public.gi_touch_updated_at();

-- ── Analiz kayıtları ──────────────────────────────────────────
create table if not exists public.gi_analyses (
  id bigint generated always as identity primary key,

  name text not null,                       -- kullanıcının verdiği kayıt adı
  video_name text,                          -- analiz edilen video dosyası
  model text,                               -- kullanılan model (kimlik)
  mode text check (mode in ('live', 'fast')),   -- izleyerek | hızlı analiz
  frame_step int not null default 1,        -- hızlı analizde her N. kare işlendi

  frame_count int,                          -- videodaki toplam kare
  processed_frames int,                     -- analizin ulaştığı kare
  duration_sec numeric,                     -- işlenen video süresi (sn)
  processing_sec numeric,                   -- analizin sürdüğü süre (sn)
  source_fps numeric,
  width int,
  height int,

  total_transitions int not null default 0, -- toplam alan→alan geçişi
  totals jsonb not null default '{}'::jsonb,        -- {"car": 12, "truck": 3}
  transitions jsonb not null default '[]'::jsonb,   -- [{"from","to","count","by_class":{}}]
  zone_visits jsonb not null default '[]'::jsonb,   -- [{"area","count","by_class":{}}]
  areas jsonb not null default '[]'::jsonb,         -- analizde kullanılan alan çokgenleri
  settings jsonb not null default '{}'::jsonb,      -- kullanılan eşikler (conf, iou, tracker, imgsz…)

  created_at timestamptz not null default now()
);

create index if not exists gi_analyses_created_at_idx on public.gi_analyses (created_at desc);

-- ── Güvenlik (politika yok = anon/authenticated erişemez) ─────
alter table public.gi_settings enable row level security;
alter table public.gi_analyses enable row level security;
