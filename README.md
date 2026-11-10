# Görüntü İşleme API

Video dosyalarındaki araçları YOLO ile tespit edip takip eden ve **alan → alan geçişlerini sayan** web servisi
(FastAPI, varsayılan port **5001**). Arayüzü Ops Console (Nuxt) içindeki *Görüntü İşleme* modülüdür;
tarayıcı bu servise doğrudan bağlanır. **Veritabanı yalnızca Supabase'tir** (yerel SQLite yok); bu projenin
kendi Supabase projesi vardır ve şeması bu klasördeki `sql/kurulum.sql` dosyasındadır.

Özellikler: tarayıcıdan video ve model yükleme · videonun üzerine alan (çokgen) çizme ve şablon kaydetme ·
**izleyerek** (canlı kutular + sayım) veya **hızlı analiz** (ekransız, ilerleme çubuğu) · sınıf bazlı sayım ·
ayarları ve analiz sonuçlarını Supabase'e kaydetme · modeli OpenVINO'ya dönüştürme.

> Bu proje bir masaüstü uygulaması değildir; eğitim (training) de burada yapılmaz.
> Model başka yerde eğitilip `best.pt` olarak içe aktarılır.

---

## 1. Veritabanı kurulumu (bir kez)

1. [supabase.com](https://supabase.com) → **New project** (bu proje için ayrı bir proje; Yol Kapama'nınkine karıştırmayın).
2. **SQL Editor** → `sql/kurulum.sql` dosyasının tamamını yapıştırıp **Run**.
   Oluşan tablolar: `gi_settings` (ayarlar) ve `gi_analyses` (kaydedilen analizler). Dosya tekrar çalıştırılabilir.
3. **Project Settings → API**: `Project URL` ve `service_role` anahtarını not alın (sonraki adımda `.env`'e yazılır).

İki tabloda da RLS açıktır ve politika yoktur: tarayıcıdaki anon anahtar bu tablolara erişemez; yalnızca
service_role anahtarı (bu API ve Nuxt sunucusu) erişir.

## 2. Python kurulumu

Python **3.10 – 3.12** önerilir.

```powershell
# proje klasöründe
python -m venv venv
venv\Scripts\activate            # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

İlk kurulum torch/ultralytics yüzünden birkaç GB indirir. NVIDIA GPU kullanacaksanız torch'un CUDA sürümünü
[pytorch.org](https://pytorch.org) komutuyla kurun (aksi hâlde CPU kullanılır).

## 3. Ayarlar (`.env`)

```powershell
copy .env.example .env           # sonra .env'i düzenleyin
```

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | — | **Zorunlu.** Bu projenin Supabase adresi ve service_role anahtarı. Yoksa API açılmaz |
| `GORUNTU_HOST` | `127.0.0.1` | Yalnızca bu bilgisayar. Sunucuda `0.0.0.0` (kimlik doğrulama eklenene kadar internete açmayın) |
| `GORUNTU_PORT` | `5001` | |
| `GORUNTU_CORS_ORIGINS` | `http://localhost:3000,...` | Nuxt arayüzünün adresi |
| `GORUNTU_ALLOW_PATH_REGISTER` | `1` | Modeli disk yolunu yazarak kaydetmeye izin verir. Sunucuda `0` yapın |
| `GORUNTU_MAX_UPLOAD_MB` | `16384` | Video yükleme üst sınırı |

## 4. Çalıştırma

```powershell
python app.py
```

Açılışta Supabase bağlantısı sınanır (adres/anahtar yanlışsa ya da tablo yoksa açıklayıcı bir hata ile durur),
sonra adres ve YOLO durumu yazılır. API belgeleri: `http://127.0.0.1:5001/docs`.
Nuxt tarafında `NUXT_PUBLIC_GORUNTU_API_BASE` bu adresi gösterir.

## 5. Veri nerede duruyor?

| Ne | Nerede |
|---|---|
| Ayarlar, aktif model, **model adları**, model sınıf seçimleri | Supabase `gi_settings` |
| Kaydedilen analizler | Supabase `gi_analyses` — **API yalnızca yazar**; listeleme/silme Nuxt sunucusundan yapılır |
| Videolar | Bu bilgisayarın diski: `dosyalar/yuklenen/` |
| Model dosyaları (`best.pt`…) | Bu bilgisayarın diski: `dosyalar/Model/<model adı>/weights/best.pt` |
| Alan (çokgen) dosyaları ve şablonlar | Bu bilgisayarın diski: `dosyalar/alanlar/*.json` |

**Dosyalar veritabanında değil diskte tutulur** (video ve model dosyaları yüzlerce MB–GB; Supabase'in ücretsiz DB kotası 500 MB). `dosyalar/` klasörü Git'e girmez; yedeği ayrıca alın. API başka bir bilgisayarda açılırsa o bilgisayarda bu dosyalar olmaz; veritabanındaki ayarlar ve kayıtlar ise her yerde aynıdır.

**Model adları:** model dosyalarının hepsi `best.pt` olduğundan Modeller sayfasında yüklerken modele bir ad verilir (zorunlu), sonradan *Adı değiştir* ile değiştirilebilir. Ad `gi_settings`'te durur; dosya yolu (kimlik) değişmez. Ad verilmeyen modeller klasör adıyla görünür. Kaydedilen analiz, analiz anındaki model adını taşır (sonradan adı değiştirmek eski kayıtları etkilemez). İki modelde aynı ad (büyük/küçük harf farkı olmadan) kullanılamaz.

Biten bir analiz arayüzde **Veritabanına kaydet** ile kaydedilir; kayıt, sayım tablolarının yanında model, alanlar,
eşikler ve analiz süresini de içerir. Supabase'e ulaşılamazsa kayıt hata verir (HTTP 502) ve oturum açık kaldığı
sürece yeniden denenebilir. Ayar yazılamazsa eski değer korunur.

## 6. Hız

Süre ≈ `(toplam kare ÷ kare atlama) ÷ saniyede işlenen kare`. Örn. 50 dk, 30 fps = 90 000 kare;
saniyede 7 kare işleyen bir CPU'da: her kare = 3,5 sa · her 3. kare = 1,2 sa · her 5. kare = 43 dk.

1. **Kare atlama** (en büyük kazanç): hızlı analizde varsayılan *Otomatik* — etkin hız ≈ 10 kare/sn
   (30 fps video → her 3. kare). Arayüzde 1, 2, 3, 5 de seçilebilir. Alanlar çok küçükse ya da araçlar çok
   hızlıysa adımı düşürün; önemli bir video için bir kez “her kare” ile karşılaştırın.
2. **OpenVINO**: Modeller sayfasında `.pt` için “OpenVINO'ya dönüştür”. CPU'da (özellikle Intel) belirgin
   hızlanma sağlayabilir; kazanç donanıma bağlıdır — analiz ekranındaki *kare/sn* ile karşılaştırın.
3. **Kalite modu** (Ayarlar): *Hızlı* (480) < *Dengeli* (640) < *Kaliteli* (960). Küçüldükçe hızlanır, küçük araçlar
   kaçabilir. (OpenVINO/ONNX modellerde boyut dönüştürme sırasında sabitlenir.)
4. **GPU**: NVIDIA kartta CUDA'lı torch ile on kata varan hız; Cihaz ayarı `auto` ise otomatik kullanılır.
5. Kare okuma ayrı thread'de yapılır (çözme süresi çıkarımın içinde gizlenir) — kazanç videoya/işlemciye göre değişir.

## 7. Klasör yapısı

```
app.py            başlatıcı
api/              FastAPI uçları, oturum yönetimi, analiz kaydı (Supabase'e yazar)
core/             tespit + takip, alan sayımı, video okuma, ayarlar, Supabase istemcisi
sql/kurulum.sql    Supabase tabloları (tek dosya; veritabanı şemasının tek kaynağı bu projedir)
dosyalar/         çalışma verisi (git'e girmez): yuklenen/ (videolar), Model/, alanlar/, logs/
```

## 8. Uçlar (özet)

`/health` · `/videos` · `/area-templates` · `/models` (+ `/models/export`) · `/settings` ·
`/sessions` (başlat, pause/resume/stop, `/stream` MJPEG, `/events` SSE, `/save`).
Tümü `/docs` altında denenebilir.

## 9. Sorun giderme

* **“ultralytics kurulu değil”** → sanal ortam etkin mi? `pip install -r requirements.txt`
* **Arayüz “API'ye bağlanılamadı” diyor** → `python app.py` çalışıyor mu, `.env`'de `GORUNTU_CORS_ORIGINS` Nuxt adresini içeriyor mu?
* **Açılışta “Supabase'e erişilemedi / Tablo yok”** → `.env` değerleri doğru mu, `sql/kurulum.sql` çalıştırıldı mı?
* **“401/403 Anahtar geçersiz”** → `SUPABASE_SERVICE_ROLE_KEY` için `anon` değil `service_role` anahtarı gerekir.
* **Video açılamadı** → codec desteklenmiyor olabilir; videoyu H.264 (.mp4) olarak dışa aktarıp yeniden deneyin.
* Günlükler: `dosyalar/logs/app.log`
