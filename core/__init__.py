"""
core/
─────
Analiz çekirdeği (arayüzden ve web katmanından bağımsız).

  paths       : Tüm dosya/klasör yolları tek yerde (çalışma dizininden bağımsız)
  logger      : print yerine logging
  database    : Supabase üzerinde ayar deposu (gi_settings)
  config      : Tüm eşikler/ayarlar tek yerde (Settings)
  detection   : YOLO + izleme (DetectionEngine) → supervision.Detections
  zones       : Alan bazlı geçiş sayımı (supervision.PolygonZone)
  annotate    : Kutu / iz / alan çizimi (supervision)
  player      : Video okuma + işleme thread'i (izleyerek ve hızlı analiz)
  areas       : Alanları JSON'a kaydet / yükle
  text        : Türkçe karakter destekli yazı çizimi
  exporter    : .pt → ONNX / OpenVINO / TensorRT dönüştürme
"""
