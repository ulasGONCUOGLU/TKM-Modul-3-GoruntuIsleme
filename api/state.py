"""
api/state.py
────────────
Süreç genelinde paylaşılan nesneler. Testler `state.engine`'i sahte motorla değiştirir.
"""

import threading

from api.session import SessionManager


class _State:
    def __init__(self):
        self.engine = None                  # DetectionEngine; ilk kullanımda oluşturulur
        self.sessions = SessionManager()
        self._lock = threading.Lock()

    def get_engine(self):
        with self._lock:
            if self.engine is None:
                from core.detection import DetectionEngine
                self.engine = DetectionEngine()
            return self.engine


state = _State()
