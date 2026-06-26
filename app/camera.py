"""
Abstração da câmera CSI via picamera2.
Interface compatível com cv2.VideoCapture (read / get / release).
"""

import cv2

try:
    from picamera2 import Picamera2
    _PICAM_OK = True
except ImportError:
    _PICAM_OK = False


class CSICamera:
    def __init__(self, width: int = 1280, height: int = 720, framerate: int = 30):
        if not _PICAM_OK:
            raise RuntimeError(
                "picamera2 não encontrada.\n"
                "Instale com: sudo apt install -y python3-picamera2\n"
                "e crie o venv com: python3 -m venv venv --system-site-packages"
            )
        self.width  = width
        self.height = height
        self._cam   = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": float(framerate)},
        )
        self._cam.configure(cfg)
        self._cam.start()
        print(f"🎥 Câmera CSI iniciada: {width}x{height} @ {framerate}fps", flush=True)

    def read(self):
        try:
            frame_rgb = self._cam.capture_array()
            return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️  Erro na captura: {e}", flush=True)
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self):
        try:
            self._cam.stop()
            self._cam.close()
        except Exception:
            pass
