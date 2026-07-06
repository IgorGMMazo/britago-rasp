"""
Abstração da câmera USB via OpenCV (driver UVC — nativo do kernel Linux,
sem dependências específicas de fabricante como libcamera/picamera2).
Interface compatível com o pipeline: read / get / release.
"""

import cv2


class USBCamera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, framerate: int = 30):
        self.width = width
        self.height = height
        self._cap = cv2.VideoCapture(index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera USB no índice {index}.\n"
                "Verifique com: v4l2-ctl --list-devices\n"
                "e confira se o dispositivo aparece como /dev/video0 (ou similar)."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, framerate)

        largura_real = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura_real  = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"🎥 Câmera USB iniciada: {largura_real}x{altura_real} "
            f"(índice {index}, pedido {width}x{height})",
            flush=True,
        )

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            print("⚠️  Erro na captura do frame", flush=True)
            return False, None
        return True, frame

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._cap.release()