"""
Abstração da câmera via OpenCV. Aceita duas formas de fonte de vídeo:

1. Câmera USB física local — driver UVC nativo do kernel Linux, sem
   dependências específicas de fabricante (libcamera/picamera2).
   Exemplo: source=0 (índice /dev/video0)

2. Stream de rede (MJPEG/RTSP) — útil para testes sem câmera USB física,
   usando outro dispositivo (ex.: notebook rodando webcam_server.py) como
   fonte de imagem via rede local.
   Exemplo: source="http://192.168.0.15:5000/video"

Interface compatível com o pipeline: read / get / release.
"""

import cv2


class USBCamera:
    def __init__(self, source=0, width: int = 1280, height: int = 720, framerate: int = 30):
        self.width = width
        self.height = height
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo '{source}'.\n"
                "Se for câmera USB local: verifique com 'v4l2-ctl --list-devices'\n"
                "e confira se o dispositivo aparece como /dev/video0 (ou similar).\n"
                "Se for stream de rede: confirme que o servidor (ex.: webcam_server.py)\n"
                "está rodando e que o IP/porta na URL estão corretos."
            )

        # Nota: em streams de rede (URL), essas chamadas geralmente são
        # ignoradas silenciosamente pelo OpenCV — a resolução real é a que
        # o servidor de origem está transmitindo, não a que pedimos aqui.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, framerate)

        largura_real = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura_real  = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"🎥 Fonte de vídeo iniciada: {largura_real}x{altura_real} "
            f"(source={source!r}, pedido {width}x{height})",
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