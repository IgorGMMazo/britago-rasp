"""
Abstração da câmera via OpenCV. Aceita duas formas de fonte de vídeo:
1. Câmera USB física local — driver UVC nativo do kernel Linux, sem
   dependências específicas de fabricante (libcamera/picamera2).
   Exemplo: source=0 (índice /dev/video0)
2. Stream de rede (MJPEG/RTSP) — útil para testes sem câmera USB física,
   ou câmeras IP reais (ex.: UniFi Protect via RTSP).
   Exemplo: source="rtsp://10.1.3.200:7447/TOKEN"

Interface compatível com o pipeline: read / get / release.

Inclui reconexão automática: streams RTSP caem eventualmente (rede,
reinício do servidor, etc). Sem isso, uma queda mata o pipeline até
alguém reiniciar manualmente.
"""
import os
import time
import cv2

# Força TCP no RTSP (reduz artefatos/erros de decodificação por perda
# de pacote em UDP). Precisa vir antes de qualquer VideoCapture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class USBCamera:
    def __init__(self, source=0, width: int = 1280, height: int = 720, framerate: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.framerate = framerate
        self._falhas_seguidas = 0
        self._max_falhas_antes_reconectar = 10
        self._cap = None
        self._abrir()

    def _abrir(self):
        if self._cap is not None:
            self._cap.release()

        if isinstance(self.source, str):
            # Stream de rede (RTSP/MJPEG): força o backend FFmpeg para que
            # OPENCV_FFMPEG_CAPTURE_OPTIONS (rtsp_transport;tcp) tenha efeito.
            # Sem isso o OpenCV pode cair no GStreamer, que ignora a opção.
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            # Câmera USB local (índice /dev/videoN) → backend padrão (V4L2).
            self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo '{self.source}'.\n"
                "Se for câmera USB local: verifique com 'v4l2-ctl --list-devices'\n"
                "e confira se o dispositivo aparece como /dev/video0 (ou similar).\n"
                "Se for stream de rede: confirme que o servidor está rodando e\n"
                "que o IP/porta/token na URL estão corretos."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.framerate)
        # Buffer mínimo: evita acumular frames antigos em caso de
        # processamento mais lento que a taxa de chegada do stream.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        largura_real = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura_real  = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"🎥 Fonte de vídeo iniciada: {largura_real}x{altura_real} "
            f"(source={self.source!r}, pedido {self.width}x{self.height})",
            flush=True,
        )
        self._falhas_seguidas = 0

    def read(self):
        ret, frame = self._cap.read()

        if not ret:
            self._falhas_seguidas += 1
            print(
                f"⚠️  Erro na captura do frame ({self._falhas_seguidas}/"
                f"{self._max_falhas_antes_reconectar})",
                flush=True,
            )

            if self._falhas_seguidas >= self._max_falhas_antes_reconectar:
                print("🔄 Muitas falhas seguidas — reconectando à fonte de vídeo...", flush=True)
                try:
                    self._abrir()
                except RuntimeError as e:
                    print(f"❌ Falha ao reconectar: {e}", flush=True)
                    time.sleep(2)

            return False, None

        self._falhas_seguidas = 0
        return True, frame

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        if self._cap is not None:
            self._cap.release()
