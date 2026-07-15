"""
Abstração da câmera via OpenCV. Aceita duas formas de fonte de vídeo:

1. Câmera USB física local — driver UVC nativo do kernel Linux, sem
   dependências específicas de fabricante (libcamera/picamera2).
   Exemplo: source=0 (índice /dev/video0)

2. Stream de rede (MJPEG/RTSP/RTSPS) — útil para testes sem câmera USB
   física, ou para consumir uma câmera IP/NVR diretamente.
   Exemplo: source="rtsps://10.1.3.200:7441/xxxxx?enableSrtp"

Interface compatível com o pipeline: read / get / release.

Notas sobre RTSP/RTSPS:

- RTSPS só criptografa o canal de *sinalização* RTSP via TLS. O RTP de
  mídia em si, por padrão, ainda tenta ir por UDP — se a rede entre o Pi
  e a câmera/NVR não deixa esse UDP passar (comum atrás de VPN/NAT), o
  canal de controle abre normalmente mas nenhum frame chega, e o ffmpeg
  acaba derrubando a conexão ("[tls @ ...] IO error: End of file").
  Por isso forçamos `rtsp_transport=tcp`, que faz o RTP viajar
  interleaved dentro da mesma conexão TCP/TLS já aberta.
- A leitura roda em uma thread dedicada, separada do consumidor
  (inferência YOLO, lenta na CPU do Pi), sempre expondo o frame mais
  recente e descartando os intermediários — o consumidor nunca fica
  "atrasado" em relação à câmera. Também reconecta automaticamente se a
  leitura falhar repetidas vezes.
"""

import os
import time
import threading

import cv2

# Força o transporte RTP em TCP (interleaved) em vez do padrão UDP do
# ffmpeg. Precisa estar setado antes de cv2.VideoCapture() abrir a fonte.
# Não afeta fontes que não sejam RTSP (ex.: câmera USB local).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class USBCamera:
    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        framerate: int = 30,
        max_falhas_antes_reconectar: int = 5,
        atraso_reconexao_s: float = 2.0,
        atraso_reconexao_maximo_s: float = 30.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.framerate = framerate
        self.max_falhas_antes_reconectar = max_falhas_antes_reconectar
        self.atraso_reconexao_s = atraso_reconexao_s
        self.atraso_reconexao_maximo_s = atraso_reconexao_maximo_s

        self._cap = None
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._parar = threading.Event()

        self._abrir_captura()

        self._thread = threading.Thread(target=self._loop_captura, daemon=True)
        self._thread.start()

    def _abrir_captura(self):
        if self._cap is not None:
            self._cap.release()

        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo '{self.source}'.\n"
                "Se for câmera USB local: verifique com 'v4l2-ctl --list-devices'\n"
                "e confira se o dispositivo aparece como /dev/video0 (ou similar).\n"
                "Se for stream de rede: confirme que a URL/credenciais estão\n"
                "corretas e que o host está acessível a partir do Pi."
            )

        # Nota: em streams de rede (URL), essas chamadas geralmente são
        # ignoradas silenciosamente pelo OpenCV — a resolução real é a que
        # a origem está transmitindo, não a que pedimos aqui.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.framerate)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Timeouts do backend FFmpeg (disponíveis a partir do OpenCV 4.5.4).
        # Evita que uma conexão travada fique presa indefinidamente dentro
        # do cap.read()/cap.open() do FFmpeg.
        timeout_abertura = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        if timeout_abertura is not None:
            cap.set(timeout_abertura, 5000)
        timeout_leitura = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if timeout_leitura is not None:
            cap.set(timeout_leitura, 5000)

        largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura_real  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"🎥 Fonte de vídeo iniciada: {largura_real}x{altura_real} "
            f"(source={self.source!r}, pedido {self.width}x{self.height})",
            flush=True,
        )

        self._cap = cap

    def _loop_captura(self):
        falhas_consecutivas  = 0
        tentativas_reconexao = 0

        while not self._parar.is_set():
            ret, frame = self._cap.read()

            if not ret:
                falhas_consecutivas += 1
                print(
                    f"⚠️  Erro na captura do frame ({falhas_consecutivas}"
                    f"/{self.max_falhas_antes_reconectar})",
                    flush=True,
                )

                with self._lock:
                    self._ret = False

                if falhas_consecutivas >= self.max_falhas_antes_reconectar:
                    # Backoff exponencial: um NVR/câmera que derrubou a
                    # sessão pode levar dezenas de segundos para liberá-la
                    # de novo — reconectar rápido demais só bate na mesma
                    # trava (reset/timeout logo de cara).
                    atraso = min(
                        self.atraso_reconexao_s * (2 ** tentativas_reconexao),
                        self.atraso_reconexao_maximo_s,
                    )
                    print(
                        f"🔄 Muitas falhas seguidas — reconectando em "
                        f"{atraso:.0f}s...",
                        flush=True,
                    )
                    time.sleep(atraso)
                    # Cresce a cada ciclo de reconexão, mesmo que
                    # _abrir_captura() "funcione" (isOpened=True) — no
                    # UniFi Protect isso já aconteceu com a sessão ainda
                    # presa, e as leituras seguintes voltaram a falhar.
                    # Só zera quando um frame é lido de verdade (abaixo).
                    tentativas_reconexao += 1
                    try:
                        self._abrir_captura()
                    except RuntimeError as erro:
                        print(f"❌ Falha ao reconectar: {erro}", flush=True)
                    falhas_consecutivas = 0
                else:
                    time.sleep(0.05)
                continue

            falhas_consecutivas  = 0
            tentativas_reconexao = 0
            with self._lock:
                self._frame = frame
                self._ret = True

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._parar.set()
        self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
