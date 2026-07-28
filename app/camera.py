"""
Abstração da câmera via OpenCV. Aceita duas formas de fonte de vídeo:

1. Câmera USB física local — driver UVC nativo do kernel Linux, sem
   dependências específicas de fabricante (libcamera/picamera2).
   Exemplo: source=0 (índice /dev/video0)

2. Stream de rede (MJPEG/RTSP/RTSPS) — câmera IP (ex.: UniFi Protect) na
   rede local, ou outro dispositivo (ex.: notebook rodando webcam_server.py)
   como fonte de imagem via rede.
   Exemplo: source="rtsps://192.168.0.10:7441/token?enableSrtp"

Uma thread interna fica lendo continuamente da fonte e guarda só o frame
mais recente — assim, se a inferência (mais lenta que a câmera) demorar
para pedir o próximo frame, ela nunca fica processando um backlog atrasado,
sempre pega o que há de mais atual. Se a leitura falhar repetidas vezes
(stream caiu), a própria thread tenta reconectar sozinha, com backoff
exponencial pra não martelar a câmera/rede quando ela já está com problema.

Streams de rede via FFmpeg podem deixar cap.open()/cap.read() bloqueado por
bastante tempo numa queda de conexão. Isso é resolvido com os timeouts
internos do próprio OpenCV (CAP_PROP_OPEN_TIMEOUT_MSEC/CAP_PROP_READ_TIMEOUT_MSEC)
— NÃO com uma segunda thread chamando release() no objeto de fora: essa
abordagem foi tentada aqui antes e causou um segfault, porque release()
rodando ao mesmo tempo que read() no mesmo VideoCapture (duas threads, um
objeto só) corrompe o estado interno do FFmpeg. Os timeouts nativos
resolvem o mesmo problema (não travar pra sempre) sem esse risco, porque
rodam dentro da própria chamada bloqueante, numa thread só.

Interface compatível com o pipeline: read / get / release.
"""

import os
import threading
import time
from datetime import datetime

import cv2

# RTSP sobre TCP evita a corrupção de frame (pacotes UDP perdidos) em rede
# instável — o cenário típico de Wi-Fi. Só é consultado pelo backend FFmpeg
# (streams de rede via URL); o backend V4L2 usado por câmera USB local
# ignora essa variável, então é seguro deixar sempre setada.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

FALHAS_PARA_RECONECTAR = 10
ESPERA_INICIAL         = 2.0
ESPERA_MAXIMA          = 30.0

# Timeout nativo do OpenCV/FFmpeg para abrir a conexão ou ler um frame. Sem
# isso o default é ~30s (foi o que vimos travar de verdade em produção);
# baixamos pra reconectar mais rápido quando a rede cai. Só vale pro backend
# FFmpeg (streams de rede) — é ignorado pelo V4L2 (câmera USB local).
TIMEOUT_FFMPEG_MSEC = 15000


class USBCamera:
    def __init__(self, source=0, width: int = 1280, height: int = 720, framerate: int = 30):
        self.source    = source
        self.width     = width
        self.height    = height
        self.framerate = framerate

        self._cap                = None
        self._cap_lock           = threading.Lock()
        self._frame_lock         = threading.Lock()
        self._frame_atual        = None
        self._frame_novo         = False
        self._encerrar           = False
        self._ts_conexao_aberta  = None
        self._ts_ultima_queda    = None

        self._abrir_captura()

        self._thread = threading.Thread(target=self._loop_leitura, daemon=True)
        self._thread.start()

    def _abrir_captura(self):
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()

            cap = cv2.VideoCapture()
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_FFMPEG_MSEC)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_FFMPEG_MSEC)
            cap.open(self.source)
            self._cap = cap

            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Não foi possível abrir a fonte de vídeo '{self.source}'.\n"
                    "Se for câmera USB local: verifique com 'v4l2-ctl --list-devices'\n"
                    "e confira se o dispositivo aparece como /dev/video0 (ou similar).\n"
                    "Se for stream de rede: confirme que o servidor/câmera está no ar\n"
                    "e que o IP/porta/token na URL estão corretos."
                )

            # Nota: em streams de rede (URL), essas chamadas geralmente são
            # ignoradas silenciosamente pelo OpenCV — a resolução real é a que
            # o servidor de origem está transmitindo, não a que pedimos aqui.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.framerate)

            largura_real = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura_real  = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._ts_conexao_aberta = time.time()
            print(
                f"🎥 [{datetime.now().strftime('%H:%M:%S')}] Fonte de vídeo iniciada: "
                f"{largura_real}x{altura_real} "
                f"(source={self.source!r}, pedido {self.width}x{self.height})",
                flush=True,
            )

    def _loop_leitura(self):
        falhas_seguidas = 0
        espera_atual    = ESPERA_INICIAL

        while not self._encerrar:
            with self._cap_lock:
                cap = self._cap
            ret, frame = cap.read()

            if not ret:
                falhas_seguidas += 1
                print(f"⚠️  Erro na captura do frame ({falhas_seguidas}/{FALHAS_PARA_RECONECTAR})", flush=True)

                if falhas_seguidas >= FALHAS_PARA_RECONECTAR:
                    agora = time.time()
                    duracao_sessao = (
                        agora - self._ts_conexao_aberta if self._ts_conexao_aberta else None
                    )
                    intervalo_queda_anterior = (
                        agora - self._ts_ultima_queda if self._ts_ultima_queda else None
                    )
                    detalhes = f"sessão durou {duracao_sessao:.1f}s" if duracao_sessao is not None else "duração da sessão desconhecida"
                    if intervalo_queda_anterior is not None:
                        detalhes += f" | {intervalo_queda_anterior:.1f}s desde a queda anterior"
                    self._ts_ultima_queda = agora

                    print(
                        f"🔁 [{datetime.now().strftime('%H:%M:%S')}] Muitas falhas seguidas "
                        f"({detalhes}) — tentando reconectar à fonte de vídeo...",
                        flush=True,
                    )
                    try:
                        self._abrir_captura()
                        falhas_seguidas = 0
                        espera_atual    = ESPERA_INICIAL  # sucesso: reseta o backoff
                    except RuntimeError as e:
                        print(
                            f"❌ Falha ao reconectar: {e} "
                            f"(próxima tentativa em {espera_atual:.0f}s)",
                            flush=True,
                        )
                        time.sleep(espera_atual)
                        espera_atual = min(espera_atual * 2, ESPERA_MAXIMA)
                else:
                    time.sleep(0.05)
                continue

            falhas_seguidas = 0
            with self._frame_lock:
                self._frame_atual = frame
                self._frame_novo  = True

    def read(self):
        with self._frame_lock:
            if not self._frame_novo:
                return False, None
            self._frame_novo = False
            return True, self._frame_atual

    def get(self, prop):
        with self._cap_lock:
            return self._cap.get(prop)

    def release(self):
        self._encerrar = True
        self._thread.join(timeout=2)
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
