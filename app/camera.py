"""
Abstração da câmera via OpenCV. Aceita duas formas de fonte de vídeo:

1. Câmera USB física local — driver UVC nativo do kernel Linux, sem
   dependências específicas de fabricante (libcamera/picamera2).
   Exemplo: source=0 (índice /dev/video0)

2. Stream de rede (MJPEG/RTSP) — útil para testes sem câmera USB física,
   usando outro dispositivo (ex.: notebook rodando webcam_server.py) como
   fonte de imagem via rede local.
   Exemplo: source="http://192.168.0.15:5000/video"

Uma thread interna fica lendo continuamente da fonte e guarda só o frame
mais recente — assim, se a inferência (mais lenta que a câmera) demorar
para pedir o próximo frame, ela nunca fica processando um backlog atrasado,
sempre pega o que há de mais atual. Se a leitura falhar repetidas vezes
(stream caiu), a própria thread tenta reconectar sozinha.

Interface compatível com o pipeline: read / get / release.
"""

import threading
import time
from datetime import datetime

import cv2

FALHAS_PARA_RECONECTAR = 10
ESPERA_ENTRE_TENTATIVAS = 2.0


class USBCamera:
    def __init__(self, source=0, width: int = 1280, height: int = 720, framerate: int = 30):
        self.source    = source
        self.width     = width
        self.height    = height
        self.framerate = framerate

        self._cap                = None
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
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo '{self.source}'.\n"
                "Se for câmera USB local: verifique com 'v4l2-ctl --list-devices'\n"
                "e confira se o dispositivo aparece como /dev/video0 (ou similar).\n"
                "Se for stream de rede: confirme que o servidor (ex.: webcam_server.py)\n"
                "está rodando e que o IP/porta na URL estão corretos."
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

        while not self._encerrar:
            ret, frame = self._cap.read()

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
                    except RuntimeError as e:
                        print(f"❌ Falha ao reconectar: {e}", flush=True)
                        time.sleep(ESPERA_ENTRE_TENTATIVAS)
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
        return self._cap.get(prop)

    def release(self):
        self._encerrar = True
        self._thread.join(timeout=2)
        self._cap.release()
