"""
Inferência YOLO + ByteTrack + dHash — câmera CSI (Raspberry Pi 4B).

Captura frames pela entrada CSI via picamera2, detecta pedras grandes dentro
do ROI e salva recortes "limpos" (sem marcações) em PASTA_SAIDA.
A triagem por dHash evita salvar a mesma pedra repetidamente.
"""

import os
import cv2
import time
import queue
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import imagehash

from camera import CSICamera


# ── Configuração via variáveis de ambiente ─────────────────────────────────
def _env_float(nome, padrao):
    return float(os.getenv(nome, padrao))

def _env_int(nome, padrao):
    return int(os.getenv(nome, padrao))


WEIGHTS     = os.getenv("WEIGHTS", "weights/best.pt")
PASTA_SAIDA = os.getenv("PASTA_SAIDA", "pedras_grandes")
DEVICE      = os.getenv("DEVICE", "cpu")

CAMERA_WIDTH     = _env_int("CAMERA_WIDTH",     1280)
CAMERA_HEIGHT    = _env_int("CAMERA_HEIGHT",    720)
CAMERA_FRAMERATE = _env_int("CAMERA_FRAMERATE", 30)

CONF                   = _env_float("CONF",                   0.30)
IOU                    = _env_float("IOU",                    0.45)
IMGSZ                  = _env_int("IMGSZ",                    640)
AREA_MINIMA            = _env_float("AREA_MINIMA",            0.12)
MARGEM_COMPARACAO      = _env_int("MARGEM_COMPARACAO",        22)
LIMIAR_HAMMING         = _env_int("LIMIAR_HAMMING",           30)
MIN_FRAMES_CONFIRMACAO = _env_int("MIN_FRAMES_CONFIRMACAO",   8)

ROI_X    = _env_int("ROI_X", 1)
ROI_Y    = _env_int("ROI_Y", 44)
ROI_W    = _env_int("ROI_W", 945)
ROI_H    = _env_int("ROI_H", 674)
AREA_ROI = ROI_W * ROI_H

Path(PASTA_SAIDA).mkdir(parents=True, exist_ok=True)


# ── Fila e contadores ──────────────────────────────────────────────────────
fila_processamento = queue.Queue()
fotos_salvas       = 0
fotos_bloqueadas   = 0
lock_contadores    = threading.Lock()


# ── Thread de triagem por dHash ────────────────────────────────────────────
def worker_filtro_hashing():
    global fotos_salvas, fotos_bloqueadas

    print("🧠 Módulo de Triagem por dHash inicializado.", flush=True)
    historico_hashes = []
    max_historico    = 20

    while True:
        item = fila_processamento.get()
        if item is None:
            break

        crop_comparacao, imagem_recortada, track_id, proporcao = item

        img_pil    = Image.fromarray(cv2.cvtColor(crop_comparacao, cv2.COLOR_BGR2RGB))
        hash_atual = imagehash.dhash(img_pil)

        e_repetido      = False
        menor_distancia = 999
        for hash_antigo in historico_hashes:
            distancia       = hash_atual - hash_antigo
            menor_distancia = min(menor_distancia, distancia)
            if distancia <= LIMIAR_HAMMING:
                e_repetido = True
                break

        if e_repetido:
            with lock_contadores:
                fotos_bloqueadas += 1
            print(f"🔁 dHash BLOQUEOU  ID#{track_id} | distância={menor_distancia} (≤{LIMIAR_HAMMING})", flush=True)
        else:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            nome = str(Path(PASTA_SAIDA) / f"pedra_id{track_id}_{ts}_crop.jpg")
            cv2.imwrite(nome, imagem_recortada)

            historico_hashes.append(hash_atual)
            if len(historico_hashes) > max_historico:
                historico_hashes.pop(0)

            with lock_contadores:
                fotos_salvas += 1
            print(f"📸 SALVO           ID#{track_id} | {proporcao*100:.1f}% do ROI | {Path(nome).name}", flush=True)

        fila_processamento.task_done()


thread_filtro = threading.Thread(target=worker_filtro_hashing, daemon=True)
thread_filtro.start()


# ── Detecção e rastreamento ────────────────────────────────────────────────
def ponto_dentro_roi(cx, cy, rx, ry, rw, rh) -> bool:
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


print(f"🚀 Carregando modelo {WEIGHTS} em device={DEVICE}...", flush=True)
model = YOLO(WEIGHTS)

smoother        = sv.DetectionsSmoother(length=5)
contagem_frames: dict[int, int] = {}
ids_salvos:      set[int]       = set()

cam     = CSICamera(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FRAMERATE)
largura = CAMERA_WIDTH
altura  = CAMERA_HEIGHT

frame_count = 0
t_log       = time.time()

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count         += 1
        frame_original_limpo = frame.copy()

        results = model.track(
            source  = frame,
            imgsz   = IMGSZ,
            conf    = CONF,
            iou     = IOU,
            tracker = "bytetrack.yaml",
            persist = True,
            device  = DEVICE,
            verbose = False,
        )

        detections = sv.Detections.from_ultralytics(results[0])
        detections = smoother.update_with_detections(detections)

        ids_frame_atual = set()

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if not ponto_dentro_roi(cx, cy, ROI_X, ROI_Y, ROI_W, ROI_H):
                continue

            ids_frame_atual.add(track_id)
            contagem_frames[track_id] = contagem_frames.get(track_id, 0) + 1

            area_box  = (x2 - x1) * (y2 - y1)
            proporcao = area_box / AREA_ROI
            eh_grande = proporcao >= AREA_MINIMA
            confirmado = contagem_frames[track_id] >= MIN_FRAMES_CONFIRMACAO

            if eh_grande and confirmado and track_id not in ids_salvos:
                cx1 = max(0, x1 - MARGEM_COMPARACAO)
                cy1 = max(0, y1 - MARGEM_COMPARACAO)
                cx2 = min(largura, x2 + MARGEM_COMPARACAO)
                cy2 = min(altura,  y2 + MARGEM_COMPARACAO)
                crop = frame_original_limpo[cy1:cy2, cx1:cx2]

                if crop.size > 0:
                    fila_processamento.put((crop, crop.copy(), track_id, proporcao))
                    ids_salvos.add(track_id)

        ids_sumidos = set(contagem_frames.keys()) - ids_frame_atual
        for tid in ids_sumidos:
            del contagem_frames[tid]

        if frame_count % 300 == 0:
            agora = time.time()
            fps   = 300.0 / (agora - t_log + 1e-9)
            t_log = agora
            with lock_contadores:
                print(
                    f"❤️  frames={frame_count} | fps≈{fps:.1f} | "
                    f"salvas={fotos_salvas} | bloqueadas={fotos_bloqueadas} | "
                    f"fila={fila_processamento.qsize()}",
                    flush=True,
                )

except KeyboardInterrupt:
    print("\n🛑 Encerrando inferência...", flush=True)
finally:
    fila_processamento.put(None)
    cam.release()
    print(
        f"✅ Pipeline encerrado. Salvas={fotos_salvas} Bloqueadas={fotos_bloqueadas}",
        flush=True,
    )
