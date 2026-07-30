"""
Inferência YOLO + ByteTrack + dHash — câmera USB (Raspberry Pi 4B, Ubuntu Server). 123

Captura frames pela câmera USB via OpenCV, detecta pedras grandes dentro
do ROI e salva recortes "limpos" (sem marcações) em PASTA_SAIDA.
A triagem por dHash evita salvar a mesma pedra repetidamente.
"""

import os
import cv2
import time
import queue
import base64
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
from ultralytics import YOLO

# Workaround: fuse() do ultralytics troca para forward_fuse, que causa
# "Illegal instruction" no Cortex-A72 (Pi 4). Desabilita o fuse por completo.
import ultralytics.nn.tasks as _tasks
_tasks.BaseModel.fuse = lambda self, verbose=True: self
import supervision as sv
from PIL import Image
import imagehash

from camera import USBCamera

# ── Configuração via variáveis de ambiente ─────────────────────────────────
def _env_float(nome, padrao):
    return float(os.getenv(nome, padrao))

def _env_int(nome, padrao):
    return int(os.getenv(nome, padrao))


WEIGHTS     = os.getenv("WEIGHTS", "weights/best.pt")
PASTA_SAIDA = os.getenv("PASTA_SAIDA", "pedras_grandes")
DEVICE      = os.getenv("DEVICE", "cpu")

# Webhook de debug: se definido, envia periodicamente o frame mais recente em
# que alguma pedra foi identificada dentro do ROI (mesmo as que nunca viram
# "grande" o bastante pra passar pelos filtros de PASTA_SAIDA). Serve pra
# diagnosticar se a detecção está funcionando quando pedras_grandes/ fica
# vazia. Opcional — se não for setado, esse envio fica desligado.
URL_WEBHOOK_DEBUG      = os.getenv("URL_WEBHOOK_DEBUG", "")
INTERVALO_WEBHOOK_DEBUG = _env_int("INTERVALO_WEBHOOK_DEBUG", 5)

_camera_raw = os.getenv("CAMERA_INDEX", "0")
CAMERA_SOURCE = int(_camera_raw) if _camera_raw.isdigit() else _camera_raw

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

# ── Diagnóstico de saúde (RAM/temperatura/throttling) ──────────────────────
# Ajuda a investigar travamentos após rodar por muito tempo: se o processo
# vazar memória (há relatos conhecidos disso em model.track(persist=True) do
# ultralytics) ou o Pi estiver superaquecendo, isso aparece nos logs do
# systemd (journalctl -u britago) bem antes do próximo travamento, sem
# precisar reproduzir o problema pra descobrir a causa.

_FLAGS_THROTTLED = {
    0:  "under-voltage (agora)",
    1:  "freq limitada (agora)",
    2:  "throttled (agora)",
    3:  "limite térmico soft (agora)",
    16: "under-voltage (já ocorreu)",
    17: "freq limitada (já ocorreu)",
    18: "throttled (já ocorreu)",
    19: "limite térmico soft (já ocorreu)",
}


def memoria_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for linha in f:
                if linha.startswith("VmRSS:"):
                    return int(linha.split()[1]) / 1024
    except Exception:
        return None
    return None


def temperatura_cpu_c():
    try:
        saida = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
        )
        if saida.returncode == 0:
            # formato: "temp=53.8'C"
            return float(saida.stdout.split("=")[1].split("'")[0])
    except Exception:
        pass

    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000
    except Exception:
        return None


def status_throttling():
    try:
        saida = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        )
        if saida.returncode != 0:
            return None
        bitmask = int(saida.stdout.strip().split("=")[1], 16)
        ativos = [msg for bit, msg in _FLAGS_THROTTLED.items() if bitmask & (1 << bit)]
        return ", ".join(ativos) if ativos else "OK"
    except Exception:
        return None


# ── Frames de debug periódicos: 1 a cada N segundos, com ROI + detecções ──
FRAMES_DEBUG_DIR       = os.getenv("FRAMES_DEBUG_DIR", "frames_debug")
FRAMES_DEBUG_INTERVALO = _env_int("FRAMES_DEBUG_INTERVALO", 5)
FRAMES_DEBUG_MAX       = _env_int("FRAMES_DEBUG_MAX", 100)
Path(FRAMES_DEBUG_DIR).mkdir(parents=True, exist_ok=True)

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


# ── Thread de envio ao webhook de debug (todas as pedras identificadas) ────
# maxsize=1: guarda só o frame mais recente pendente de envio. Se o worker
# ainda não terminou de mandar o anterior quando um novo frame chega, o
# anterior é descartado (ver bloco de produção abaixo) em vez de enfileirar
# — evita acumular atraso quando o POST ao n8n demora.
fila_webhook_debug = queue.Queue(maxsize=1)


def worker_webhook_debug():
    print("📡 Módulo de debug (todas as pedras identificadas) inicializado.", flush=True)
    while True:
        item = fila_webhook_debug.get()
        if item is None:
            break

        jpg_bytes, nome_arquivo = item
        try:
            payload = {
                "filename": nome_arquivo,
                "mimetype": "image/jpeg",
                "image_base64": base64.b64encode(jpg_bytes).decode("utf-8"),
            }
            response = requests.post(URL_WEBHOOK_DEBUG, json=payload, timeout=10)
            if response.status_code != 200:
                print(f"⚠️  Webhook de debug retornou {response.status_code}: {response.text}", flush=True)
        except Exception as e:
            print(f"❌ Falha ao enviar frame de debug: {e}", flush=True)

        fila_webhook_debug.task_done()


if URL_WEBHOOK_DEBUG:
    threading.Thread(target=worker_webhook_debug, daemon=True).start()


# ── Detecção e rastreamento ────────────────────────────────────────────────
def ponto_dentro_roi(cx, cy, rx, ry, rw, rh) -> bool:
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def desenhar_deteccoes(frame, detections):
    frame_desenhado = frame.copy()
    cv2.rectangle(frame_desenhado, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
    for i in range(len(detections)):
        dx1, dy1, dx2, dy2 = map(int, detections.xyxy[i])
        cv2.rectangle(frame_desenhado, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
    return frame_desenhado


def salvar_frame_debug(frame, detections):
    frame_debug = desenhar_deteccoes(frame, detections)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    cv2.imwrite(str(Path(FRAMES_DEBUG_DIR) / f"frame_{ts}.jpg"), frame_debug)

    # Nomes com timestamp ordenam cronologicamente — remove os mais antigos
    # quando a pasta ultrapassa o limite.
    arquivos = sorted(Path(FRAMES_DEBUG_DIR).glob("*.jpg"))
    for antigo in arquivos[:-FRAMES_DEBUG_MAX] if len(arquivos) > FRAMES_DEBUG_MAX else []:
        antigo.unlink(missing_ok=True)


print(f"🚀 Carregando modelo {WEIGHTS} em device={DEVICE}...", flush=True)
model = YOLO(WEIGHTS)

smoother        = sv.DetectionsSmoother(length=5)
contagem_frames: dict[int, int] = {}
ids_salvos:      set[int]       = set()

cam     = USBCamera(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FRAMERATE)
largura = CAMERA_WIDTH
altura  = CAMERA_HEIGHT

frame_count            = 0
t_log                  = time.time()
t_ultimo_frame_debug   = 0.0
t_ultimo_webhook_debug = 0.0

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

        agora_ts = time.time()

        if (
            URL_WEBHOOK_DEBUG
            and ids_frame_atual
            and agora_ts - t_ultimo_webhook_debug >= INTERVALO_WEBHOOK_DEBUG
        ):
            frame_anotado = desenhar_deteccoes(frame, detections)
            ok_jpg, buffer = cv2.imencode(".jpg", frame_anotado)
            if ok_jpg:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # Descarta o frame pendente antigo (se o worker ainda não
                # enviou) para sempre priorizar o mais recente na fila.
                if fila_webhook_debug.full():
                    try:
                        fila_webhook_debug.get_nowait()
                    except queue.Empty:
                        pass
                fila_webhook_debug.put((buffer.tobytes(), f"frame_{ts}.jpg"))
                t_ultimo_webhook_debug = agora_ts

        ids_sumidos = set(contagem_frames.keys()) - ids_frame_atual
        for tid in ids_sumidos:
            del contagem_frames[tid]

        print(f"👁️  pedras detectadas no frame: {len(ids_frame_atual)}", flush=True)

        if agora_ts - t_ultimo_frame_debug >= FRAMES_DEBUG_INTERVALO:
            salvar_frame_debug(frame, detections)
            t_ultimo_frame_debug = agora_ts

        if frame_count % 300 == 0:
            agora = time.time()
            fps   = 300.0 / (agora - t_log + 1e-9)
            t_log = agora

            mem_mb   = memoria_rss_mb()
            temp_c   = temperatura_cpu_c()
            throttle = status_throttling()

            mem_str      = f"{mem_mb:.0f}MB" if mem_mb is not None else "?"
            temp_str     = f"{temp_c:.1f}°C" if temp_c is not None else "?"
            throttle_str = throttle if throttle is not None else "?"

            with lock_contadores:
                print(
                    f"❤️  frames={frame_count} | fps≈{fps:.1f} | "
                    f"salvas={fotos_salvas} | bloqueadas={fotos_bloqueadas} | "
                    f"fila={fila_processamento.qsize()} | "
                    f"ram={mem_str} | temp={temp_str} | throttle={throttle_str}",
                    flush=True,
                )

except KeyboardInterrupt:
    print("\n🛑 Encerrando inferência...", flush=True)
finally:
    fila_processamento.put(None)
    if URL_WEBHOOK_DEBUG:
        fila_webhook_debug.put(None)
    cam.release()
    print(
        f"✅ Pipeline encerrado. Salvas={fotos_salvas} Bloqueadas={fotos_bloqueadas}",
        flush=True,
    )
