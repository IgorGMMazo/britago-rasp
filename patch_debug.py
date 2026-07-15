import re

caminho = "app/inferencia.py"
with open(caminho, "r", encoding="utf-8") as f:
    conteudo = f.read()

# 1. Config do modo debug — logo após a config de ROI
ancora_config = 'AREA_ROI = ROI_W * ROI_H'
novo_config = '''AREA_ROI = ROI_W * ROI_H

# ── Debug visual: salva cada frame com ROI + boxes desenhados ─────────────
DEBUG_DIR = os.getenv("DEBUG_DIR", "debug_frames")
DEBUG_LIMPEZA_SEGUNDOS = _env_int("DEBUG_LIMPEZA_SEGUNDOS", 120)
Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)'''
assert ancora_config in conteudo, "Âncora 1 não encontrada"
conteudo = conteudo.replace(ancora_config, novo_config, 1)

# 2. Thread de limpeza automática — logo após a thread de hashing começar
ancora_thread = 'thread_filtro = threading.Thread(target=worker_filtro_hashing, daemon=True)\nthread_filtro.start()'
novo_thread = ancora_thread + '''


def worker_limpeza_debug():
    while True:
        time.sleep(DEBUG_LIMPEZA_SEGUNDOS)
        agora = time.time()
        removidos = 0
        for f in Path(DEBUG_DIR).glob("*.jpg"):
            try:
                if agora - f.stat().st_mtime > DEBUG_LIMPEZA_SEGUNDOS:
                    f.unlink()
                    removidos += 1
            except FileNotFoundError:
                pass
        if removidos:
            print(f"🧹 Limpeza {DEBUG_DIR}: {removidos} arquivo(s) removido(s)", flush=True)


thread_limpeza_debug = threading.Thread(target=worker_limpeza_debug, daemon=True)
thread_limpeza_debug.start()'''
assert ancora_thread in conteudo, "Âncora 2 não encontrada"
conteudo = conteudo.replace(ancora_thread, novo_thread, 1)

# 3. Salvar frame com ROI + boxes desenhados — logo antes do log de heartbeat
ancora_save = "        if frame_count % 300 == 0:"
novo_save = '''        frame_debug = frame.copy()
        cv2.rectangle(frame_debug, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
        for i in range(len(detections)):
            dx1, dy1, dx2, dy2 = map(int, detections.xyxy[i])
            cv2.rectangle(frame_debug, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
        ts_debug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cv2.imwrite(str(Path(DEBUG_DIR) / f"frame_{ts_debug}.jpg"), frame_debug)

        if frame_count % 300 == 0:'''
assert ancora_save in conteudo, "Âncora 3 não encontrada"
conteudo = conteudo.replace(ancora_save, novo_save, 1)

with open(caminho, "w", encoding="utf-8") as f:
    f.write(conteudo)

print("✅ Patch aplicado com sucesso.")
