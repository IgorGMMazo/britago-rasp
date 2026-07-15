#!/usr/bin/env bash
# Mostra o frame de debug mais recente direto no terminal SSH (sem GUI).
#
# Como funciona: o inferencia.py salva em DEBUG_DIR (padrão: debug_frames/)
# os últimos DEBUG_MAX_FRAMES frames com o ROI e as detecções desenhados.
# Este script pega o mais recente e usa o 'chafa' para desenhá-lo como
# arte ANSI/unicode no próprio terminal — não precisa de monitor nem X11.
#
# Instalar o chafa (uma vez só):
#   sudo apt install -y chafa
#
# Uso:
#   ./scripts/ver_debug.sh            # mostra o frame mais recente e sai
#   ./scripts/ver_debug.sh --watch    # atualiza a cada 2s (Ctrl+C para sair)

set -e

DEBUG_DIR="${DEBUG_DIR:-debug_frames}"

if ! command -v chafa >/dev/null 2>&1; then
    echo "❌ 'chafa' não está instalado. Rode: sudo apt install -y chafa"
    echo ""
    echo "Alternativa sem instalar nada: sirva a pasta por HTTP e veja"
    echo "pelo navegador do seu computador:"
    echo "  cd $DEBUG_DIR && python3 -m http.server 8000"
    echo "  # depois abra http://<ip-do-rasp>:8000/ no navegador"
    exit 1
fi

frame_mais_recente() {
    ls -t "$DEBUG_DIR"/*.jpg 2>/dev/null | head -n 1
}

mostrar_um() {
    local f
    f=$(frame_mais_recente)
    if [ -z "$f" ]; then
        echo "Nenhum frame de debug em '$DEBUG_DIR/' ainda (o pipeline está rodando?)"
        return 1
    fi
    echo "Frame: $(basename "$f")  |  $(date -r "$f" '+%H:%M:%S')  |  total no buffer: $(ls "$DEBUG_DIR"/*.jpg 2>/dev/null | wc -l)"
    chafa --symbols=all "$f"
}

if [ "$1" = "--watch" ]; then
    while true; do
        clear
        mostrar_um || true
        sleep 2
    done
else
    mostrar_um
fi
