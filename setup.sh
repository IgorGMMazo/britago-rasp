#!/usr/bin/env bash
# Executa no Raspberry Pi 4B com Ubuntu Server 24.04 (noble) — câmera USB (UVC).
# NÃO usar em Raspberry Pi OS: este script não instala libcamera/picamera2,
# pois a câmera do projeto é USB (driver UVC nativo do kernel, sem dependências
# específicas de fabricante).

set -e

echo "=== [1/4] Instalando dependências do sistema ==="
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    libatlas-base-dev \
    v4l-utils \
    chafa

echo ""
echo "=== [2/4] Criando ambiente virtual ==="
python3 -m venv venv

echo ""
echo "=== [3/4] Instalando dependências Python ==="
source venv/bin/activate
pip install --upgrade pip

# PyTorch: em aarch64 (ARM 64-bit), o wheel oficial do PyPI já é CPU-only
# nativamente — não existe build CUDA pra essa arquitetura em hardware comum
# como o Pi 4B. Por isso não precisamos apontar pra um índice especial: o pip
# já resolve sozinho o wheel correto (torch-*-cp312-cp312-manylinux_*_aarch64.whl).
pip install torch torchvision

pip install -r requirements.txt

echo ""
echo "=== [4/4] Preparando estrutura ==="
mkdir -p weights pedras_grandes

if [ ! -f .env ]; then
    cp .env.example .env
fi

chmod +x run.sh
chmod +x scripts/ver_debug.sh 2>/dev/null || true

echo ""
echo "=========================================="
echo " Setup concluído!"
echo "=========================================="
echo ""
echo " Confirme que a câmera USB foi reconhecida pelo kernel:"
echo "   v4l2-ctl --list-devices"
echo "   (deve listar algo como /dev/video0)"
echo ""
echo " Próximos passos:"
echo "   1. Confirme que o modelo treinado está em weights/best.pt"
echo "        (se precisar copiar do seu notebook:"
echo "         scp best.pt igormazo@<ip-do-rasp>:$(pwd)/weights/best.pt)"
echo ""
echo "   2. Configure o arquivo .env:"
echo "        nano .env"
echo "      (ajuste URL_WEBHOOK, PHONE, ROI, CAMERA_INDEX e parâmetros de câmera)"
echo ""
echo "   3. Inicie o sistema:"
echo "        ./run.sh"
echo ""
echo "   Para conferir ao vivo o que a câmera/ROI estão vendo, direto por SSH:"
echo "        ./scripts/ver_debug.sh --watch"
echo ""
echo " Para iniciar automaticamente no boot:"
echo "   sudo bash -c \"cat > /etc/systemd/system/britago.service << EOF"
echo "   [Unit]"
echo "   Description=Britago Deteccao de Pedras"
echo "   After=network.target"
echo ""
echo "   [Service]"
echo "   Type=simple"
echo "   User=$USER"
echo "   WorkingDirectory=$(pwd)"
echo "   ExecStart=$(pwd)/run.sh"
echo "   Restart=always"
echo "   RestartSec=10"
echo ""
echo "   [Install]"
echo "   WantedBy=multi-user.target"
echo "   EOF\""
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable --now britago"