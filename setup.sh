#!/usr/bin/env bash
# Executa no Raspberry Pi 4B com Raspberry Pi OS (Bullseye ou Bookworm).
# Necessário: câmera CSI habilitada via raspi-config > Interface Options > Camera

set -e

echo "=== [1/4] Instalando dependências do sistema ==="
sudo apt-get update
sudo apt-get install -y \
    python3-libcamera \
    libcamera-tools \
    python3-pip \
    python3-venv \
    ffmpeg \
    libatlas-base-dev

echo ""
echo "=== [2/4] Criando ambiente virtual ==="
# --system-site-packages expõe picamera2 (instalada via apt) dentro do venv
python3 -m venv venv --system-site-packages

echo ""
echo "=== [3/4] Instalando dependências Python ==="
source venv/bin/activate
pip install --upgrade pip
# picamera2 sem dependências opcionais que exigem compilação (PiDNG, python-prctl)
pip install picamera2 --no-deps
pip install piexif simplejpeg videodev2
# PyTorch CPU-only antes do ultralytics — evita baixar pacotes CUDA (>1.5GB inúteis no RPi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo ""
echo "=== [4/4] Preparando estrutura ==="
mkdir -p weights pedras_grandes

if [ ! -f .env ]; then
    cp .env.example .env
fi

chmod +x run.sh

echo ""
echo "=========================================="
echo " Setup concluído!"
echo "=========================================="
echo ""
echo " Próximos passos:"
echo "   1. Copie o modelo treinado:"
echo "        scp best.pt pi@<ip-do-rasp>:$(pwd)/weights/best.pt"
echo ""
echo "   2. Configure o arquivo .env:"
echo "        nano .env"
echo "      (ajuste URL_WEBHOOK, PHONE, ROI e parâmetros de câmera)"
echo ""
echo "   3. Inicie o sistema:"
echo "        ./run.sh"
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
