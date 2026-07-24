#!/usr/bin/env bash
set -e

if [ ! -f .env ]; then
    echo "❌ Arquivo .env não encontrado."
    echo "   Execute: cp .env.example .env && nano .env"
    exit 1
fi

source venv/bin/activate

# Exporta variáveis do .env para o ambiente
set -a
source .env
set +a

WEIGHTS_PATH="${WEIGHTS:-weights/best.pt}"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "❌ Modelo não encontrado: $WEIGHTS_PATH"
    echo "   Copie best.pt para a pasta weights/"
    exit 1
fi

mkdir -p "${PASTA_SAIDA:-pedras_grandes}" "${PASTA_ENVIADAS:-pedras_enviadas}"

echo "🚀 Iniciando britago-rasp..."
python3 app/inferencia.py &
PID_INF=$!

python3 app/webhook.py &
PID_HOOK=$!

trap "echo '🛑 Encerrando...'; kill $PID_INF $PID_HOOK 2>/dev/null; wait" INT TERM

wait $PID_INF $PID_HOOK
