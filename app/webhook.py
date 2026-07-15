"""
Monitor da pasta pedras_grandes/ — versão Raspberry Pi.

Observa a pasta de saída e, a cada nova imagem, envia para o webhook
(WhatsApp) em Base64. Toda configuração vem de variáveis de ambiente.
"""

import os
import base64
import time

import requests

CAMINHO_PASTA         = os.getenv("PASTA_SAIDA", "pedras_grandes")
URL_WEBHOOK           = os.getenv("URL_WEBHOOK", "")
INTERVALO_VERIFICACAO = int(os.getenv("INTERVALO_VERIFICACAO", "5"))
PHONE                 = os.getenv("PHONE", "5562982878127")
MENSAGEM              = os.getenv(
    "MENSAGEM", "Pedra grande detectada. A pedra é grande de fato?"
)
BACKOFF_BASE_S        = int(os.getenv("BACKOFF_BASE_S", "5"))
BACKOFF_MAXIMO_S      = int(os.getenv("BACKOFF_MAXIMO_S", "300"))

if not URL_WEBHOOK:
    raise SystemExit(
        "❌ URL_WEBHOOK não definido. Configure no .env "
        "(ex.: URL_WEBHOOK=https://.../webhook/identificador)."
    )

os.makedirs(CAMINHO_PASTA, exist_ok=True)
arquivos_enviados = set()

# Backoff exponencial por arquivo: sem isso, um arquivo que falha (ex.:
# webhook fora do ar) é retentado a cada INTERVALO_VERIFICACAO segundos
# para sempre, martelando rede/CPU do Pi indefinidamente durante uma
# indisponibilidade prolongada — o que já observamos coincidir com
# quedas da conexão RTSPS da câmera (mesma interface de rede do Pi).
tentativas_falhas  = {}  # nome_arquivo -> nº de falhas consecutivas
proxima_tentativa  = {}  # nome_arquivo -> timestamp da próxima tentativa permitida


def enviar_para_whatsapp(caminho_arquivo, nome_arquivo):
    try:
        with open(caminho_arquivo, "rb") as f:
            string_base64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "phone": PHONE,
            "message": MENSAGEM,
            "type": "send-button-list",
            "buttonList": {
                "image": f"data:image/jpeg;base64,{string_base64}",
                "buttons": [
                    {"id": "1", "label": "SIM"},
                    {"id": "2", "label": "NÃO"},
                ],
            },
        }

        response = requests.post(URL_WEBHOOK, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"✅ [SUCESSO] {nome_arquivo} enviado com sucesso!", flush=True)
            return True
        print(
            f"⚠️ [AVISO] Falha ao enviar {nome_arquivo}. "
            f"Status: {response.status_code} - {response.text}",
            flush=True,
        )
        return False
    except Exception as e:
        print(f"❌ [ERRO] Falha crítica ao processar {nome_arquivo}: {e}", flush=True)
        return False


def main():
    print(f"🔍 Monitorando {CAMINHO_PASTA} (intervalo {INTERVALO_VERIFICACAO}s)...", flush=True)
    try:
        while True:
            agora = time.time()
            for arquivo in os.listdir(CAMINHO_PASTA):
                if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if arquivo in arquivos_enviados:
                    continue
                if proxima_tentativa.get(arquivo, 0) > agora:
                    continue

                caminho_completo = os.path.join(CAMINHO_PASTA, arquivo)
                time.sleep(0.5)
                if enviar_para_whatsapp(caminho_completo, arquivo):
                    arquivos_enviados.add(arquivo)
                    tentativas_falhas.pop(arquivo, None)
                    proxima_tentativa.pop(arquivo, None)
                else:
                    falhas = tentativas_falhas.get(arquivo, 0) + 1
                    tentativas_falhas[arquivo] = falhas
                    atraso = min(BACKOFF_BASE_S * (2 ** falhas), BACKOFF_MAXIMO_S)
                    proxima_tentativa[arquivo] = time.time() + atraso
                    print(
                        f"⏳ Próxima tentativa de {arquivo} em {atraso:.0f}s "
                        f"(falha #{falhas})",
                        flush=True,
                    )

            time.sleep(INTERVALO_VERIFICACAO)
    except KeyboardInterrupt:
        print("\n🛑 Monitoramento encerrado.", flush=True)


if __name__ == "__main__":
    main()
