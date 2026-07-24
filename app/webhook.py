"""
Monitor da pasta pedras_grandes/ — versão Raspberry Pi.

Observa a pasta de saída e, a cada nova imagem, envia para o webhook do n8n
em Base64. Todo o tratamento de imagem/mensagem (comparação, formatação,
envio ao WhatsApp etc.) é feito pelo workflow do n8n — aqui só entregamos
a imagem. Toda configuração vem de variáveis de ambiente.

Controle de "já enviado" é feito pela presença do arquivo em CAMINHO_PASTA:
após o envio ter sucesso, o arquivo é movido para PASTA_ENVIADAS. Assim, se
o processo reiniciar (systemd, queda de energia etc.), ele não reenvia tudo
de novo — o que já foi mandado não está mais na pasta monitorada. A pasta de
enviados mantém no máximo ENVIADAS_MAX imagens (as mais antigas são
apagadas), pra não encher o disco rodando indefinidamente.
"""

import os
import base64
import time
from pathlib import Path

import requests

CAMINHO_PASTA         = os.getenv("PASTA_SAIDA", "pedras_grandes")
PASTA_ENVIADAS        = os.getenv("PASTA_ENVIADAS", "pedras_enviadas")
ENVIADAS_MAX          = int(os.getenv("ENVIADAS_MAX", "300"))
URL_WEBHOOK           = os.getenv("URL_WEBHOOK", "")
INTERVALO_VERIFICACAO = int(os.getenv("INTERVALO_VERIFICACAO", "5"))

if not URL_WEBHOOK:
    raise SystemExit(
        "❌ URL_WEBHOOK não definido. Configure no .env "
        "(ex.: URL_WEBHOOK=https://.../webhook/identificador)."
    )

os.makedirs(CAMINHO_PASTA, exist_ok=True)
os.makedirs(PASTA_ENVIADAS, exist_ok=True)


def _mover_para_enviadas(caminho_arquivo, nome_arquivo):
    destino = Path(PASTA_ENVIADAS) / nome_arquivo
    os.replace(caminho_arquivo, destino)

    arquivos = sorted(Path(PASTA_ENVIADAS).glob("*"), key=lambda p: p.stat().st_mtime)
    for antigo in arquivos[:-ENVIADAS_MAX] if len(arquivos) > ENVIADAS_MAX else []:
        antigo.unlink(missing_ok=True)


def enviar_para_n8n(caminho_arquivo, nome_arquivo):
    try:
        with open(caminho_arquivo, "rb") as f:
            string_base64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "filename": nome_arquivo,
            "mimetype": "image/jpeg",
            "image_base64": string_base64,
        }

        response = requests.post(URL_WEBHOOK, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"✅ [SUCESSO] {nome_arquivo} enviado ao n8n com sucesso!", flush=True)
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
            for arquivo in os.listdir(CAMINHO_PASTA):
                if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                caminho_completo = os.path.join(CAMINHO_PASTA, arquivo)
                time.sleep(0.5)
                if enviar_para_n8n(caminho_completo, arquivo):
                    _mover_para_enviadas(caminho_completo, arquivo)

            time.sleep(INTERVALO_VERIFICACAO)
    except KeyboardInterrupt:
        print("\n🛑 Monitoramento encerrado.", flush=True)


if __name__ == "__main__":
    main()
