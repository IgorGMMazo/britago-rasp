"""
Monitor da pasta pedras_grandes/ — versão Raspberry Pi.

Observa a pasta de saída e, a cada nova pedra detectada, envia para o webhook
do n8n em uma única requisição contendo o recorte (usado para comparação de
hash) e o frame completo, ambos em Base64, além da confiança da detecção.
Todo o tratamento de imagem/mensagem (comparação, formatação, envio ao
WhatsApp etc.) é feito pelo workflow do n8n — aqui só entregamos os dados.
Toda configuração vem de variáveis de ambiente.

Cada pedra salva por app/inferencia.py gera três arquivos com o mesmo prefixo:
"<prefixo>_crop.jpg", "<prefixo>_full.jpg" e "<prefixo>_meta.json" (este
último com os nomes dos dois arquivos de imagem e a confiança da detecção).
O meta.json é o gatilho: só disparamos o envio quando ele existe, garantindo
que as duas imagens já estejam gravadas no disco.

Controle de "já enviado" é feito pela presença dos arquivos em CAMINHO_PASTA:
após o envio ter sucesso, os três arquivos são movidos para PASTA_ENVIADAS.
Assim, se o processo reiniciar (systemd, queda de energia etc.), ele não
reenvia tudo de novo — o que já foi mandado não está mais na pasta
monitorada. A pasta de enviados mantém no máximo ENVIADAS_MAX arquivos (os
mais antigos são apagados), pra não encher o disco rodando indefinidamente.
"""

import os
import json
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


def _limitar_enviadas():
    arquivos = sorted(Path(PASTA_ENVIADAS).glob("*"), key=lambda p: p.stat().st_mtime)
    for antigo in arquivos[:-ENVIADAS_MAX] if len(arquivos) > ENVIADAS_MAX else []:
        antigo.unlink(missing_ok=True)


def _base64_arquivo(caminho):
    with open(caminho, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def enviar_para_n8n(caminho_meta, nome_meta):
    pasta = Path(CAMINHO_PASTA)
    try:
        with open(caminho_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        caminho_crop = pasta / meta["arquivo_crop"]
        caminho_full = pasta / meta["arquivo_full"]
        if not caminho_crop.exists() or not caminho_full.exists():
            # Ainda não terminaram de ser gravadas em disco; tenta de novo no próximo ciclo.
            return False

        payload = {
            "filename_crop": meta["arquivo_crop"],
            "filename_full": meta["arquivo_full"],
            "mimetype": "image/jpeg",
            "confianca": meta.get("confianca"),
            "image_base64_crop": _base64_arquivo(caminho_crop),
            "image_base64_full": _base64_arquivo(caminho_full),
        }

        response = requests.post(URL_WEBHOOK, json=payload, timeout=30)
        if response.status_code == 200:
            print(
                f"✅ [SUCESSO] {meta['arquivo_crop']} + {meta['arquivo_full']} enviados ao n8n com sucesso!",
                flush=True,
            )
            _mover_para_enviadas(caminho_crop, meta["arquivo_crop"])
            _mover_para_enviadas(caminho_full, meta["arquivo_full"])
            _mover_para_enviadas(caminho_meta, nome_meta)
            _limitar_enviadas()
            return True

        print(
            f"⚠️ [AVISO] Falha ao enviar {nome_meta}. "
            f"Status: {response.status_code} - {response.text}",
            flush=True,
        )
        return False
    except Exception as e:
        print(f"❌ [ERRO] Falha crítica ao processar {nome_meta}: {e}", flush=True)
        return False


def main():
    print(f"🔍 Monitorando {CAMINHO_PASTA} (intervalo {INTERVALO_VERIFICACAO}s)...", flush=True)
    try:
        while True:
            for arquivo in os.listdir(CAMINHO_PASTA):
                if not arquivo.lower().endswith(".json"):
                    continue

                caminho_completo = os.path.join(CAMINHO_PASTA, arquivo)
                time.sleep(0.5)
                enviar_para_n8n(caminho_completo, arquivo)

            time.sleep(INTERVALO_VERIFICACAO)
    except KeyboardInterrupt:
        print("\n🛑 Monitoramento encerrado.", flush=True)


if __name__ == "__main__":
    main()
