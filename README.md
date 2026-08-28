# Short Engine

Motor local, CLI-first, para transformar vídeos e podcasts em cortes curtos. A
arquitetura prioriza Apple Silicon: MLX Whisper faz a transcrição, PySceneDetect
e Silero VAD formam a timeline, Gemini avalia texto e frames, YOLO26 rastreia o
assunto dominante em MPS e FFmpeg renderiza o resultado.

As legendas usam um preset bold/karaoke para Shorts: blocos de quatro palavras,
contorno de alto contraste e destaque amarelo na palavra ativa.

## Preparação

Requer macOS Apple Silicon, Python 3.12+, `uv` e um FFmpeg compilado com
`libass`. No Homebrew, essa capacidade está na fórmula keg-only `ffmpeg-full`.

```bash
brew install ffmpeg-full
export PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH"
uv sync --all-extras --dev
cp .env.example .env
# preencha GEMINI_API_KEY em .env
uv run short-engine doctor
```

O checkpoint MLX e o modelo YOLO são baixados na primeira execução. Modelos
ficam no cache do usuário; segredos e pesos nunca são gravados no Git.

## Uso

```bash
# Pipeline completo
uv run short-engine run video.mp4 --clips 3 --language pt

# YouTube autenticado pelo perfil local do Chrome
uv run short-engine run 'https://youtu.be/ID' \
  --cookies-from-browser 'chrome:Profile 3' --clips 3 --language pt

# Analisa/rankeia agora e renderiza depois, sem repetir MLX ou Gemini
uv run short-engine analyze video.mp4 --clips 3 --language pt
uv run short-engine render output/RUN_ID/manifest.json --candidate CANDIDATE_ID

uv run short-engine inspect output/RUN_ID/manifest.json
```

Cada execução possui manifesto atômico e retomável em `output/<run-id>/`. Uma
mudança de modelo ou configuração invalida apenas as etapas dependentes.

## Desenvolvimento

```bash
make check
make test
```

A especificação e os critérios de aceite estão em [SPEC.md](SPEC.md).
