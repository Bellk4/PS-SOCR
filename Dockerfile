FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
  apt-get install -y --no-install-recommends git ca-certificates && \
  rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY README.md ./README.md
COPY LICENSE ./LICENSE
COPY THIRD_PARTY_NOTICES.md ./THIRD_PARTY_NOTICES.md

RUN mkdir -p /app/models/hf_cache /app/models/hf_home

ARG TORCH_CHANNEL=cpu
RUN python -m pip install --upgrade pip && \
    if [ "$TORCH_CHANNEL" = "cpu" ]; then \
      python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    else \
      python -m pip install --upgrade --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL} torch torchvision || \
      python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    fi && \
    python -m pip install fastapi uvicorn websockets wsproto python-multipart pillow pypdfium2 accelerate python-dotenv auth0-server-python httpx && \
    python -m pip install git+https://github.com/huggingface/transformers.git

ENV HOST=0.0.0.0 \
    PORT=8000 \
    GLM_MODEL_CACHE=/app/models/hf_cache \
    HF_HOME=/app/models/hf_home \
    HF_HUB_CACHE=/app/models/hf_cache \
    TRANSFORMERS_CACHE=/app/models/hf_cache

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
