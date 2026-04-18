FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG HF_TOKEN=""

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=30
ENV HF_XET_HIGH_PERFORMANCE=1
ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV APP_PORT=80
ENV PORT=80
ENV PORT_HEALTH=80
ENV VLLM_HOST=http://127.0.0.1:8080
ENV GLMOCR_CONFIG_PATH=/app/glmocr.config.yaml
ENV GLMOCR_LAYOUT_DEVICE=cpu

# Install Python 3.10 (ships with Ubuntu 22.04) and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 80

# Install SGLang (alternative inference engine for faster cold starts)
RUN pip3 install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Install transformers from main branch (required by GLM-OCR, needs >=5.0.0)
RUN pip3 install git+https://github.com/huggingface/transformers.git

# Install GLM-OCR selfhosted stack plus HTTP service dependencies
RUN pip3 install "glmocr[selfhosted,server]" fastapi uvicorn requests pypdf pypdfium2

# Bake OCR and layout model weights into the image to reduce boot time.
RUN python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
snapshot_download("zai-org/GLM-OCR", token=token)
snapshot_download("PaddlePaddle/PP-DocLayoutV3_safetensors", token=token)
PY

# Copy app files
WORKDIR /app
COPY service.py /app/service.py
COPY prompts.py /app/prompts.py
COPY glmocr.config.yaml /app/glmocr.config.yaml
COPY benchmark_datev.py /app/benchmark_datev.py

CMD ["python3", "-u", "/app/service.py"]
