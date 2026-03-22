FROM vllm/vllm-openai:nightly

# git is required for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install transformers from main branch (required by GLM-OCR, not yet on PyPI)
RUN pip uninstall -y transformers || true \
    && pip install -U git+https://github.com/huggingface/transformers.git

# Install RunPod SDK for serverless handler
RUN pip install runpod

# Bake model weights into the image to eliminate cold start downloads (~2 GB)
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

# Copy handler
COPY src/handler.py /handler.py

# RunPod handler starts vLLM internally on port 8000
CMD ["python3", "-u", "/handler.py"]
