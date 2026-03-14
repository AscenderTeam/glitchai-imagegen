# CUDA runtime base for quantized FLUX.2 inference
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV HF_HUB_DISABLE_XET=1
ENV PYTHONUNBUFFERED=1

# Install PyTorch from the CUDA wheel index, then the rest from PyPI
RUN uv pip install --system \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --system \
    "bitsandbytes>=0.46.0" \
    "git+https://github.com/huggingface/diffusers" \
    "transformers>=4.48.0" \
    "accelerate>=0.34.0" \
    "huggingface_hub>=0.34.0" \
    "sentencepiece>=0.2.0" \
    "runpod>=1.7.0" \
    "pillow>=10.0.0"

COPY main.py .

CMD ["python3", "-u", "main.py"]
