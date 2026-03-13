# Use a high-performance CUDA base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install dependencies using uv
# We use --system to install into the main python env since we are in a container
RUN uv pip install --system \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --system \
    "git+https://github.com/huggingface/diffusers" \
    "transformers>=4.48.0" \
    "accelerate>=0.34.0" \
    "huggingface_hub>=0.34.0" \
    sentencepiece \
    runpod \
    pillow

# Copy the rest of the application
COPY main.py .

# Optional: Pre-download the model (adds about 12GB to the image)
# RUN python3 -c "import torch; from diffusers import ZImagePipeline; ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16)"

# Run the handler
CMD ["python3", "-u", "main.py"]
