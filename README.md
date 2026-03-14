# FLUX.2 Turbo RunPod Worker

RunPod Serverless worker for `fal/FLUX.2-dev-Turbo`, tuned for `RTX A5000 24 GB` deployments.

This worker uses the official low-VRAM FLUX.2 path from the Hugging Face / Black Forest Labs docs:

- quantized model repo: `diffusers/FLUX.2-dev-bnb-4bit`
- local 4-bit FLUX.2 text encoder
- `fal/FLUX.2-dev-Turbo` LoRA on top
- CPU offload enabled in the pipeline

## Why this setup

According to the official FLUX.2 docs:

- full FLUX.2 inference is `80 GB+` VRAM without memory-saving tricks
- the official 4-bit + remote text-encoder setup runs in about `18 GB` VRAM
- the official 4-bit + local 4-bit text-encoder setup runs in about `20 GB` VRAM

This project chooses the `~20 GB` official local-encoder path so Turbo runs on a `24 GB` A5000 without depending on a paused external Space.

## What the worker supports

- text-to-image
- single-image editing

By default, the A5000 profile allows one input image at a time. You can raise `MAX_INPUT_IMAGES`, but more reference images increase VRAM usage.

## Deployment

1. Build and push the Docker image:

```bash
docker build -t your-username/flux2-turbo-a5000-worker:latest .
docker push your-username/flux2-turbo-a5000-worker:latest
```

2. Create a RunPod Serverless template with that image.
3. Optionally set `HF_TOKEN` in the template environment for authenticated Hub access and better rate limits.
4. Attach a volume with enough free space for the quantized model cache.
5. Deploy the endpoint on an `RTX A5000 24 GB` worker.

The worker automatically uses `/runpod-volume/huggingface` when a RunPod volume is mounted there. Otherwise it falls back to `/tmp/huggingface`.

## Environment Variables

- `HF_TOKEN`: optional, but recommended for authenticated Hub downloads and better rate limits
- `QUANTIZED_MODEL_ID`: default `diffusers/FLUX.2-dev-bnb-4bit`
- `TURBO_LORA_ID`: default `fal/FLUX.2-dev-Turbo`
- `TURBO_LORA_WEIGHT`: default `flux.2-turbo-lora.safetensors`
- `MODEL_CACHE_DIR`: override the Hugging Face cache location
- `MODEL_DOWNLOAD_MIN_FREE_GB`: startup free-space threshold, default `50`
- `ENABLE_CPU_OFFLOAD`: default `1`; set to `0` to try full-GPU mode on the A5000
- `DEFAULT_WIDTH`: default output width, default `1024`
- `DEFAULT_HEIGHT`: default output height, default `1024`
- `MAX_IMAGE_SIDE`: max auto-sized output side, default `1024`
- `MAX_INPUT_IMAGES`: default `1` for the A5000 profile

## API Input

### Text-to-image

```json
{
  "input": {
    "prompt": "A chrome robot portrait in a dramatic studio setup",
    "num_inference_steps": 8,
    "guidance_scale": 2.5,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }
}
```

### Image-to-image

Use `image` as a base64 string, with or without a `data:image/...;base64,` prefix.

```json
{
  "input": {
    "prompt": "Transform this screenshot into a clean cinematic sci-fi matte painting while keeping the layout recognizable",
    "image": "<base64 PNG or JPEG>",
    "num_inference_steps": 8,
    "guidance_scale": 2.5,
    "seed": 42
  }
}
```

## Local Curl Example

```bash
IMAGE_B64=$(base64 -w 0 "/home/w/Pictures/2026-03-13-153742_hyprshot.png")

curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d "{\"input\":{\"prompt\":\"Transform this screenshot into a clean cinematic sci-fi matte painting while keeping the layout recognizable\",\"image\":\"${IMAGE_B64}\",\"num_inference_steps\":8,\"guidance_scale\":2.5,\"seed\":42}}"
```

## Response Shape

```json
{
  "image": "<base64 PNG>",
  "seed": 42,
  "mode": "image-to-image",
  "width": 1024,
  "height": 576,
  "num_input_images": 1
}
```
