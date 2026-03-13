# Z-Image-Turbo RunPod Serverless

High-performance inference engine for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B Single-Stream DiT) hosted on RunPod Serverless.

## Features
- **Fast Inference**: Generates high-quality images in 8-10 steps.
- **RunPod Optimized**: Warm-start model loading and serverless handler.
- **Modern Stack**: Python, `diffusers`, and `uv` for package management.

## Deployment

1. Build and push the Docker image:
   ```bash
   docker build -t your-username/z-image-turbo-worker:latest .
   docker push your-username/z-username/z-image-turbo-worker:latest
   ```

2. Create a RunPod Serverless template using the image.
3. Deploy an endpoint with the template.

## Test Input

```json
{
  "input": {
    "prompt": "A cinematic portrait of a robotic explorer in a futuristic jungle, 8k resolution, highly detailed",
    "num_inference_steps": 9,
    "guidance_scale": 0.0,
    "height": 1024,
    "width": 1024
  }
}
```
