import runpod
import torch
from diffusers import ZImagePipeline
from PIL import Image
import io
import base64
import os

# Global variable to hold the pipeline for warm starts
pipe = None

def load_model():
    global pipe
    if pipe is None:
        model_id = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
        print(f"Loading model: {model_id}...")
        
        # Load the pipeline with bfloat16 for efficiency
        # Z-Image-Turbo requires recent diffusers version
        pipe = ZImagePipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        pipe.to("cuda")
        print("Model loaded successfully.")

def handler(job):
    """
    RunPod handler function.
    """
    job_input = job["input"]
    prompt = job_input.get("prompt")
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("num_inference_steps", 8)
    guidance_scale = job_input.get("guidance_scale", 0.0) 
    seed = job_input.get("seed", None)

    if not prompt:
        return {"error": "No prompt provided"}

    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    # Generate image
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        image = output.images[0]

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str}

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
