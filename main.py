import base64
import io
import os
import shutil
from pathlib import Path

import runpod
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from PIL import Image
from transformers import Mistral3ForConditionalGeneration

QUANTIZED_MODEL_ID = os.getenv("QUANTIZED_MODEL_ID", "diffusers/FLUX.2-dev-bnb-4bit")
TURBO_LORA_ID = os.getenv("TURBO_LORA_ID", "fal/FLUX.2-dev-Turbo")
TURBO_LORA_WEIGHT = os.getenv("TURBO_LORA_WEIGHT", "flux.2-turbo-lora.safetensors")
CACHE_MIN_FREE_GB = float(os.getenv("MODEL_DOWNLOAD_MIN_FREE_GB", "50"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))
MIN_IMAGE_SIDE = int(os.getenv("MIN_IMAGE_SIDE", "256"))
MAX_INPUT_IMAGES = int(os.getenv("MAX_INPUT_IMAGES", "1"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "2.5"))
DEFAULT_INFERENCE_STEPS = int(os.getenv("DEFAULT_INFERENCE_STEPS", "8"))
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

pipe = None


def get_hf_token():
    return os.getenv("HF_TOKEN")


def use_cpu_offload():
    value = os.getenv("ENABLE_CPU_OFFLOAD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def resolve_cache_dir():
    configured_cache_dir = os.getenv("MODEL_CACHE_DIR") or os.getenv("HF_HOME")
    if configured_cache_dir:
        cache_dir = configured_cache_dir
    elif os.path.isdir("/runpod-volume") and os.path.ismount("/runpod-volume"):
        cache_dir = "/runpod-volume/huggingface"
    else:
        cache_dir = "/tmp/huggingface"

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir) / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return cache_dir


def ensure_cache_capacity(cache_dir):
    free_bytes = shutil.disk_usage(cache_dir).free
    free_gb = free_bytes / (1024**3)
    print(f"Using Hugging Face cache dir: {cache_dir} ({free_gb:.2f} GB free)")

    if free_gb < CACHE_MIN_FREE_GB:
        raise RuntimeError(
            f"Insufficient disk space to download {QUANTIZED_MODEL_ID}. "
            f"Found {free_gb:.2f} GB free in {cache_dir}, but this setup expects about "
            f"{CACHE_MIN_FREE_GB:.0f}+ GB for the initial quantized model and LoRA download. "
            "Attach a larger RunPod volume or set MODEL_CACHE_DIR/HF_HOME to a path with more space."
        )


def round_dimension(value):
    value = int(value)
    value = max(MIN_IMAGE_SIDE, min(MAX_IMAGE_SIDE, value))
    return max(8, round(value / 8) * 8)


def resolve_dimensions(job_input, input_images):
    width = job_input.get("width")
    height = job_input.get("height")

    if width is not None and height is not None:
        return round_dimension(width), round_dimension(height)

    if input_images:
        image_width, image_height = input_images[0].size
        aspect_ratio = image_width / image_height

        if aspect_ratio >= 1:
            width = MAX_IMAGE_SIDE
            height = int(MAX_IMAGE_SIDE / aspect_ratio)
        else:
            height = MAX_IMAGE_SIDE
            width = int(MAX_IMAGE_SIDE * aspect_ratio)

        return round_dimension(width), round_dimension(height)

    return round_dimension(width or DEFAULT_WIDTH), round_dimension(
        height or DEFAULT_HEIGHT
    )


def decode_image(encoded_image):
    if not encoded_image:
        raise ValueError("Empty image payload")

    if encoded_image.startswith("data:"):
        encoded_image = encoded_image.split(",", 1)[1]

    image_bytes = base64.b64decode(encoded_image)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_input_images(job_input):
    images = []

    if job_input.get("image"):
        images.append(decode_image(job_input["image"]))

    for encoded_image in job_input.get("images", []):
        images.append(decode_image(encoded_image))

    if len(images) > MAX_INPUT_IMAGES:
        raise ValueError(
            f"Too many input images. Maximum supported is {MAX_INPUT_IMAGES}."
        )

    return images


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_generator(seed):
    if seed is None:
        return None, None

    seed = int(seed)
    return torch.Generator(device="cuda").manual_seed(seed), seed


def load_model():
    global pipe
    if pipe is not None:
        return pipe

    token = get_hf_token()
    cache_dir = resolve_cache_dir()
    ensure_cache_capacity(cache_dir)
    cpu_offload = use_cpu_offload()
    module_device_map = "cpu" if cpu_offload else {"": 0}

    print(f"Loading quantized text encoder: {QUANTIZED_MODEL_ID}")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        QUANTIZED_MODEL_ID,
        subfolder="text_encoder",
        cache_dir=cache_dir,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=module_device_map,
    )

    print(f"Loading quantized transformer: {QUANTIZED_MODEL_ID}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        QUANTIZED_MODEL_ID,
        subfolder="transformer",
        cache_dir=cache_dir,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=module_device_map,
    )

    pipe = Flux2Pipeline.from_pretrained(
        QUANTIZED_MODEL_ID,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=cache_dir,
        token=token,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading Turbo LoRA: {TURBO_LORA_ID}")
    pipe.load_lora_weights(
        TURBO_LORA_ID,
        cache_dir=cache_dir,
        token=token,
        weight_name=TURBO_LORA_WEIGHT,
    )

    if cpu_offload:
        pipe.enable_model_cpu_offload()
        print("Quantized FLUX.2 Turbo loaded with CPU offload enabled.")
    else:
        pipe.to("cuda")
        print("Quantized FLUX.2 Turbo loaded fully on GPU without CPU offload.")

    return pipe


def handler(job):
    model = load_model()
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")

    if not prompt:
        return {"error": "No prompt provided"}

    input_images = load_input_images(job_input)
    width, height = resolve_dimensions(job_input, input_images)
    guidance_scale = float(job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    num_inference_steps = int(
        job_input.get("num_inference_steps", DEFAULT_INFERENCE_STEPS)
    )
    generator, seed = build_generator(job_input.get("seed"))

    pipe_kwargs = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
    }

    if input_images:
        pipe_kwargs["image"] = input_images

    sigmas = job_input.get("sigmas")
    if sigmas:
        pipe_kwargs["sigmas"] = [float(value) for value in sigmas]
    elif num_inference_steps == len(TURBO_SIGMAS):
        pipe_kwargs["sigmas"] = TURBO_SIGMAS

    with torch.inference_mode():
        result = model(**pipe_kwargs).images[0]

    return {
        "image": encode_image(result),
        "seed": seed,
        "mode": "image-to-image" if input_images else "text-to-image",
        "width": width,
        "height": height,
        "num_input_images": len(input_images),
    }


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
