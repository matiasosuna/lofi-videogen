"""FastAPI video generation server — runs on AWS GPU instance."""
import os
import uuid
import threading
import time
import subprocess

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Lofi Video Gen Server")

WORK_DIR = "/tmp/videogen"
MODELS_DIR = "/opt/models"
os.makedirs(WORK_DIR, exist_ok=True)

# ─── Task tracking ────────────────────────────────────────────────────────

_tasks = {}
_lock = threading.Lock()


def _get_task(task_id):
    with _lock:
        return _tasks.get(task_id, {}).copy()


def _update_task(task_id, **kwargs):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].update(kwargs)


# ─── Model registry ──────────────────────────────────────────────────────

MODELS = {
    "wan2.2": {
        "name": "Wan 2.2",
        "repo": "Wan-AI/Wan2.2-T2V-14B",
        "repo_i2v": "Wan-AI/Wan2.2-I2V-14B",
        "type": "wan",
        "vram": "24GB",
        "description": "Fast, good quality. Best value.",
    },
    "hunyuan": {
        "name": "HunyuanVideo",
        "repo": "tencent/HunyuanVideo",
        "type": "hunyuan",
        "vram": "48GB",
        "description": "Best quality. Cinematic realism.",
    },
    "skyreels": {
        "name": "SkyReels V1",
        "repo": "Skywork/SkyReels-V1-Hunyuan-I2V",
        "type": "skyreels",
        "vram": "48GB",
        "description": "Cinematic. Best for professional content.",
    },
}

_loaded_model = None
_loaded_model_name = None
_model_lock = threading.Lock()


# ─── Model loading ────────────────────────────────────────────────────────

def _load_wan(mode="t2v"):
    """Load Wan 2.2 model."""
    import diffusers
    if mode == "i2v":
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-14B",
            torch_dtype=torch.float16,
        )
    else:
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-T2V-14B",
            torch_dtype=torch.float16,
        )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe


def _load_hunyuan():
    """Load HunyuanVideo model."""
    from diffusers import HunyuanVideoPipeline
    pipe = HunyuanVideoPipeline.from_pretrained(
        "tencent/HunyuanVideo",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe


def _load_skyreels():
    """Load SkyReels V1 model."""
    from diffusers import HunyuanVideoImageToVideoPipeline
    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
        "Skywork/SkyReels-V1-Hunyuan-I2V",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe


def _get_model(model_name, mode="t2v"):
    """Get or load a model. Unloads previous model if different."""
    global _loaded_model, _loaded_model_name

    cache_key = f"{model_name}_{mode}"
    with _model_lock:
        if _loaded_model_name == cache_key:
            return _loaded_model

        # Unload previous
        if _loaded_model is not None:
            del _loaded_model
            torch.cuda.empty_cache()
            _loaded_model = None
            _loaded_model_name = None

        if model_name == "wan2.2":
            _loaded_model = _load_wan(mode)
        elif model_name == "hunyuan":
            _loaded_model = _load_hunyuan()
        elif model_name == "skyreels":
            _loaded_model = _load_skyreels()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        _loaded_model_name = cache_key
        return _loaded_model


# ─── Generation workers ──────────────────────────────────────────────────

def _generate_worker(task_id, model_name, prompt, negative_prompt,
                     image_path, num_frames, width, height, steps, guidance, seed):
    try:
        mode = "i2v" if image_path else "t2v"
        _update_task(task_id, status="loading", message=f"Loading {model_name}...", progress=10)

        pipe = _get_model(model_name, mode)

        _update_task(task_id, status="generating", message="Generating video...", progress=20)

        generator = torch.manual_seed(seed) if seed >= 0 else None

        kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }

        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        if generator:
            kwargs["generator"] = generator

        if image_path and mode == "i2v":
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            kwargs["image"] = img

        # Generate
        output = pipe(**kwargs)

        _update_task(task_id, status="encoding", message="Encoding video...", progress=85)

        # Export
        from diffusers.utils import export_to_video
        out_filename = f"{task_id}.mp4"
        out_path = os.path.join(WORK_DIR, out_filename)
        export_to_video(output.frames[0], out_path, fps=16)

        # Get file size
        file_size = os.path.getsize(out_path)

        _update_task(
            task_id,
            status="done",
            message="Video generated!",
            progress=100,
            result={
                "filename": out_filename,
                "file_size": file_size,
                "duration": round(num_frames / 16, 1),
                "resolution": f"{width}x{height}",
                "model": model_name,
            },
        )

    except Exception as e:
        _update_task(task_id, status="error", message=str(e), progress=0, error=str(e))


# ─── API Routes ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3) if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "loaded_model": _loaded_model_name,
    }


@app.get("/models")
def list_models():
    return {"models": MODELS}


class GenerateRequest(BaseModel):
    model: str = "wan2.2"
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 81  # ~5 seconds at 16fps
    width: int = 832
    height: int = 480
    steps: int = 30
    guidance: float = 7.5
    seed: int = -1


@app.post("/generate")
def generate_text2video(req: GenerateRequest):
    if req.model not in MODELS:
        return JSONResponse({"error": f"Unknown model: {req.model}"}, 400)

    task_id = str(uuid.uuid4())
    with _lock:
        _tasks[task_id] = {
            "progress": 0,
            "status": "queued",
            "message": "Queued...",
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_generate_worker,
        args=(task_id, req.model, req.prompt, req.negative_prompt,
              None, req.num_frames, req.width, req.height,
              req.steps, req.guidance, req.seed),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id}


@app.post("/generate-from-image")
async def generate_img2video(
    file: UploadFile = File(...),
    model: str = Form("wan2.2"),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    num_frames: int = Form(81),
    width: int = Form(832),
    height: int = Form(480),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    seed: int = Form(-1),
):
    if model not in MODELS:
        return JSONResponse({"error": f"Unknown model: {model}"}, 400)

    # Save uploaded image
    task_id = str(uuid.uuid4())
    img_path = os.path.join(WORK_DIR, f"{task_id}_input.png")
    with open(img_path, "wb") as f:
        content = await file.read()
        f.write(content)

    with _lock:
        _tasks[task_id] = {
            "progress": 0,
            "status": "queued",
            "message": "Queued...",
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_generate_worker,
        args=(task_id, model, prompt, negative_prompt,
              img_path, num_frames, width, height,
              steps, guidance, seed),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task = _get_task(task_id)
    if not task:
        return JSONResponse({"error": "Unknown task"}, 404)
    return task


@app.get("/download/{filename}")
def download_video(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, 400)
    filepath = os.path.join(WORK_DIR, filename)
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "File not found"}, 404)
    return FileResponse(filepath, media_type="video/mp4", filename=filename)


@app.post("/shutdown")
def shutdown():
    """Shutdown the instance after a short delay."""
    def _shutdown():
        time.sleep(3)
        os.system("sudo shutdown -h now")
    threading.Thread(target=_shutdown, daemon=True).start()
    return {"status": "shutting_down"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
