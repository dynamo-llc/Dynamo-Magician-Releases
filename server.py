"""
Dynamo Magician – FastAPI backend
Wraps the WanI2V_PreQuant generation pipeline and exposes:
  POST /generate        – start a generation job
  GET  /stream/{job_id} – SSE stream of progress events
  GET  /output/{job_id} – download the produced MP4
  GET  /health          – liveness check
"""

import asyncio
import base64
import faulthandler
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from generate_prequant import WanI2V_PreQuant, save_video

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Dynamo Magician API")

# Enable faulthandler so C-level crashes (CUDA segfault, illegal instruction)
# write a traceback to crash_fault.txt instead of dying silently.
_fault_log = open(Path(__file__).parent / "crash_fault.txt", "w", buffering=1)
faulthandler.enable(file=_fault_log)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # React dev server on any port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built React app from dm-web-test/dist (optional – for production)
_dist = Path(__file__).parent / "dm-web-test" / "dist"
if _dist.exists():
    app.mount("/app", StaticFiles(directory=str(_dist), html=True), name="spa")

# ── In-memory job store ────────────────────────────────────────────────────────
# job_id → { "events": [str, ...], "done": bool, "output": path|None, "error": str|None }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Single shared pipeline (guards against concurrent loads)
_pipeline: Optional[WanI2V_PreQuant] = None
_pipeline_lock = threading.Lock()


# ── Request schema ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    image_b64: str          # base64-encoded image (no data-URI prefix)
    prompt: str
    width: int = 832
    height: int = 480
    steps: int = 40
    guidance: float = 5.0
    frames: int = 81
    seed: int = -1
    vram_mode: str = "low"  # "low" or "high"
    action_path: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def _push_event(job_id: str, event_type: str, data: dict) -> None:
    payload = json.dumps({"type": event_type, **data})
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["events"].append(payload)


def _generation_thread(job_id: str, req: GenerateRequest) -> None:
    global _pipeline

    try:
        _push_event(job_id, "status", {"message": "Decoding source image…"})

        # Decode base64 → PIL image
        raw = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        _push_event(job_id, "status", {"message": "Initialising model stack…"})

        with _pipeline_lock:
            if _pipeline is None or _pipeline.vram_mode != req.vram_mode:
                if _pipeline is not None:
                    _pipeline = None
                    torch.cuda.empty_cache()
                _pipeline = WanI2V_PreQuant(
                    checkpoint_dir=str(Path(__file__).parent),
                    vram_mode=req.vram_mode,
                )

        _push_event(job_id, "status", {"message": "Inference in progress…"})

        def progress_cb(current: int, total: int) -> None:
            _push_event(job_id, "progress", {"current": current, "total": total})

        video = _pipeline.generate(
            input_prompt=req.prompt,
            img=image,
            action_path=req.action_path,
            max_area=req.height * req.width,
            frame_num=req.frames,
            sampling_steps=req.steps,
            guide_scale=req.guidance,
            seed=req.seed,
            progress_callback=progress_cb,
        )

        _push_event(job_id, "status", {"message": "Encoding output…"})
        output_path = str(Path(__file__).parent / f"output_{job_id[:8]}.mp4")
        save_video(video, output_path)

        with _jobs_lock:
            _jobs[job_id]["output"] = output_path

        _push_event(job_id, "complete", {"output": f"/output/{job_id}"})

    except torch.cuda.OutOfMemoryError:
        msg = "CUDA out of memory – lower resolution or use low-VRAM mode"
        _push_event(job_id, "error", {"message": msg})
        with _jobs_lock:
            _jobs[job_id]["error"] = msg
        with _pipeline_lock:
            _pipeline = None
            torch.cuda.empty_cache()

    except BaseException as exc:
        # Catches Exception subclasses (RuntimeError, etc.) AND  SystemExit /
        # KeyboardInterrupt so that every failure writes crash_log.txt.
        import traceback
        tb = traceback.format_exc()
        msg = str(exc) or type(exc).__name__
        _push_event(job_id, "error", {"message": msg})
        with _jobs_lock:
            _jobs[job_id]["error"] = msg
        try:
            with open("crash_log.txt", "w") as fh:
                fh.write(tb)
        except Exception:
            pass
        # Reset the pipeline so the next request rebuilds it cleanly
        with _pipeline_lock:
            _pipeline = None
            torch.cuda.empty_cache()
        # Re-raise process-control exceptions (Ctrl-C, sys.exit) so the
        # server shuts down normally.
        if not isinstance(exc, Exception):
            raise

    finally:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["done"] = True


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")


@app.get("/health")
def health():
    return {"status": "online"}


@app.post("/generate")
def start_generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"events": [], "done": False, "output": None, "error": None}

    t = threading.Thread(target=_generation_thread, args=(job_id, req), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_events(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        last_ping = time.monotonic()
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                break

            events = job["events"]
            while sent < len(events):
                yield f"data: {events[sent]}\n\n"
                sent += 1
                last_ping = time.monotonic()

            if job["done"] and sent >= len(events):
                break

            # Send a keepalive SSE comment every 15 s so the browser/proxy
            # does not time out the idle chunked-transfer connection during
            # long model-loading phases (T5 + NF4 can take 5+ minutes).
            now = time.monotonic()
            if now - last_ping >= 15:
                yield ": ping\n\n"
                last_ping = now

            await asyncio.sleep(0.1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/output/{job_id}")
def download_output(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.get("output")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output not ready")
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))


# ── System Metrics ─────────────────────────────────────────────────────────────
_nvml_available = False
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    pass


@app.get("/metrics")
def get_metrics():
    cpu = round(psutil.cpu_percent(interval=0.1))
    vm  = psutil.virtual_memory()
    ram = round(vm.percent)

    gpu  = 0
    vram = 0
    gpu_temp = 0
    if _nvml_available:
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            util   = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem    = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu    = int(util.gpu)
            vram   = round(mem.used / mem.total * 100)
            gpu_temp = int(_pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
    elif torch.cuda.is_available():
        try:
            used  = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            vram  = round(used / total * 100)
        except Exception:
            pass

    cpu_temp = 0
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for key in ('coretemp', 'k10temp', 'cpu_thermal', 'acpitz'):
                if key in temps:
                    entries = temps[key]
                    if entries:
                        cpu_temp = round(sum(e.current for e in entries) / len(entries))
                        break
    except Exception:
        pass

    return {"cpu": cpu, "gpu": gpu, "vram": vram, "ram": ram, "cpu_temp": cpu_temp, "gpu_temp": gpu_temp}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
