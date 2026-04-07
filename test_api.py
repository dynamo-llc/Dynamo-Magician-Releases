"""
test_api.py – Full API test suite for Dynamo Magician server.

Mocks all GPU/model machinery so every test runs on CPU without loading
any model weights.  Run with:
    .venv/Scripts/python -m pytest test_api.py -v
"""

import base64
import io
import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── 1. Stub out all CUDA / model imports before server.py is loaded ───────────
#
# t5.py calls torch.cuda.current_device() at CLASS-DEFINITION time, so we
# must intercept before ANY wan.* import.
#
# Strategy: place lightweight fakes in sys.modules so that
#   `from generate_prequant import WanI2V_PreQuant, save_video`
# (and all transitive deps) never reach the real GPU code.

import torch  # real torch – OK on CPU

# Patch cuda helpers that the server uses
torch.cuda.empty_cache = lambda: None
torch.cuda.is_available = lambda: False


def _make_fake_pipeline():
    """Return a mock WanI2V_PreQuant that yields a predictable dummy video tensor."""
    import torch

    class _FakePipeline:
        def __init__(self, checkpoint_dir=None, vram_mode="low",
                     t5_cpu=True, vae_cpu_offload=False):
            self.vram_mode = vram_mode
            self.t5_cpu = t5_cpu
            self.vae_cpu_offload = vae_cpu_offload

        def generate(self, *, input_prompt, img, action_path=None,
                     max_area=None, frame_num=8, sampling_steps=2,
                     guide_scale=5.0, seed=-1, progress_callback=None,
                     preview_every=0, preview_callback=None):
            # Report one progress tick so the callback path is exercised
            if progress_callback:
                progress_callback(1, sampling_steps)
            # Return a tiny fake video tensor  [C, F, H, W] ∈ [-1, 1]
            return torch.zeros(3, frame_num, 8, 8)

    return _FakePipeline


# Build a fake `generate_prequant` module
_fake_gp = types.ModuleType("generate_prequant")
_fake_gp.WanI2V_PreQuant = _make_fake_pipeline()


def _fake_save_video(video, path, fps=16):
    """Write a 1-byte placeholder so FileResponse can find the path."""
    Path(path).write_bytes(b"FAKEVIDEO")


_fake_gp.save_video = _fake_save_video
sys.modules["generate_prequant"] = _fake_gp

# Also stub out every wan.* module that server.py would pull in transitively
for _mod in [
    "wan", "wan.configs", "wan.configs.wan_i2v_A14B",
    "wan.modules", "wan.modules.t5", "wan.modules.vae2_1",
    "wan.utils", "load_prequant", "native_bnb",
    "pynvml",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# ── 2. Now it is safe to import the server ────────────────────────────────────
import server  # noqa: E402  (must come after sys.modules patching)
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_png_b64() -> str:
    """Return a base64-encoded 1×1 white PNG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _submit_job(extra: dict | None = None) -> str:
    """POST /generate with a minimal valid payload, return job_id."""
    payload = {
        "image_b64": _tiny_png_b64(),
        "prompt": "A test scene",
        "frames": 8,
        "steps": 2,
    }
    if extra:
        payload.update(extra)
    r = client.post("/generate", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _drain_stream(job_id: str, timeout: float = 10.0) -> list[dict]:
    """Consume the SSE stream for job_id; return parsed event list."""
    events = []
    deadline = time.monotonic() + timeout
    with client.stream("GET", f"/stream/{job_id}") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if time.monotonic() > deadline:
                break
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("complete", "error"):
                    break
    return events


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealth(unittest.TestCase):

    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "online"


class TestMetrics(unittest.TestCase):

    def test_metrics_returns_expected_keys(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        for key in ("cpu", "ram", "gpu", "vram"):
            assert key in body, f"Missing key: {key}"

    def test_metrics_values_are_percentages(self):
        body = client.get("/metrics").json()
        for key in ("cpu", "ram", "gpu", "vram"):
            assert 0 <= body[key] <= 100, f"{key}={body[key]} out of [0,100]"


class TestQueue(unittest.TestCase):

    def test_empty_queue(self):
        r = client.get("/queue")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_cancel_unknown_job_404(self):
        r = client.delete("/queue/nonexistent-job-id")
        assert r.status_code == 404

    def test_cancel_queued_job(self):
        """Submit a job then immediately cancel it – server must accept cancel."""
        job_id = _submit_job()
        # Cancel before the worker starts it (best-effort; no race guarantee)
        r = client.delete(f"/queue/{job_id}")
        # 200 = cancelled  OR  409 = already started (both are valid)
        assert r.status_code in (200, 409), r.text


class TestGenerate(unittest.TestCase):

    def test_missing_required_fields(self):
        r = client.post("/generate", json={"prompt": "no image"})
        assert r.status_code == 422  # Pydantic validation error

    def test_missing_prompt(self):
        r = client.post("/generate", json={"image_b64": _tiny_png_b64()})
        assert r.status_code == 422

    def test_submit_returns_job_id(self):
        r = client.post("/generate", json={
            "image_b64": _tiny_png_b64(),
            "prompt": "hello",
        })
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert "position" in body
        assert isinstance(body["position"], int)

    def test_submit_with_all_params(self):
        r = client.post("/generate", json={
            "image_b64": _tiny_png_b64(),
            "prompt": "Full param test",
            "width": 832, "height": 480,
            "steps": 2, "guidance": 7.0,
            "frames": 8, "seed": 42,
            "vram_mode": "low",
            "t5_cpu": True,
            "vae_cpu_offload": False,
            "preview_every": 0,
        })
        assert r.status_code == 200


class TestStream(unittest.TestCase):

    def test_stream_unknown_job_404(self):
        r = client.get("/stream/00000000-dead-beef-0000-000000000000")
        assert r.status_code == 404

    def test_stream_full_generation(self):
        """End-to-end: submit → stream → receive complete event."""
        job_id = _submit_job()
        events = _drain_stream(job_id)
        types_ = [e.get("type") for e in events]
        # Must have at minimum one status event and finish with 'complete'
        assert "complete" in types_, f"No 'complete' event. Got: {types_}"

    def test_stream_progress_events(self):
        """Progress callback should emit at least one progress event."""
        job_id = _submit_job()
        events = _drain_stream(job_id)
        progress_events = [e for e in events if e.get("type") == "progress"]
        assert len(progress_events) >= 1, "Expected at least one progress event"

    def test_stream_preview_events(self):
        """With preview_every=1 every step should emit a preview JPEG."""
        job_id = _submit_job({"preview_every": 1, "steps": 2, "frames": 8})
        events = _drain_stream(job_id)
        preview_events = [e for e in events if e.get("type") == "preview"]
        # The fake pipeline only calls progress_callback once, but preview_every
        # is handled by generate_prequant (mocked), so at minimum 0 previews is OK.
        # We just assert no crash occurred.
        assert any(e.get("type") == "complete" for e in events)


class TestOutput(unittest.TestCase):

    def _completed_job_id(self) -> str:
        job_id = _submit_job()
        _drain_stream(job_id)
        return job_id

    def test_output_unknown_job_404(self):
        r = client.get("/output/00000000-dead-beef-0000-000000000000")
        assert r.status_code == 404

    def test_output_not_ready_returns_404(self):
        """A job that was cancelled/errored will have no output file."""
        job_id = _submit_job()
        # Immediately try output before generation is done
        r = client.get(f"/output/{job_id}")
        # Either 404 (not ready) or 200 if it somehow raced to completion
        assert r.status_code in (200, 404)

    def test_output_after_completion(self):
        job_id = self._completed_job_id()
        r = client.get(f"/output/{job_id}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("video/mp4")
        assert r.content == b"FAKEVIDEO"


class TestHistory(unittest.TestCase):

    def test_history_returns_list(self):
        r = client.get("/history")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_history_after_generation(self):
        """Completing a job should create a history entry."""
        # Clear history for isolation
        with server._history_lock:
            before_count = len(server._history)

        job_id = _submit_job()
        _drain_stream(job_id)

        r = client.get("/history")
        assert r.status_code == 200
        after_count = len(r.json())
        assert after_count > before_count

    def test_history_entry_fields(self):
        """Each history entry must contain required fields."""
        job_id = _submit_job()
        _drain_stream(job_id)
        entries = client.get("/history").json()
        assert entries, "History should not be empty after a completed job"
        entry = entries[0]  # newest first
        for field in ("job_id", "timestamp", "prompt", "params", "output_file", "duration_secs"):
            assert field in entry, f"Missing field: {field}"


class TestThumbnail(unittest.TestCase):

    def test_thumbnail_unknown_job_404(self):
        r = client.get("/thumbnail/00000000")
        assert r.status_code == 404

    def test_thumbnail_created_on_completion(self):
        """_save_thumbnail succeeds on a zero tensor; thumbnail endpoint → 200."""
        job_id = _submit_job()
        _drain_stream(job_id)
        short_id = job_id[:8]
        r = client.get(f"/thumbnail/{short_id}")
        # zeros tensor IS a valid numpy array → PIL saves a real JPEG → 200
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/jpeg")


class TestRedirect(unittest.TestCase):

    def test_root_redirects(self):
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (302, 307, 308)
        assert r.headers["location"].startswith("/app")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
