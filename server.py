"""Local web server for AerialDetector.

Run:
    python server.py            # then open http://127.0.0.1:8000

Flow: the browser uploads a video -> a background thread runs YOLO detection and
writes an annotated H.264 mp4 into a temp dir -> the browser streams that file
into a <video> player. Nothing is written to the user's machine until they click
Download (which serves the same file as an attachment).

Detection is CPU-bound and single-job-at-a-time here; that's fine for local use.
"""
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aerial.detector import AerialDetector
from aerial.process import process_video

BASE = Path(__file__).resolve().parent
WEB = BASE / "web"
WORK = Path(tempfile.gettempdir()) / "aerialdetector_jobs"
WORK.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AerialDetector")

# job_id -> dict(state, processed, total, fps, output, input, summary, error, name)
JOBS = {}
_LOCK = threading.Lock()

# Detector cache keyed by (weights, conf, slice) so we don't reload every job.
_DETECTORS = {}


def _get_detector(weights, conf, slice_mode):
    key = (weights, round(conf, 3), slice_mode)
    with _LOCK:
        det = _DETECTORS.get(key)
        if det is None:
            det = AerialDetector(weights=weights, conf=conf, device="auto",
                                 slice_mode=slice_mode)
            _DETECTORS[key] = det
        return det


def _run_job(job_id, in_path, out_path, weights, conf, slice_mode, track):
    job = JOBS[job_id]
    try:
        job["state"] = "loading"
        det = _get_detector(weights, conf, slice_mode)
        job["state"] = "processing"
        job["device"] = det.device_label

        def cb(processed, total, n_dets, fps):
            job["processed"] = processed
            job["total"] = total
            job["fps"] = round(fps, 2)
            job["last_dets"] = n_dets

        summary = process_video(det, in_path, out_path, track=track,
                                progress_cb=cb)
        job["summary"] = summary
        job["state"] = "done"
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        job["state"] = "error"
        job["error"] = str(e)


@app.post("/api/detect")
async def detect(file: UploadFile = File(...),
                 model: str = Form("visdrone"),
                 conf: float = Form(0.25),
                 slice_mode: bool = Form(False),
                 track: bool = Form(True)):
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    in_path = job_dir / f"input{suffix}"
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    out_path = job_dir / "detected.mp4"

    weights = str(BASE / "models" / "visdrone.pt") if model == "visdrone" \
        else "yolo11s.pt"

    JOBS[job_id] = {"state": "queued", "processed": 0, "total": None,
                    "fps": 0.0, "output": str(out_path), "input": str(in_path),
                    "name": file.filename or "input.mp4", "error": None}

    threading.Thread(
        target=_run_job,
        args=(job_id, str(in_path), str(out_path), weights, conf,
              bool(slice_mode), bool(track)),
        daemon=True).start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    pct = None
    if job["total"]:
        pct = min(100, round(job["processed"] * 100 / job["total"]))
    return {"state": job["state"], "processed": job["processed"],
            "total": job["total"], "percent": pct, "fps": job.get("fps"),
            "device": job.get("device"), "error": job.get("error"),
            "summary": job.get("summary")}


@app.get("/api/result/{job_id}")
def result(job_id: str, download: bool = False):
    job = JOBS.get(job_id)
    if not job or job["state"] != "done":
        raise HTTPException(404, "result not ready")
    out = Path(job["output"])
    if not out.is_file():
        raise HTTPException(404, "output missing")
    if download:
        stem = Path(job["name"]).stem
        return FileResponse(out, media_type="video/mp4",
                            filename=f"{stem}_detected.mp4")
    # inline playback (supports range requests for seeking)
    return FileResponse(out, media_type="video/mp4")


@app.get("/api/detections/{job_id}")
def detections(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["state"] != "done":
        raise HTTPException(404, "not ready")
    jp = job["summary"].get("json_path")
    if not jp or not Path(jp).is_file():
        raise HTTPException(404, "no detections file")
    return FileResponse(jp, media_type="application/json",
                        filename="detections.json")


@app.delete("/api/job/{job_id}")
def cleanup(job_id: str):
    """Drop a job and its temp files (called when the user starts over)."""
    JOBS.pop(job_id, None)
    shutil.rmtree(WORK / job_id, ignore_errors=True)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


if __name__ == "__main__":
    print("AerialDetector web UI -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
