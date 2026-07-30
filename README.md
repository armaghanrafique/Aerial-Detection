# AerialDetector

Object detection for **drone / quadcopter footage** — a standalone system that
takes a video shot from above and detects the outdoor objects in it (cars,
people, buses, trucks, motorcycles, bicycles, vans, …), draws labelled boxes,
tracks each object across frames, and writes an annotated video plus a JSON of
every detection.

Built on **Ultralytics YOLO11** with:
- a **VisDrone-trained** model specialised for the top-down aerial view,
- optional **SAHI** tiled inference for tiny high-altitude objects,
- a built-in **tracker** for stable IDs + unique-object counts,
- **automatic GPU/CPU** selection.

## Install

```powershell
# from this folder
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of torch first (see comments in
`requirements.txt`); the detector then uses it automatically.

## Web UI (drag-and-drop in the browser)

```powershell
python server.py
# then open http://127.0.0.1:8000
```

Drag a video onto the page (or click to choose one), pick the model / confidence
/ slicing / tracking, and press **Detect**. A progress bar shows frames-per-second
and percent. When it finishes, the annotated video plays **in the browser** — it
is *not* saved to your computer. It is written to your machine only when you click
**⬇ Download video** (or **Download detections (JSON)**). "Start over" clears the
job and its temp files.

Everything runs locally: the browser talks only to the Python server on your own
machine; no video ever leaves your computer.

## Run (command line)

```powershell
# Aerial-specialised model + tracking (recommended for drone footage)
python -m aerial -i footage.mp4 -o out\detected.mp4 --weights models\visdrone.pt --track --conf 0.2

# Maximum small-object recall: SAHI slicing (slower, best for high altitude)
python -m aerial -i footage.mp4 -o out\detected.mp4 --weights models\visdrone.pt --slice --track --conf 0.2

# General COCO model (80 classes), outdoor classes only
python -m aerial -i footage.mp4 -o out\detected.mp4 --outdoor --track

# Single image
python -m aerial -i frame.jpg -o annotated.jpg --weights models\visdrone.pt
```

Outputs an annotated video and `<output>.detections.json` (per-frame boxes,
classes, confidences, track IDs, plus a summary with class totals and the
unique-object count).

## Models

| Weights | Classes | When |
|---|---|---|
| `models/visdrone.pt` | 10 aerial: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor | **drone footage** — trained top-down |
| `yolo11s.pt` / `--model` | 80 COCO classes | general; auto-downloads |

`models/visdrone.pt` is included (from `mshamrai/yolov8s-visdrone`). To train a
better one on your own footage or with a larger backbone, see
[`train_visdrone.py`](train_visdrone.py) (GPU job).

## Device: automatic GPU / CPU

`--device auto` (default) → NVIDIA CUDA GPU if available, else Apple MPS, else
CPU. It prints the chosen device at startup, e.g.
`device: cuda:0 (NVIDIA RTX 4070)` or `device: cpu (no CUDA/MPS GPU detected)`.

## Layout

```
AerialDetector/
  server.py       local web server (FastAPI) -> http://127.0.0.1:8000
  web/            browser UI (index.html, app.js, style.css)
  aerial/
    detector.py   YOLO wrapper: plain or SAHI-sliced, auto GPU/CPU
    tracker.py    dependency-free IoU tracker (stable IDs)
    draw.py       box / label / HUD annotation
    video.py      imageio-ffmpeg video read/write + audio remux
    process.py    shared frame loop (used by CLI + web)
    cli.py        command-line interface
  train_visdrone.py   fine-tune YOLO on VisDrone (GPU)
  models/         weights (visdrone.pt included)
  requirements.txt
```

See [`aerial/README.md`](aerial/README.md) for the full flag reference.
