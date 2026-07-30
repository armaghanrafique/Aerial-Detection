# aerial — object detection for drone / quadcopter footage

Detects outdoor objects (cars, people, buses, trucks, motorcycles, …) in video
shot from **above**, where objects are small and numerous. Built on Ultralytics
YOLO with optional **SAHI** tiled inference for tiny top-down objects and a
built-in tracker for stable IDs + unique counts.

## Two models

| Weights | Classes | When |
|---|---|---|
| `yolo11s.pt` (default, COCO) | 80 general classes | works out of the box; broad variety but trained on ground-level photos |
| `models/visdrone.pt` (VisDrone) | 10 aerial classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor | **best for real drone footage** — trained top-down |

The VisDrone model is domain-specialized: on a *ground-level* view it may detect
little, but on genuine aerial footage it far outperforms the COCO model.

## Usage

```bash
# Aerial-specialized model, with tracking (recommended for drone footage)
python -m aerial -i footage.mp4 -o out.mp4 --weights models/visdrone.pt --track

# Maximum small-object recall: SAHI slicing (slower, best for high altitude)
python -m aerial -i footage.mp4 -o out.mp4 --weights models/visdrone.pt \
    --slice --slice-size 640 --track --conf 0.2

# General COCO model, outdoor classes only
python -m aerial -i footage.mp4 -o out.mp4 --outdoor --track

# Single image
python -m aerial -i frame.jpg -o annotated.jpg --weights models/visdrone.pt
```

Outputs an annotated video plus `<output>.detections.json` (per-frame boxes,
classes, confidences, track IDs, and a summary with class totals + unique count).

## Key flags

| Flag | Meaning |
|---|---|
| `--weights PATH` | a YOLO `.pt` (e.g. `models/visdrone.pt`) |
| `--model NAME` | ultralytics model if no `--weights` (`yolo11n/s/m/l/x.pt`) |
| `--slice` | SAHI tiled inference — more small objects, slower |
| `--slice-size`, `--slice-overlap` | tile geometry (default 640, 0.2) |
| `--track` | stable IDs across frames + unique-object count |
| `--conf` | confidence threshold; **lower = more detections** (try 0.2 for aerial) |
| `--imgsz` | inference size; larger finds smaller objects, slower |
| `--outdoor` | with a COCO model, keep only outdoor-relevant classes |
| `--classes 2 5 7` | keep only these class ids |
| `--device` | `auto` (default: GPU if available, else CPU) · `cpu` · `cuda:0` · `mps` |
| `--vid-stride N` | process every Nth frame (speed up) |
| `--max-frames N` | stop early (quick tests) |

## Device: automatic GPU / CPU

`--device auto` (the default) uses an **NVIDIA CUDA GPU if one is available**,
then Apple MPS, otherwise **CPU** — no config needed. It prints the chosen
device at startup, e.g. `device: cuda:0 (NVIDIA RTX 4070)` or
`device: cpu (no CUDA/MPS GPU detected)`.

**This machine has no NVIDIA GPU** (AMD integrated only, which PyTorch can't use
on Windows), so it resolves to CPU: roughly **1–3 FPS** plain, **<1 FPS** sliced
— fine for offline processing. To use a GPU: run on an NVIDIA machine with the
CUDA build of PyTorch (`pip install torch --index-url
https://download.pytorch.org/whl/cu121`) and auto-detect handles the rest.
To go faster on CPU: use `yolo11n.pt`, raise `--vid-stride`, or lower `--imgsz`.

## Better accuracy: fine-tune your own

`train_visdrone.py` (repo root) fine-tunes YOLO on VisDrone (dataset
auto-downloads). It's a GPU job — run it on a GPU box, then copy `best.pt` to
`models/visdrone.pt`. You can also fine-tune on *your own* labeled drone frames
for the classes and altitude you actually fly.
