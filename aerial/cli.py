"""Command-line interface for aerial object detection on drone footage.

    python -m aerial -i footage.mp4 -o out.mp4
    python -m aerial -i footage.mp4 -o out.mp4 --slice --track
    python -m aerial -i frame.jpg   -o out.jpg --weights models/visdrone.pt

Outputs an annotated video/image and (for video) a JSON sidecar of per-frame
detections. Reuses the project's imageio-ffmpeg video I/O.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .detector import AerialDetector
from .draw import draw_detections
from .video import probe_video

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _build_parser():
    p = argparse.ArgumentParser(
        prog="aerial",
        description="Detect outdoor objects (cars, people, ...) in "
                    "quadcopter/drone footage using YOLO, tuned for the "
                    "top-down aerial view.")
    p.add_argument("-i", "--input", required=True,
                   help="input video, image, or folder of images")
    p.add_argument("-o", "--output", required=True,
                   help="output annotated video/image path")
    p.add_argument("--weights", default=None,
                   help="path/URL to a YOLO .pt (e.g. a VisDrone-trained model)")
    p.add_argument("--model", default="yolo11s.pt",
                   help="ultralytics model name if --weights not given "
                        "(yolo11n/s/m/l/x.pt). Default: yolo11s.pt")
    p.add_argument("--device", default="auto",
                   help="auto (GPU if available, else CPU) | cpu | cuda:0 | mps")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence threshold (lower = more detections)")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640,
                   help="inference size; larger finds smaller objects, slower")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="only keep these class ids")
    p.add_argument("--outdoor", action="store_true",
                   help="with a COCO model, keep only outdoor-relevant classes")
    # small-object slicing (SAHI)
    p.add_argument("--slice", dest="slice_mode", action="store_true",
                   help="SAHI tiled inference: best small-object recall, slower")
    p.add_argument("--slice-size", type=int, default=640, help="SAHI tile size")
    p.add_argument("--slice-overlap", type=float, default=0.2,
                   help="SAHI tile overlap ratio (0-1)")
    # tracking
    p.add_argument("--track", action="store_true",
                   help="assign stable IDs across frames + count unique objects")
    # video controls
    p.add_argument("--vid-stride", type=int, default=1,
                   help="process every Nth frame (speed up)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after N processed frames (for quick tests)")
    p.add_argument("--no-audio", action="store_true",
                   help="don't re-mux source audio into the output video")
    p.add_argument("--no-conf", action="store_true", help="hide confidences")
    p.add_argument("--json", default=None,
                   help="detections JSON path (default: <output>.json)")
    return p


def _run_image(det, args):
    import imageio.v2 as imageio
    frame = imageio.imread(args.input)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    frame = frame[:, :, :3].copy()
    detections = det.detect(frame)
    draw_detections(frame, detections, show_conf=not args.no_conf)
    imageio.imwrite(args.output, frame)
    counts = Counter(d["name"] for d in detections)
    print(f"  {len(detections)} objects: {dict(counts)}")
    print(f"  wrote {args.output}")


def _run_video(det, args):
    from .process import process_video

    n_frames = probe_video(args.input)[1]

    def _progress(processed, total, n_dets, fps):
        if processed % 20 == 0 or processed == 1:
            tail = f"/{n_frames}" if n_frames else ""
            print(f"\r  frame {processed}{tail}  "
                  f"{n_dets} objs  {fps:4.1f} FPS", end="", flush=True)

    summary = process_video(
        det, args.input, args.output, track=args.track,
        vid_stride=args.vid_stride, max_frames=args.max_frames,
        no_audio=args.no_audio, show_conf=not args.no_conf,
        json_path=args.json, progress_cb=_progress)

    print()
    print(f"  done: {summary['frames_processed']} frames @ "
          f"{summary['avg_fps']} FPS")
    print(f"  class totals (detections): {summary['class_totals']}")
    if summary["unique_tracks"] is not None:
        print(f"  unique objects tracked: {summary['unique_tracks']}")
    print(f"  video -> {args.output}")
    print(f"  json  -> {summary['json_path']}")


def main(argv=None):
    args = _build_parser().parse_args(argv)
    src = Path(args.input)
    if not src.exists():
        sys.exit(f"input not found: {src}")

    print(f"[aerial] loading detector "
          f"({args.weights or args.model}, slice={args.slice_mode})")
    det = AerialDetector(
        weights=args.weights, model=args.model, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, device=args.device, classes=args.classes,
        outdoor=args.outdoor, slice_mode=args.slice_mode,
        slice_size=args.slice_size, slice_overlap=args.slice_overlap)
    print(f"[aerial] device: {det.device_label}")
    print(f"[aerial] classes available: {len(det.names)}  "
          f"(filtering to {det.classes if det.classes else 'all'})")

    if src.suffix.lower() in _IMG_EXT:
        _run_image(det, args)
    else:
        _run_video(det, args)


if __name__ == "__main__":
    main()
