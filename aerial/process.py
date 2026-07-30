"""Shared video-processing loop used by both the CLI and the web server.

Reads a video, runs the detector on each frame, draws boxes/HUD, writes an
annotated H.264 mp4 (browser-playable) and a JSON sidecar of all detections.
Reports progress through an optional callback so a UI can show a progress bar.
"""
import json
import time
from collections import Counter

import numpy as np

from .draw import draw_detections, draw_hud
from .tracker import SimpleTracker
from .video import VideoWriter, iter_frames, probe_video, remux_audio


def process_video(det, input_path, output_path, *, track=True, vid_stride=1,
                  max_frames=None, no_audio=False, show_conf=True,
                  json_path=None, progress_cb=None):
    """Run detection over a video. Returns a summary dict.

    progress_cb(processed, total_or_None, n_dets, fps) is called each frame.
    """
    fps, n_frames = probe_video(input_path)
    total = None
    if n_frames:
        total = n_frames // max(1, vid_stride)
        if max_frames:
            total = min(total, max_frames)
    elif max_frames:
        total = max_frames

    out_fps = fps / max(1, vid_stride)
    writer = VideoWriter(output_path, fps=out_fps)
    tracker = SimpleTracker() if track else None
    json_path = json_path or (str(output_path) + ".detections.json")

    records = []
    class_totals = Counter()
    processed = 0
    t0 = time.time()
    try:
        for idx, frame in enumerate(iter_frames(input_path)):
            if idx % vid_stride != 0:
                continue
            frame = np.ascontiguousarray(frame)
            detections = det.detect(frame)
            if tracker is not None:
                tracker.update(detections)
            class_totals.update(d["name"] for d in detections)

            draw_detections(frame, detections, show_conf=show_conf)
            hud = [f"frame {idx}"]
            draw_hud(frame, hud)
            writer.append(frame)

            records.append({
                "frame": idx,
                "detections": [
                    {"cls": d["cls"], "name": d["name"],
                     "conf": round(d["conf"], 4),
                     "xyxy": [round(v, 1) for v in d["xyxy"]],
                     **({"id": d["id"]} if "id" in d else {})}
                    for d in detections],
            })

            processed += 1
            if progress_cb is not None:
                elapsed = time.time() - t0
                speed = processed / elapsed if elapsed else 0.0
                progress_cb(processed, total, len(detections), speed)
            if max_frames and processed >= max_frames:
                break
    finally:
        writer.close()

    if not no_audio and not max_frames:
        remux_audio(output_path, input_path)

    elapsed = time.time() - t0
    summary = {
        "input": str(input_path),
        "weights": det.weights,
        "device": det.device_label,
        "tracked": track,
        "frames_processed": processed,
        "avg_fps": round(processed / elapsed, 2) if elapsed else None,
        "class_totals": dict(class_totals),
        "unique_tracks": tracker.total_ids if tracker else None,
    }
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "frames": records}, f, indent=1)
    summary["json_path"] = json_path
    return summary
