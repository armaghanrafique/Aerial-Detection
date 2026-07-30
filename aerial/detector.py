"""Aerial object detector — Ultralytics YOLO, with optional SAHI slicing.

Designed for drone / quadcopter footage shot from above, where objects are
small and numerous. Two inference paths, both returning the same detection
format so the rest of the pipeline doesn't care which was used:

  - plain  : one forward pass over the (optionally resized) frame. Fast.
  - sliced : SAHI cuts the frame into overlapping tiles, runs YOLO on each and
             merges the boxes. Far better small-object recall on high-altitude
             footage, at a roughly (n_tiles)x compute cost.

Weights:
  - default `yolo11s.pt`  -> COCO (80 classes: person, car, bus, truck, ...).
    Auto-downloads on first use. Works today but is trained on ground-level
    photos, so top-down recall is limited.
  - drop-in `--weights visdrone.pt` -> a model fine-tuned on VisDrone (10
    drone-specific classes). Best accuracy for this use case. See
    `train_visdrone.py` to produce one.

A detection is a dict:
    {"cls": int, "name": str, "conf": float, "xyxy": (x1, y1, x2, y2)}
with pixel coordinates on the input frame.
"""
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def resolve_device(requested="auto"):
    """Pick the compute device.

    "auto" (default) -> NVIDIA CUDA GPU if available, else Apple MPS, else CPU.
    Any explicit value ("cpu", "cuda:0", "mps", "0,1") is returned unchanged.

    Returns (device_str, human_label).
    """
    if requested and requested != "auto":
        return requested, requested
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda:0", f"cuda:0 ({name})"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "mps (Apple GPU)"
    except Exception:  # noqa: BLE001 — torch missing/broken -> fall back to CPU
        pass
    return "cpu", "cpu (no CUDA/MPS GPU detected)"

# COCO class ids that actually occur in outdoor aerial scenes. Used by the
# `--outdoor` preset to suppress spurious indoor detections from a COCO model.
COCO_OUTDOOR = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    6,   # train
    7,   # truck
    8,   # boat
    9,   # traffic light
    10,  # fire hydrant
    11,  # stop sign
    12,  # parking meter
    13,  # bench
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    24,  # backpack
    25,  # umbrella
    28,  # suitcase
]


class AerialDetector:
    def __init__(self, weights=None, model="yolo11s.pt", conf=0.25, iou=0.5,
                 imgsz=640, device="auto", classes=None, outdoor=False,
                 slice_mode=False, slice_size=640, slice_overlap=0.2):
        from ultralytics import YOLO

        self.device, self.device_label = resolve_device(device)
        self.weights = self._resolve(weights or model)
        self.model = YOLO(self.weights)
        self.model.to(self.device)
        self.names = self.model.names            # {id: name}
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        if classes is None and outdoor and self._looks_like_coco():
            classes = COCO_OUTDOOR
        self.classes = classes

        self.slice_mode = slice_mode
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self._sahi = None
        if slice_mode:
            self._init_sahi()

    # ---- weight resolution -------------------------------------------------
    def _resolve(self, w):
        """Accept a bare ultralytics name, an explicit path, or a models/ file."""
        p = Path(w)
        if p.is_file():
            return str(p)
        local = MODELS_DIR / w
        if local.is_file():
            return str(local)
        return w  # let ultralytics resolve/download known names (yolo11s.pt, ...)

    def _looks_like_coco(self):
        return len(self.names) >= 80 and self.names.get(0, "").lower() == "person"

    # ---- SAHI setup --------------------------------------------------------
    def _init_sahi(self):
        from sahi import AutoDetectionModel
        # sahi >= 0.11.16 uses model_type="ultralytics"; older uses "yolov8".
        for mtype in ("ultralytics", "yolov8"):
            try:
                self._sahi = AutoDetectionModel.from_pretrained(
                    model_type=mtype,
                    model_path=self.weights,
                    confidence_threshold=self.conf,
                    device=self.device,
                    image_size=self.imgsz,
                )
                return
            except Exception:  # noqa: BLE001 — try the next model_type name
                continue
        raise RuntimeError(
            "SAHI could not wrap the YOLO model. Check `pip install sahi` "
            "and that the weights load in ultralytics.")

    # ---- inference ---------------------------------------------------------
    def detect(self, frame_rgb):
        return self._detect_sliced(frame_rgb) if self.slice_mode \
            else self._detect_plain(frame_rgb)

    def _detect_plain(self, frame_rgb):
        res = self.model.predict(
            frame_rgb, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            classes=self.classes, device=self.device, verbose=False)
        out = []
        for r in res:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for box, c, p in zip(xyxy, cls, cf):
                out.append({
                    "cls": int(c), "name": self.names.get(int(c), str(c)),
                    "conf": float(p),
                    "xyxy": (float(box[0]), float(box[1]),
                             float(box[2]), float(box[3])),
                })
        return out

    def _detect_sliced(self, frame_rgb):
        from sahi.predict import get_sliced_prediction
        keep = set(self.classes) if self.classes else None
        result = get_sliced_prediction(
            frame_rgb, self._sahi,
            slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.slice_overlap,
            overlap_width_ratio=self.slice_overlap,
            postprocess_type="NMS", postprocess_match_threshold=self.iou,
            verbose=0)
        out = []
        for op in result.object_prediction_list:
            cid = op.category.id
            if keep is not None and cid not in keep:
                continue
            b = op.bbox
            out.append({
                "cls": int(cid), "name": op.category.name,
                "conf": float(op.score.value),
                "xyxy": (float(b.minx), float(b.miny),
                         float(b.maxx), float(b.maxy)),
            })
        return out
