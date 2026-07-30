"""Annotation: draw detection boxes + labels onto RGB frames with cv2."""
import cv2

# A fixed, high-contrast palette (RGB). Class id -> colour, stable across frames.
_PALETTE = [
    (255, 56, 56), (255, 159, 56), (255, 224, 56), (153, 255, 56),
    (56, 255, 120), (56, 255, 232), (56, 168, 255), (56, 72, 255),
    (149, 56, 255), (255, 56, 224), (255, 56, 133), (128, 128, 128),
    (0, 200, 0), (0, 128, 255), (200, 0, 0), (255, 128, 0),
]


def _color(cls_id):
    return _PALETTE[cls_id % len(_PALETTE)]


def draw_detections(frame_rgb, detections, show_conf=True):
    """Draw boxes in place and return the frame.

    Each box is labelled with its class name only (plus confidence when
    `show_conf` is set). Track IDs are not drawn.
    """
    h, w = frame_rgb.shape[:2]
    thick = max(1, round((h + w) / 1000))
    font_scale = max(0.4, (h + w) / 2400)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        color = _color(det["cls"])
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, thick)

        label = det["name"]
        if show_conf:
            label += f" {det['conf']:.2f}"

        (tw, th), base = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ly = max(0, y1 - th - base)
        cv2.rectangle(frame_rgb, (x1, ly), (x1 + tw, ly + th + base),
                      color, -1)
        cv2.putText(frame_rgb, label, (x1, ly + th),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    thick, cv2.LINE_AA)
    return frame_rgb


def draw_hud(frame_rgb, lines):
    """Draw a small stats overlay (list of strings) in the top-left corner."""
    y = 6
    for line in lines:
        (tw, th), base = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame_rgb, (6, y), (6 + tw + 8, y + th + base + 6),
                      (0, 0, 0), -1)
        cv2.putText(frame_rgb, line, (10, y + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        y += th + base + 10
    return frame_rgb
