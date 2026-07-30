"""A dependency-free IoU tracker for stable object IDs across frames.

Greedy IoU matching with a short memory (`max_age`) so a car that flickers out
for a frame or two keeps its ID. Good enough to count unique objects and draw
trajectories on drone footage. Works identically for plain and SAHI detections
because it only sees boxes.

For production-grade tracking (occlusion handling, motion model), swap in
ByteTrack via ultralytics `model.track(...)` — this is the zero-extra-dep option.
"""


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_age=15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._next_id = 0
        self._tracks = {}  # id -> {"xyxy", "cls", "age", "hits"}

    def update(self, detections):
        """Attach a stable `id` to each detection dict (in place) and return it.

        Matches detections to existing tracks by best IoU within the same class.
        """
        unmatched = set(self._tracks)
        for det in detections:
            best_id, best_iou = None, self.iou_thresh
            for tid in unmatched:
                t = self._tracks[tid]
                if t["cls"] != det["cls"]:
                    continue
                i = _iou(t["xyxy"], det["xyxy"])
                if i >= best_iou:
                    best_iou, best_id = i, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = {"cls": det["cls"], "hits": 0}
            else:
                unmatched.discard(best_id)
            t = self._tracks[best_id]
            t.update(xyxy=det["xyxy"], age=0)
            t["hits"] = t.get("hits", 0) + 1
            det["id"] = best_id

        # age out tracks we didn't match this frame
        for tid in list(unmatched):
            self._tracks[tid]["age"] = self._tracks[tid].get("age", 0) + 1
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]
        return detections

    @property
    def total_ids(self):
        """Highest id issued so far == count of unique objects ever seen."""
        return self._next_id
