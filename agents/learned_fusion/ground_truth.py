"""ground_truth.py - read MUDCAD-X label.png and label each pipeline candidate."""
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_gt(label_path, min_area: int = 20) -> Tuple[List[List[int]], np.ndarray]:
    """Returns (boxes [x, y, w, h], target_mask). Any non-black pixel in label.png is a target,
    the same rule used in src/prepare_cnn_data.py and src/prepare_yolo_dataset.py."""
    label = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
    if label is None:
        raise ValueError(f"Could not read {label_path}")
    mask = np.any(label != 0, axis=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            boxes.append([int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])])
    return boxes, mask


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def label_candidate(bbox, gt_boxes, mask, iou_thr: float = 0.5, frac_thr: float = 0.6) -> Dict:
    """Positive (1) if the candidate box matches a target box (IoU >= iou_thr) OR mostly lies on
    target pixels (>= frac_thr of the box area). Otherwise negative (0)."""
    best_iou = max((iou_xywh(bbox, g) for g in gt_boxes), default=0.0)
    x, y, w, h = [int(v) for v in bbox]
    x, y = max(x, 0), max(y, 0)
    region = mask[y:y + max(h, 1), x:x + max(w, 1)]
    frac = float(region.mean()) if region.size else 0.0
    label = 1 if (best_iou >= iou_thr or frac >= frac_thr) else 0
    return {"label": label, "best_iou": round(best_iou, 4), "mask_fraction": round(frac, 4)}