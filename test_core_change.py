from core.change_detection_multiscale import detect_change_multiscale

import numpy as np
from PIL import Image


# =========================================================
# RUN MULTI-SCALE ON PAIR 01
# =========================================================

result = detect_change_multiscale(
    "dataset/change/pair_05/old.png",
    "dataset/change/pair_05/new.png",
    "integration_data/multiscale_pair_02"
)


# =========================================================
# LOAD PREDICTION
# =========================================================

pred_img = Image.open(
    result["mask_path"]
).convert("L")

pred = np.array(pred_img) > 127


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

gt_img = Image.open(
    "dataset/change/pair_05/mask.png"
).convert("L")

if gt_img.size != pred_img.size:
    gt_img = gt_img.resize(
        pred_img.size,
        Image.Resampling.NEAREST
    )

gt = np.array(gt_img) > 127


# =========================================================
# METRICS
# =========================================================

tp = np.logical_and(pred, gt).sum()
fp = np.logical_and(pred, np.logical_not(gt)).sum()
fn = np.logical_and(np.logical_not(pred), gt).sum()

union = np.logical_or(pred, gt).sum()

iou = tp / union if union > 0 else 0

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)

actual_change = gt.mean() * 100
predicted_change = pred.mean() * 100


# =========================================================
# RESULT
# =========================================================

print("\n========== PAIR 01 MULTI-SCALE ==========")

print(f"IoU: {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

print(f"Actual Change: {actual_change:.2f}%")
print(f"Predicted Change: {predicted_change:.2f}%")

print(f"512 Tiles: {result['tiles_512']}")
print(f"256 Tiles: {result['tiles_256']}")
print(f"Total Tiles: {result['total_tiles']}")

print("==========================================")