from PIL import Image
from pathlib import Path
import numpy as np

pred_dir = Path("integration_data/spacenet7_model_test")
gt_dir = Path("integration_data/spacenet7_validation")

# Official GT building masks
g17 = np.array(Image.open(gt_dir / "gt_buildings_2017_07.png")) > 0
g19 = np.array(Image.open(gt_dir / "gt_buildings_2019_09.png")) > 0

# Ground-truth temporal changes
gt_new = np.logical_and(g19, ~g17)
gt_removed = np.logical_and(g17, ~g19)
gt_change = np.logical_or(gt_new, gt_removed)

Image.fromarray((gt_new * 255).astype(np.uint8)).save(gt_dir / "gt_new_buildings.png")
Image.fromarray((gt_removed * 255).astype(np.uint8)).save(gt_dir / "gt_removed_buildings.png")
Image.fromarray((gt_change * 255).astype(np.uint8)).save(gt_dir / "gt_change_mask.png")

def metrics(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return iou, precision, recall, f1

tests = [
    ("NEW", "new_buildings.png", gt_new),
    ("REMOVED", "removed_buildings.png", gt_removed),
    ("TOTAL CHANGE", "change_mask.png", gt_change),
]

for name, filename, gt in tests:
    pred = np.array(Image.open(pred_dir / filename)) > 0
    iou, precision, recall, f1 = metrics(pred, gt)

    print(
        name,
        "| IoU =", round(iou, 4),
        "| Precision =", round(precision, 4),
        "| Recall =", round(recall, 4),
        "| F1 =", round(f1, 4)
    )

print("GT NEW AREA %:", round(gt_new.mean() * 100, 2))
print("GT REMOVED AREA %:", round(gt_removed.mean() * 100, 2))
print("GT TOTAL CHANGE %:", round(gt_change.mean() * 100, 2))
print("CHANGE VALIDATION COMPLETE")
