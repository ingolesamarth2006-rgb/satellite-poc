from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PRED_DIR = Path("integration_data/spacenet7_model_test")
GT_DIR = Path("integration_data/spacenet7_validation")
OUT_DIR = Path("integration_data/spacenet7_gridsearch_best")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prob(stem):
    npy_path = PRED_DIR / f"{stem}.npy"
    png_path = PRED_DIR / f"{stem}.png"

    if npy_path.exists():
        arr = np.load(npy_path).astype(np.float32)
        arr = np.squeeze(arr)
    elif png_path.exists():
        arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32)
    else:
        raise FileNotFoundError(f"Missing probability file for {stem}")

    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0


def save_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def remove_small(mask, min_size):
    if min_size <= 0:
        return mask

    labeled, num = ndi.label(mask)
    if num == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labeled]


def metrics(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    iou = tp / (tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def print_metric(name, pred, gt):
    m = metrics(pred, gt)
    print(
        f"{name:<14} | "
        f"IoU={m['iou']:.4f}  "
        f"Precision={m['precision']:.4f}  "
        f"Recall={m['recall']:.4f}  "
        f"F1={m['f1']:.4f}  "
        f"PredArea={pred.mean()*100:.2f}%  "
        f"GTArea={gt.mean()*100:.2f}%"
    )


# ---------------------------------------------------
# Load predicted probabilities
# ---------------------------------------------------
prob17 = load_prob("probability_2017")
prob19 = load_prob("probability_2019")

# ---------------------------------------------------
# Load GT building masks
# ---------------------------------------------------
gt17 = load_mask(GT_DIR / "gt_buildings_2017_07.png")
gt19 = load_mask(GT_DIR / "gt_buildings_2019_09.png")

# Ground-truth temporal change
gt_new = np.logical_and(gt19, ~gt17)
gt_removed = np.logical_and(gt17, ~gt19)
gt_change = np.logical_or(gt_new, gt_removed)

print("Loaded:")
print("prob17 shape:", prob17.shape, " range:", (float(prob17.min()), float(prob17.max())))
print("prob19 shape:", prob19.shape, " range:", (float(prob19.min()), float(prob19.max())))
print("GT 2017 area:", round(gt17.mean() * 100, 2), "%")
print("GT 2019 area:", round(gt19.mean() * 100, 2), "%")
print("GT NEW area:", round(gt_new.mean() * 100, 2), "%")
print("GT REMOVED area:", round(gt_removed.mean() * 100, 2), "%")
print("GT TOTAL CHANGE area:", round(gt_change.mean() * 100, 2), "%")
print()

# ---------------------------------------------------
# Grid search
# ---------------------------------------------------
th17_list = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
th19_list = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
building_min_sizes = [0, 8, 16, 32, 64]
change_min_sizes = [0, 5, 10, 20, 40]

results = []
best = None
best_payload = None

total_runs = len(th17_list) * len(th19_list) * len(building_min_sizes) * len(change_min_sizes)
run_id = 0

for th17 in th17_list:
    for th19 in th19_list:
        for build_min in building_min_sizes:
            for change_min in change_min_sizes:
                run_id += 1

                b17 = prob17 >= th17
                b19 = prob19 >= th19

                b17 = remove_small(b17, build_min)
                b19 = remove_small(b19, build_min)

                new_pred = np.logical_and(b19, ~b17)
                removed_pred = np.logical_and(b17, ~b19)

                new_pred = remove_small(new_pred, change_min)
                removed_pred = remove_small(removed_pred, change_min)
                change_pred = np.logical_or(new_pred, removed_pred)

                m_total = metrics(change_pred, gt_change)
                m_new = metrics(new_pred, gt_new)
                m_removed = metrics(removed_pred, gt_removed)
                m17 = metrics(b17, gt17)
                m19 = metrics(b19, gt19)

                row = {
                    "th17": th17,
                    "th19": th19,
                    "build_min": build_min,
                    "change_min": change_min,
                    "total_f1": m_total["f1"],
                    "total_iou": m_total["iou"],
                    "total_precision": m_total["precision"],
                    "total_recall": m_total["recall"],
                    "new_f1": m_new["f1"],
                    "removed_f1": m_removed["f1"],
                    "b17_f1": m17["f1"],
                    "b19_f1": m19["f1"],
                    "pred_change_area": float(change_pred.mean() * 100),
                }
                results.append(row)

                score = (m_total["f1"], m_total["iou"], m_total["precision"])
                if best is None or score > best:
                    best = score
                    best_payload = {
                        "params": row,
                        "b17": b17.copy(),
                        "b19": b19.copy(),
                        "new_pred": new_pred.copy(),
                        "removed_pred": removed_pred.copy(),
                        "change_pred": change_pred.copy(),
                    }

                if run_id % 100 == 0 or run_id == total_runs:
                    print(f"[{run_id}/{total_runs}] done")

# ---------------------------------------------------
# Show top results
# ---------------------------------------------------
results_sorted = sorted(
    results,
    key=lambda x: (x["total_f1"], x["total_iou"], x["total_precision"]),
    reverse=True
)

print("\nTOP 10 RESULTS")
for i, r in enumerate(results_sorted[:10], start=1):
    print(
        f"{i:02d}. "
        f"th17={r['th17']:.2f}, th19={r['th19']:.2f}, "
        f"build_min={r['build_min']}, change_min={r['change_min']} | "
        f"TOTAL: F1={r['total_f1']:.4f}, IoU={r['total_iou']:.4f}, "
        f"P={r['total_precision']:.4f}, R={r['total_recall']:.4f} | "
        f"NEW_F1={r['new_f1']:.4f} | REMOVED_F1={r['removed_f1']:.4f} | "
        f"B17_F1={r['b17_f1']:.4f} | B19_F1={r['b19_f1']:.4f}"
    )

# ---------------------------------------------------
# Save best outputs
# ---------------------------------------------------
best_params = best_payload["params"]
best_b17 = best_payload["b17"]
best_b19 = best_payload["b19"]
best_new = best_payload["new_pred"]
best_removed = best_payload["removed_pred"]
best_change = best_payload["change_pred"]

save_mask(best_b17, OUT_DIR / "best_building_2017.png")
save_mask(best_b19, OUT_DIR / "best_building_2019.png")
save_mask(best_new, OUT_DIR / "best_new_buildings.png")
save_mask(best_removed, OUT_DIR / "best_removed_buildings.png")
save_mask(best_change, OUT_DIR / "best_total_change.png")

print("\nBEST PARAMETERS")
for k, v in best_params.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

print("\nBEST RESULT DETAILED METRICS")
print_metric("BUILDING 2017", best_b17, gt17)
print_metric("BUILDING 2019", best_b19, gt19)
print_metric("NEW", best_new, gt_new)
print_metric("REMOVED", best_removed, gt_removed)
print_metric("TOTAL CHANGE", best_change, gt_change)

print("\nSaved outputs in:", OUT_DIR)