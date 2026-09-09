from PIL import Image
import numpy as np

old_path = "test_c/old.png"
new_path = "test_c/new.png"
gt_path = "test_c/ground_truth.png"

print("Loading images...")

old_img = Image.open(old_path).convert("RGB")
new_img = Image.open(new_path).convert("RGB")
gt_img = Image.open(gt_path).convert("L")

# same size
new_img = new_img.resize(old_img.size)
gt_img = gt_img.resize(old_img.size)

old_arr = np.array(old_img).astype(np.int16)
new_arr = np.array(new_img).astype(np.int16)
gt_arr = np.array(gt_img)

print("Comparing old and new images...")

# simple difference
diff = np.abs(new_arr - old_arr).mean(axis=2)

# threshold
threshold = 20
pred_mask = (diff > threshold).astype(np.uint8) * 255

# save predicted mask
pred_mask_img = Image.fromarray(pred_mask)
pred_mask_img.save("test_c/pred_mask.png")

# create overlay
overlay = np.array(new_img).copy()
overlay[pred_mask > 0] = [255, 0, 0]
overlay_img = Image.fromarray(overlay)
overlay_img.save("test_c/change_overlay.png")

# ground truth binary
gt_bin = gt_arr > 127
pred_bin = pred_mask > 127

intersection = np.logical_and(gt_bin, pred_bin).sum()
union = np.logical_or(gt_bin, pred_bin).sum()

if union == 0:
    iou = 0.0
else:
    iou = intersection / union

accuracy = (gt_bin == pred_bin).mean() * 100

print("\n========== TEST C RESULT ==========")
print("Predicted mask saved: test_c/pred_mask.png")
print("Overlay saved: test_c/change_overlay.png")
print(f"IoU with ground truth: {iou:.4f}")
print(f"Pixel Accuracy: {accuracy:.2f}%")
print("===================================")