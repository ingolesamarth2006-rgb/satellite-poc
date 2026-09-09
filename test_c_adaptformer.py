import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

MODEL_ID = "deepang/adaptformer-LEVIR-CD"

print("Loading AdaptFormer...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model.eval()

print("Loading old/new images...")

old_img = Image.open("test_c/old.png").convert("RGB")
new_img = Image.open("test_c/new.png").convert("RGB")
gt_img = Image.open("test_c/ground_truth.png").convert("L")

print("Running AI change detection...")

inputs = processor(
    images=(old_img, new_img),
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# Resize AI output to original image size
logits = F.interpolate(
    logits,
    size=(old_img.height, old_img.width),
    mode="bilinear",
    align_corners=False
)

prediction = logits.argmax(dim=1)[0].cpu().numpy()

# 0 = no change, 1 = change
pred_mask = (prediction * 255).astype(np.uint8)

Image.fromarray(pred_mask).save(
    "test_c/ai_change_mask.png"
)

# Ground truth
gt = np.array(gt_img.resize(old_img.size))
gt_bin = gt > 127
pred_bin = prediction > 0

# IoU
intersection = np.logical_and(gt_bin, pred_bin).sum()
union = np.logical_or(gt_bin, pred_bin).sum()

iou = intersection / union if union > 0 else 0

# Pixel accuracy
accuracy = (gt_bin == pred_bin).mean() * 100

# Overlay on new image
overlay = np.array(new_img).copy()
overlay[pred_bin] = [255, 0, 0]

Image.fromarray(overlay).save(
    "test_c/ai_change_overlay.png"
)

print("\n========== ADAPTFORMER TEST C ==========")
print("AI mask saved: test_c/ai_change_mask.png")
print("AI overlay saved: test_c/ai_change_overlay.png")
print(f"IoU: {iou:.4f}")
print(f"Pixel Accuracy: {accuracy:.2f}%")
print("========================================")