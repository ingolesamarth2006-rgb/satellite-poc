import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from monai.networks.nets import DynUNet

AOI = "L15-1669E-1160N_6679_3549_13"

IMAGE_DIR = Path("dataset/spacenet7_test") / AOI / "images"
OUT_DIR = Path("integration_data/spacenet7_temporal")
PROB_DIR = OUT_DIR / "probabilities"
MASK_DIR = OUT_DIR / "masks"

PROB_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ---------------- MODEL ----------------

model = DynUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    kernel_size=[[3, 3]] * 5,
    strides=[[1, 1]] + [[2, 2]] * 4,
    upsample_kernel_size=[[2, 2]] * 4,
    norm_name=("INSTANCE", {"affine": True}),
    act_name=("leakyrelu", {
        "inplace": True,
        "negative_slope": 0.01
    }),
)

ckpt = torch.load(
    r"external\SpaceNet7-Buildings-Detection\trained_models\best_model.ckpt",
    map_location="cpu",
    weights_only=True,
)

state = {
    k[6:] if k.startswith("model.") else k: v
    for k, v in ckpt["state_dict"].items()
}

print(model.load_state_dict(state, strict=True))

model = model.to(DEVICE).eval()

mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)[:, None, None]

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)[:, None, None]

files = sorted(IMAGE_DIR.glob("*.tif"))

print("TOTAL IMAGES:", len(files))

# ---------------- INFERENCE ----------------

with torch.no_grad():

    for i, path in enumerate(files, start=1):

        match = re.search(r"monthly_(\d{4}_\d{2})_", path.name)

        if not match:
            print("SKIPPING:", path.name)
            continue

        date = match.group(1)

        with rasterio.open(path) as ds:
            arr = ds.read([1, 2, 3])

        max_value = np.iinfo(arr.dtype).max

        x = arr.astype(np.float32) / max_value
        x = (x - mean) / std

        tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

        logits = model(tensor)

        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = prob > 0.5

        # Preserve probability for temporal processing
        np.save(
            PROB_DIR / f"{date}.npy",
            prob.astype(np.float16)
        )

        # Preview mask
        Image.fromarray(
            (mask * 255).astype(np.uint8)
        ).save(
            MASK_DIR / f"{date}.png"
        )

        print(
            f"[{i:02d}/{len(files)}]",
            date,
            "| building area:",
            round(mask.mean() * 100, 2),
            "%",
            "| max prob:",
            round(float(prob.max()), 4)
        )

print("26-MONTH INFERENCE COMPLETE")
print("OUTPUT:", OUT_DIR)
