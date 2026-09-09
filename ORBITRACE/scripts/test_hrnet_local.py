from pathlib import Path
import json
import numpy as np
from PIL import Image

DEV_DIR = Path(r"C:\ORBITRACE_DATA\hrnet\development")
OUT_DIR = Path(r"C:\ORBITRACE_DATA\hrnet\local_validation\L15-1669E-1160N_6679_3549_13")

P2017 = DEV_DIR / "global_monthly_2017_07_mosaic_L15-1669E-1160N_6679_3549_13.npy"
P2019 = DEV_DIR / "global_monthly_2019_09_mosaic_L15-1669E-1160N_6679_3549_13.npy"

THRESHOLD = 0.35
MIN_SIZE = 100


def load_prob(path):
    arr = np.load(path)
    arr = np.squeeze(arr)
    return arr.astype(np.float32)


def save_gray(arr, path):
    arr = np.clip(arr, 0, 1)
    img = (arr * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_mask(mask, path):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(path)


def keep_large_components(mask, min_size=100):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    output = np.zeros((h, w), dtype=bool)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    ys, xs = np.where(mask)

    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue

        stack = [(y0, x0)]
        visited[y0, x0] = True
        component = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h and
                    0 <= nx < w and
                    mask[ny, nx] and
                    not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(component) >= min_size:
            for y, x in component:
                output[y, x] = True

    return output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading maps...")
    prob_2017 = load_prob(P2017)
    prob_2019 = load_prob(P2019)

    print("Shape 2017:", prob_2017.shape)
    print("Shape 2019:", prob_2019.shape)

    built_2017 = prob_2017 >= THRESHOLD
    built_2019 = prob_2019 >= THRESHOLD

    added_raw = built_2019 & (~built_2017)
    removed_raw = built_2017 & (~built_2019)

    added = keep_large_components(added_raw, MIN_SIZE)
    removed = keep_large_components(removed_raw, MIN_SIZE)

    save_gray(prob_2017, OUT_DIR / "probability_2017.png")
    save_gray(prob_2019, OUT_DIR / "probability_2019.png")
    save_mask(built_2017, OUT_DIR / "built_2017_mask.png")
    save_mask(built_2019, OUT_DIR / "built_2019_mask.png")
    save_mask(added, OUT_DIR / "added_buildings.png")
    save_mask(removed, OUT_DIR / "removed_buildings.png")

    base = np.clip(prob_2019 * 255, 0, 255).astype(np.uint8)
    overlay = np.stack([base, base, base], axis=-1)

    # Added = red
    overlay[added] = [255, 0, 0]

    # Removed = blue
    overlay[removed] = [0, 0, 255]

    Image.fromarray(overlay).save(OUT_DIR / "final_change_overlay.png")

    summary = {
        "aoi_id": "L15-1669E-1160N_6679_3549_13",
        "threshold": THRESHOLD,
        "min_size": MIN_SIZE,
        "added_pixels": int(added.sum()),
        "removed_pixels": int(removed.sum()),
        "probability_2017": str(P2017),
        "probability_2019": str(P2019),
        "output_dir": str(OUT_DIR),
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()