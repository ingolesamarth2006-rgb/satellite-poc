from pathlib import Path
import numpy as np
from PIL import Image

PROB_DIR = Path("integration_data/spacenet7_temporal/probabilities")
OUT = Path("integration_data/spacenet7_temporal/change")
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(PROB_DIR.glob("*.npy"))
dates = [f.stem for f in files]

stack = np.stack([np.load(f).astype(np.float32) for f in files])

print("STACK:", stack.shape)
print("DATES:", dates[0], "to", dates[-1])

# Stable early and late references
early = np.median(stack[:5], axis=0)
late  = np.median(stack[-5:], axis=0)

# New building:
# low confidence in early period,
# strong confidence in late period
new_score = late - early

new_change = (
    (early < 0.35) &
    (late > 0.60) &
    (new_score > 0.30)
)

Image.fromarray(
    (new_change * 255).astype(np.uint8)
).save(OUT / "temporal_new_buildings.png")

Image.fromarray(
    (np.clip(new_score,0,1) * 255).astype(np.uint8)
).save(OUT / "temporal_change_score.png")

# Find earliest supported observation.
# Require detection in 3 consecutive observations.
first_idx = np.full(new_change.shape, -1, dtype=np.int16)

for i in range(len(stack)-2):

    persistent = (
        (stack[i] > 0.50) &
        (stack[i+1] > 0.50) &
        (stack[i+2] > 0.50) &
        new_change
    )

    assign = persistent & (first_idx == -1)
    first_idx[assign] = i

valid = first_idx >= 0

if valid.any():
    vals, counts = np.unique(first_idx[valid], return_counts=True)
    dominant = vals[np.argmax(counts)]

    print("DOMINANT EARLIEST SUPPORTED CHANGE:", dates[int(dominant)])

    for idx in vals:
        mask = first_idx == idx
        if mask.sum() > 50:
            print(
                dates[int(idx)],
                "| change pixels:",
                int(mask.sum())
            )

print("TEMPORAL NEW AREA %:", round(new_change.mean()*100,2))
print("TEMPORAL CHANGE COMPLETE")
print("OUTPUT:", OUT)
