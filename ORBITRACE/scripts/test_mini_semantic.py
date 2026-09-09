from pathlib import Path
import sys
import json
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.semantic_search import load_remoteclip

CATALOG = Path(r"C:\ORBITRACE_DATA\mini_catalog.json")

query = sys.argv[1] if len(sys.argv) > 1 else "forest with winding roads"

print("Loading mini catalog...")

with open(CATALOG, "r", encoding="utf-8") as f:
    catalog = json.load(f)

model, preprocess, tokenizer = load_remoteclip()

device = "cpu"

print(f"Query: {query}")
print(f"Images: {len(catalog)}")

text = tokenizer([query]).to(device)

with torch.no_grad():

    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    results = []

    for item in catalog:

        image = Image.open(item["preview_path"]).convert("RGB")

        image_tensor = (
            preprocess(image)
            .unsqueeze(0)
            .to(device)
        )

        image_features = model.encode_image(image_tensor)

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        score = (
            text_features @ image_features.T
        ).item()

        results.append({
            "score": score,
            **item
        })


results.sort(
    key=lambda x: x["score"],
    reverse=True
)

print("\n==============================")
print("TOP SEMANTIC RESULTS")
print("==============================")

for i, r in enumerate(results, start=1):

    print(f"\n#{i}")
    print("Score    :", round(r["score"], 4))
    print("AOI      :", r["aoi_id"])
    print("Date     :", r["date"])
    print(
        "Location :",
        r["city"],
        r["state"],
        r["country"]
    )

