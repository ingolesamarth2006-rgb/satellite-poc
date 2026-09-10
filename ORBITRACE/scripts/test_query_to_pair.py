from pathlib import Path
from collections import defaultdict
import sys
import json

import faiss
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.semantic_search import load_remoteclip

INDEX_PATH = Path(r"C:\ORBITRACE_DATA\faiss\mini_orbitrace.index")
META_PATH = Path(r"C:\ORBITRACE_DATA\faiss\mini_orbitrace_metadata.json")
CATALOG_PATH = Path(r"C:\ORBITRACE_DATA\mini_catalog.json")


def main():

    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "forest with winding roads"

    print("\n========== ORBITRACE QUERY -> TEMPORAL PAIR ==========")
    print("Query:", query)

    # Load FAISS + metadata
    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # RemoteCLIP
    model, preprocess, tokenizer = load_remoteclip()

    text = tokenizer([query])

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

    query_vector = text_features.cpu().numpy().astype("float32")

    # Best FAISS result
    scores, indices = index.search(query_vector, 1)

    best = metadata[int(indices[0][0])]
    score = float(scores[0][0])

    best_aoi = best["aoi_id"]

    print("\n========== BEST RETRIEVAL ==========")
    print("Score:", round(score, 4))
    print("AOI:", best_aoi)
    print("Matched Date:", best["date"])
    print(
        "Location:",
        best["city"],
        best["state"],
        best["country"]
    )

    # Find all observations for selected AOI
    observations = [
        item for item in catalog
        if item["aoi_id"] == best_aoi
    ]

    observations.sort(key=lambda x: x["date"])

    if len(observations) < 2:
        raise RuntimeError(
            f"AOI {best_aoi} does not have 2 temporal observations."
        )

    t1 = observations[0]
    t2 = observations[-1]

    print("\n========== AUTOMATIC TEMPORAL PAIR ==========")
    print("T1:", t1["date"])
    print("T1 Path:", t1["tif_path"])

    print("\nT2:", t2["date"])
    print("T2 Path:", t2["tif_path"])

    print("\nQUERY -> AOI -> T1/T2 COMPLETE")


if __name__ == "__main__":
    main()
