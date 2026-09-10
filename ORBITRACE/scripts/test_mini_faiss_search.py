from pathlib import Path
import sys
import json

import faiss
import torch


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# SAME WORKING REMOTECLIP LOADER
from core.semantic_search import load_remoteclip


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

INDEX_PATH = Path(
    r"C:\ORBITRACE_DATA\faiss\mini_orbitrace.index"
)

META_PATH = Path(
    r"C:\ORBITRACE_DATA\faiss\mini_orbitrace_metadata.json"
)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "forest with winding roads"

    print("\n========== ORBITRACE FAISS SEARCH ==========")
    print("Query:", query)

    # Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load metadata
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("FAISS vectors:", index.ntotal)

    # IMPORTANT:
    # Uses exact same RemoteCLIP loader that already worked
    model, preprocess, tokenizer = load_remoteclip()

    # Encode query
    text = tokenizer([query])

    with torch.no_grad():

        text_features = model.encode_text(text)

        text_features = (
            text_features /
            text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

    query_vector = (
        text_features
        .cpu()
        .numpy()
        .astype("float32")
    )

    # Search
    k = min(3, index.ntotal)

    scores, indices = index.search(
        query_vector,
        k
    )

    print("\n========== TOP RESULTS ==========")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        item = metadata[int(idx)]
        score = float(
            scores[0][rank - 1]
        )

        print(f"\n#{rank}")

        print(
            "Score:",
            round(score, 4)
        )

        print(
            "AOI:",
            item["aoi_id"]
        )

        print(
            "Date:",
            item["date"]
        )

        print(
            "Location:",
            item["city"],
            item["state"],
            item["country"]
        )

        print(
            "TIF:",
            item["tif_path"]
        )

    print(
        "\nFAISS SEARCH COMPLETE"
    )


if __name__ == "__main__":
    main()