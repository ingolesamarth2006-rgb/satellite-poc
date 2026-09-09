from pathlib import Path
import json

import faiss
import numpy as np
import torch
from PIL import Image

from core.semantic_search import load_remoteclip


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "dataset" / "semantic"

FAISS_DIR = ROOT / "integration_data" / "faiss"

INDEX_PATH = FAISS_DIR / "semantic.index"
METADATA_PATH = FAISS_DIR / "metadata.json"

DEVICE = "cpu"


def build_faiss_index():

    print("\n========== BUILDING FAISS INDEX ==========")

    # Load RemoteCLIP
    model, preprocess, tokenizer = load_remoteclip()

    image_paths = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(
            DATASET_DIR.rglob(extension)
        )

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(
            "No semantic images found."
        )

    print(f"Images found: {len(image_paths)}")

    embeddings = []
    metadata = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = (
                preprocess(image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                features = model.encode_image(
                    image_tensor
                )

                features = (
                    features /
                    features.norm(
                        dim=-1,
                        keepdim=True
                    )
                )

            vector = (
                features
                .cpu()
                .numpy()
                .astype("float32")[0]
            )

            embeddings.append(vector)

            metadata.append({
                "path": str(
                    image_path.relative_to(ROOT)
                ),
                "category": image_path.parent.name
            })

            if index % 50 == 0:
                print(
                    f"Processed: "
                    f"{index}/{len(image_paths)}"
                )

        except Exception as error:

            print(
                f"Skipped {image_path}: {error}"
            )


    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    print(
        f"\nEmbedding shape: {vectors.shape}"
    )

    # Normalized embeddings +
    # Inner Product = cosine similarity
    dimension = vectors.shape[1]

    faiss_index = faiss.IndexFlatIP(
        dimension
    )

    faiss_index.add(
        vectors
    )

    FAISS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        faiss_index,
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


    print("\n========== FAISS INDEX READY ==========")

    print(
        f"Vectors stored: {faiss_index.ntotal}"
    )

    print(
        f"Index: {INDEX_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )

    return faiss_index.ntotal


if __name__ == "__main__":

    total = build_faiss_index()

    print(
        f"\nFAISS BUILD SUCCESS: "
        f"{total} images indexed"
    )