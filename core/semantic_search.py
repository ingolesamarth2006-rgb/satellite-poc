from pathlib import Path

import torch
import open_clip
from PIL import Image


# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = next(
    (ROOT / "checkpoints").rglob("RemoteCLIP-RN50.pt")
)
DATASET_DIR = ROOT / "dataset" / "semantic"


# CPU abhi enough hai
DEVICE = "cpu"


def load_remoteclip():
    print("Loading RemoteCLIP...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50",
        pretrained=None
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Some checkpoints directly contain weights,
    # some may contain them inside state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    tokenizer = open_clip.get_tokenizer("RN50")

    print("RemoteCLIP loaded successfully.")

    return model, preprocess, tokenizer


def get_image_paths():
    image_paths = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(
            DATASET_DIR.rglob(extension)
        )

    return sorted(image_paths)


def semantic_search(query, top_k=5):
    model, preprocess, tokenizer = load_remoteclip()

    image_paths = get_image_paths()

    print(f"Images found: {len(image_paths)}")
    print(f"Query: {query}")

    text = tokenizer([query]).to(DEVICE)

    with torch.no_grad():

        text_features = model.encode_text(text)

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        results = []

        for image_path in image_paths:

            image = Image.open(image_path).convert("RGB")

            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            image_features = model.encode_image(image_tensor)

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )

            similarity = (
                text_features @ image_features.T
            ).item()

            results.append(
                {
                    "path": str(image_path),
                    "category": image_path.parent.name,
                    "score": similarity
                }
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]