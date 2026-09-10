import json
from pathlib import Path

import faiss
import numpy as np
import torch
import open_clip

INDEX_PATH = Path(r"C:\ORBITRACE_DATA\faiss\mini_orbitrace.index")
META_PATH = Path(r"C:\ORBITRACE_DATA\faiss\mini_orbitrace_metadata.json")
MODEL_PATH = next((Path(__file__).resolve().parents[1] / "checkpoints").glob("RemoteCLIP-RN50*.pt"))

DEVICE = "cpu"


def load_remoteclip():
    print("Loading RemoteCLIP...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50",
        pretrained=None
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    tokenizer = open_clip.get_tokenizer("RN50")

    print("RemoteCLIP loaded.")
    return model, tokenizer


def encode_text(query, model, tokenizer):
    text = tokenizer([query]).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype("float32")


def main():
    query = input("Enter query: ").strip()

    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model, tokenizer = load_remoteclip()
    qvec = encode_text(query, model, tokenizer)

    distances, indices = index.search(qvec, k=3)

    print("\nTop results:\n")
    for rank, idx in enumerate(indices[0], start=1):
        item = metadata[idx]
        print(f"[{rank}] score={distances[0][rank-1]:.4f}")
        print("AOI:", item["aoi_id"])
        print("Date:", item["date"])
        print("City:", item["city"])
        print("State:", item["state"])
        print("Country:", item["country"])
        print("Path:", item["tif_path"])
        print("-" * 50)


if __name__ == "__main__":
    main()