from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "deepang/adaptformer-LEVIR-CD"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_adaptformer():

    print("Loading AdaptFormer...")
    print(f"Device: {DEVICE}")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = model.to(DEVICE)
    model.eval()

    print("AdaptFormer loaded successfully.")

    return processor, model


def load_image_pair(t1_path, t2_path):

    t1_path = Path(t1_path)
    t2_path = Path(t2_path)

    if not t1_path.exists():
        raise FileNotFoundError(
            f"T1 image not found: {t1_path}"
        )

    if not t2_path.exists():
        raise FileNotFoundError(
            f"T2 image not found: {t2_path}"
        )

    t1 = Image.open(t1_path).convert("RGB")
    t2 = Image.open(t2_path).convert("RGB")

    return t1, t2