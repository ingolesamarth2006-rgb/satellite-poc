from pathlib import Path

import torch
from PIL import Image

from core.semantic_search import load_remoteclip
from core.change_detection_multiscale import detect_change_multiscale


ROOT = Path(__file__).resolve().parents[1]

LOCATIONS_DIR = ROOT / "integration_data" / "locations"
OUTPUT_DIR = ROOT / "integration_data" / "pipeline_results"

DEVICE = "cpu"


# =========================================================
# SEMANTIC SEARCH BETWEEN AVAILABLE LOCATIONS
# =========================================================

def search_locations(query, top_k=5):

    print("\n========== SEMANTIC SEARCH ==========")
    print(f"Query: {query}")

    model, preprocess, tokenizer = load_remoteclip()

    search_images = sorted(
        LOCATIONS_DIR.glob("*/search.png")
    )

    if not search_images:
        raise FileNotFoundError(
            "No search.png found inside integration_data/locations"
        )

    print(f"Locations Found: {len(search_images)}")

    text = tokenizer([query]).to(DEVICE)

    with torch.no_grad():

        text_features = model.encode_text(text)

        text_features = (
            text_features /
            text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        results = []

        for image_path in search_images:

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = (
                preprocess(image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            image_features = model.encode_image(
                image_tensor
            )

            image_features = (
                image_features /
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            similarity = (
                text_features @ image_features.T
            ).item()

            results.append({
                "location": image_path.parent.name,
                "score": similarity,
                "search_image": str(image_path)
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def analyze(query):

    print("\n")
    print("=" * 60)
    print("        SATELLITE ANALYSIS PIPELINE")
    print("=" * 60)

    # STEP 1 — Semantic search
    semantic_results = search_locations(
        query,
        top_k=5
    )

    print("\nSemantic Ranking:")

    for i, item in enumerate(
        semantic_results,
        start=1
    ):
        print(
            f"{i}. {item['location']} | "
            f"Score: {item['score']:.4f}"
        )

    # STEP 2 — Select best location
    best = semantic_results[0]

    location = best["location"]

    print(
        f"\nSelected Location: {location}"
    )

    location_dir = (
        LOCATIONS_DIR /
        location
    )

    t1_path = (
        location_dir /
        "T1.png"
    )

    t2_path = (
        location_dir /
        "T2.png"
    )

    if not t1_path.exists():
        raise FileNotFoundError(
            f"T1 missing: {t1_path}"
        )

    if not t2_path.exists():
        raise FileNotFoundError(
            f"T2 missing: {t2_path}"
        )

    print(f"T1: {t1_path}")
    print(f"T2: {t2_path}")

    # STEP 3 — Multi-scale change detection
    result_dir = (
        OUTPUT_DIR /
        location
    )

    change_result = detect_change_multiscale(
        str(t1_path),
        str(t2_path),
        str(result_dir),
        threshold=0.35
    )

    # STEP 4 — Final combined result
    final_result = {

        "query": query,

        "location": location,

        "semantic_score": best["score"],

        "t1_path": str(t1_path),

        "t2_path": str(t2_path),

        "change_mask": (
            change_result["mask_path"]
        ),

        "change_overlay": (
            change_result["overlay_path"]
        ),

        "change_probability": (
            change_result["probability_path"]
        ),

        "changed_percentage": (
            change_result["changed_percentage"]
        )
    }

    print("\n")
    print("=" * 60)
    print("                 FINAL RESULT")
    print("=" * 60)

    print(f"Query: {query}")
    print(f"Selected Location: {location}")

    print(
        f"Semantic Score: "
        f"{best['score']:.4f}"
    )

    print(
        f"Detected Change: "
        f"{change_result['changed_percentage']:.2f}%"
    )

    print(
        f"Change Mask: "
        f"{change_result['mask_path']}"
    )

    print(
        f"Overlay: "
        f"{change_result['overlay_path']}"
    )

    print("=" * 60)

    return final_result