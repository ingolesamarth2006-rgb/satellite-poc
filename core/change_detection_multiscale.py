from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from core.change_detection import (
    load_adaptformer,
    load_image_pair
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Current best threshold
THRESHOLD = 0.35

# Scale 1 = context
SCALE_1_TILE = 512
SCALE_1_STRIDE = 384

# Scale 2 = small details
SCALE_2_TILE = 256
SCALE_2_STRIDE = 192


# =========================================================
# TILE START POSITIONS
# =========================================================

def get_tile_starts(length, tile_size, stride):

    if length <= tile_size:
        return [0]

    starts = list(
        range(
            0,
            length - tile_size + 1,
            stride
        )
    )

    last_start = length - tile_size

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


# =========================================================
# CREATE PROBABILITY MAP FOR ONE SCALE
# =========================================================

def probability_map_for_scale(
    t1,
    t2,
    processor,
    model,
    tile_size,
    stride
):

    width, height = t1.size

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    x_starts = get_tile_starts(
        width,
        tile_size,
        stride
    )

    y_starts = get_tile_starts(
        height,
        tile_size,
        stride
    )

    tile_count = 0

    for top in y_starts:

        for left in x_starts:

            right = min(
                left + tile_size,
                width
            )

            bottom = min(
                top + tile_size,
                height
            )

            tile_count += 1

            print(
                f"Tile {tile_count}: "
                f"({left}, {top}) -> "
                f"({right}, {bottom})"
            )

            # Same area from T1 and T2
            t1_tile = t1.crop(
                (left, top, right, bottom)
            )

            t2_tile = t2.crop(
                (left, top, right, bottom)
            )

            # Preprocess
            inputs = processor(
                images=(t1_tile, t2_tile),
                return_tensors="pt"
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            # Model inference
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits

            tile_width = right - left
            tile_height = bottom - top

            # Return output to original tile size
            logits = F.interpolate(
                logits,
                size=(
                    tile_height,
                    tile_width
                ),
                mode="bilinear",
                align_corners=False
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # Class 1 = change
            change_probability = (
                probabilities[0, 1]
                .cpu()
                .numpy()
            )

            # Overlap fusion
            probability_sum[
                top:bottom,
                left:right
            ] += change_probability

            prediction_count[
                top:bottom,
                left:right
            ] += 1

    prediction_count[
        prediction_count == 0
    ] = 1

    final_probability = (
        probability_sum /
        prediction_count
    )

    return final_probability, tile_count


# =========================================================
# MULTI-SCALE CHANGE DETECTION
# =========================================================

def detect_change_multiscale(
    t1_path,
    t2_path,
    output_dir,
    threshold=THRESHOLD
):

    print(
        "\n========== MULTI-SCALE CHANGE DETECTION =========="
    )

    print(f"Device: {DEVICE}")
    print(f"Threshold: {threshold}")

    # -----------------------------------------------------
    # LOAD MODEL ONCE
    # -----------------------------------------------------

    processor, model = load_adaptformer()

    # -----------------------------------------------------
    # LOAD IMAGES
    # -----------------------------------------------------

    t1, t2 = load_image_pair(
        t1_path,
        t2_path
    )

    if t1.size != t2.size:
        raise ValueError(
            "T1 and T2 must have same dimensions."
        )

    width, height = t1.size

    print(
        f"Image size: {width} x {height}"
    )

    # =====================================================
    # SCALE 1 — 512
    # =====================================================

    print(
        "\n--- SCALE 1: 512 x 512 ---"
    )

    prob_512, tiles_512 = probability_map_for_scale(
        t1,
        t2,
        processor,
        model,
        tile_size=SCALE_1_TILE,
        stride=SCALE_1_STRIDE
    )

    # =====================================================
    # SCALE 2 — 256
    # =====================================================

    print(
        "\n--- SCALE 2: 256 x 256 ---"
    )

    prob_256, tiles_256 = probability_map_for_scale(
        t1,
        t2,
        processor,
        model,
        tile_size=SCALE_2_TILE,
        stride=SCALE_2_STRIDE
    )

    # =====================================================
    # MULTI-SCALE FUSION
    # =====================================================

    print(
        "\nFusing 512 + 256 probability maps..."
    )

    # Start simple: equal importance
    final_probability = (
        prob_512 + prob_256
    ) / 2.0

    # =====================================================
    # THRESHOLD
    # =====================================================

    final_prediction = (
        final_probability >= threshold
    ).astype(np.uint8)

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # SAVE 512 PROBABILITY MAP
    # =====================================================

    prob_512_path = (
        output_dir /
        "probability_512.png"
    )

    Image.fromarray(
        np.clip(
            prob_512 * 255,
            0,
            255
        ).astype(np.uint8)
    ).save(prob_512_path)

    # =====================================================
    # SAVE 256 PROBABILITY MAP
    # =====================================================

    prob_256_path = (
        output_dir /
        "probability_256.png"
    )

    Image.fromarray(
        np.clip(
            prob_256 * 255,
            0,
            255
        ).astype(np.uint8)
    ).save(prob_256_path)

    # =====================================================
    # SAVE FINAL FUSED PROBABILITY MAP
    # =====================================================

    final_probability_path = (
        output_dir /
        "probability_multiscale.png"
    )

    Image.fromarray(
        np.clip(
            final_probability * 255,
            0,
            255
        ).astype(np.uint8)
    ).save(final_probability_path)

    # =====================================================
    # SAVE FINAL MASK
    # =====================================================

    mask = (
        final_prediction * 255
    ).astype(np.uint8)

    mask_path = (
        output_dir /
        "change_mask_multiscale.png"
    )

    Image.fromarray(
        mask
    ).save(mask_path)

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = np.array(t2).copy()

    changed_pixels = (
        final_prediction > 0
    )

    overlay[
        changed_pixels
    ] = [
        255,
        0,
        0
    ]

    overlay_path = (
        output_dir /
        "change_overlay_multiscale.png"
    )

    Image.fromarray(
        overlay
    ).save(overlay_path)

    # =====================================================
    # STATISTICS
    # =====================================================

    changed_pixel_count = int(
        changed_pixels.sum()
    )

    total_pixels = (
        final_prediction.size
    )

    changed_percentage = (
        changed_pixel_count /
        total_pixels
    ) * 100

    total_tiles = (
        tiles_512 +
        tiles_256
    )

    print(
        "\nMulti-scale detection completed."
    )

    print(
        f"512 Tiles: {tiles_512}"
    )

    print(
        f"256 Tiles: {tiles_256}"
    )

    print(
        f"Total Tiles: {total_tiles}"
    )

    print(
        f"Changed Area: "
        f"{changed_percentage:.2f}%"
    )

    print(
        f"Mask: {mask_path}"
    )

    print(
        f"Overlay: {overlay_path}"
    )

    print(
        "==================================================\n"
    )

    return {

        "t1_path": str(t1_path),

        "t2_path": str(t2_path),

        "mask_path": str(
            mask_path
        ),

        "overlay_path": str(
            overlay_path
        ),

        "probability_path": str(
            final_probability_path
        ),

        "probability_512_path": str(
            prob_512_path
        ),

        "probability_256_path": str(
            prob_256_path
        ),

        "changed_percentage": (
            changed_percentage
        ),

        "changed_pixels": (
            changed_pixel_count
        ),

        "tiles_512": tiles_512,

        "tiles_256": tiles_256,

        "total_tiles": total_tiles,

        "threshold": threshold
    }