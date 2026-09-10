from pathlib import Path
import sys
import json
import math

import rasterio
from rasterio.transform import xy
from rasterio.warp import transform
import faiss
import numpy as np
import torch
from PIL import Image
from scipy import ndimage


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.semantic_search import load_remoteclip


INDEX_PATH = Path(r"C:\ORBITRACE_DATA\faiss\mini_orbitrace.index")
META_PATH = Path(r"C:\ORBITRACE_DATA\faiss\mini_orbitrace_metadata.json")
CATALOG_PATH = Path(r"C:\ORBITRACE_DATA\mini_catalog.json")

HRNET_ROOT = Path(r"C:\ORBITRACE_DATA\hrnet")
OUTPUT_ROOT = Path(r"C:\ORBITRACE_DATA\pipeline_results")

THRESHOLD = 0.35
MIN_SIZE = 100


# =========================================================
# HELPERS
# =========================================================

def keep_large_components(mask, min_size):
    """Keep only connected components with at least min_size pixels."""
    labels, _ = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    sizes = np.bincount(labels.ravel())

    keep = sizes >= min_size
    keep[0] = False

    return keep[labels]


def find_hrnet_map(aoi_id, date):
    """Find a cached HRNet probability map for a given AOI and date."""
    matches = list(
        HRNET_ROOT.rglob(
            f"*{date}*{aoi_id}*.npy"
        )
    )

    return matches[0] if matches else None


def get_hrnet_pixel_area_m2(tif_path, hrnet_shape):
    """
    Estimate the real ground area represented by one HRNet output pixel.

    The GeoTIFF may be in EPSG:3857, whose pixel area is not a true
    ground-area measurement. We therefore measure adjacent source pixels
    after transforming the AOI centre to its local UTM CRS, then account
    for the HRNet 3x output scale.
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise RuntimeError(
                f"GeoTIFF has no CRS, cannot calculate ground area: {tif_path}"
            )

        if len(hrnet_shape) != 2:
            raise ValueError(
                f"Expected a 2D HRNet map, got shape: {hrnet_shape}"
            )

        row = src.height // 2
        col = src.width // 2

        # Centre pixel
        x0, y0 = xy(
            src.transform,
            row,
            col,
            offset="center",
        )

        # Adjacent horizontal pixel
        xr, yr = xy(
            src.transform,
            row,
            col + 1,
            offset="center",
        )

        # Adjacent vertical pixel
        xd, yd = xy(
            src.transform,
            row + 1,
            col,
            offset="center",
        )

        # Resolve AOI centre to lon/lat so we can choose local UTM zone
        lon_arr, lat_arr = transform(
            src.crs,
            "EPSG:4326",
            [x0],
            [y0],
        )

        lon = lon_arr[0]
        lat = lat_arr[0]

        zone = int((lon + 180) // 6) + 1
        utm_epsg = (32600 if lat >= 0 else 32700) + zone

        # Measure source pixel dimensions in the local metric CRS
        xs, ys = transform(
            src.crs,
            f"EPSG:{utm_epsg}",
            [x0, xr, xd],
            [y0, yr, yd],
        )

        source_width_m = math.hypot(
            xs[1] - xs[0],
            ys[1] - ys[0],
        )

        source_height_m = math.hypot(
            xs[2] - xs[0],
            ys[2] - ys[0],
        )

        scale_x = hrnet_shape[1] / src.width
        scale_y = hrnet_shape[0] / src.height

        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("Invalid HRNet/source image scale.")

        hrnet_pixel_area_m2 = (
            source_width_m * source_height_m
        ) / (scale_x * scale_y)

        return hrnet_pixel_area_m2, utm_epsg


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "forest with winding roads"

    print("\n========== ORBITRACE FULL PIPELINE ==========")
    print("Query:", query)

    # -------------------------------------------------
    # LOAD SEARCH DATA
    # -------------------------------------------------

    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # -------------------------------------------------
    # SEMANTIC SEARCH
    # -------------------------------------------------

    model, _, tokenizer = load_remoteclip()

    text = tokenizer([query])

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = (
            text_features
            / text_features.norm(dim=-1, keepdim=True)
        )

    query_vector = (
        text_features
        .cpu()
        .numpy()
        .astype("float32")
    )

    scores, indices = index.search(query_vector, 1)

    best = metadata[int(indices[0][0])]
    score = float(scores[0][0])
    aoi_id = best["aoi_id"]

    print("\n========== RETRIEVAL ==========")
    print("Score:", round(score, 4))
    print("AOI:", aoi_id)
    print(
        "Location:",
        best["city"],
        best["state"],
        best["country"],
    )

    # -------------------------------------------------
    # TEMPORAL PAIRING
    # -------------------------------------------------

    observations = [
        item
        for item in catalog
        if item["aoi_id"] == aoi_id
    ]

    observations.sort(key=lambda item: item["date"])

    if len(observations) < 2:
        raise RuntimeError(
            f"AOI {aoi_id} does not have at least two temporal observations."
        )

    t1 = observations[0]
    t2 = observations[-1]

    print("\n========== TEMPORAL PAIR ==========")
    print("T1:", t1["date"])
    print("T1 Path:", t1["tif_path"])
    print("T2:", t2["date"])
    print("T2 Path:", t2["tif_path"])

    # -------------------------------------------------
    # FIND HRNET CACHE
    # -------------------------------------------------

    p1_path = find_hrnet_map(aoi_id, t1["date"])
    p2_path = find_hrnet_map(aoi_id, t2["date"])

    if p1_path is None or p2_path is None:
        print("\n========== HRNET CACHE MISS ==========")
        print("AOI:", aoi_id)
        print("No cached HRNet probability pair available.")
        print("Retrieval + temporal pairing still succeeded.")
        return

    print("\n========== HRNET CACHE ==========")
    print("T1 map:", p1_path)
    print("T2 map:", p2_path)

    # -------------------------------------------------
    # LOAD PROBABILITY MAPS
    # -------------------------------------------------

    prob1 = np.squeeze(np.load(p1_path)).astype(np.float32)
    prob2 = np.squeeze(np.load(p2_path)).astype(np.float32)

    if prob1.shape != prob2.shape:
        raise RuntimeError(
            f"HRNet map dimensions do not match: {prob1.shape} vs {prob2.shape}"
        )

    if prob1.ndim != 2:
        raise RuntimeError(
            f"Expected 2D probability maps, got shape: {prob1.shape}"
        )

    print("Map shape:", prob1.shape)

    # -------------------------------------------------
    # HRNET POST-PROCESSING
    # -------------------------------------------------

    built1 = prob1 >= THRESHOLD
    built2 = prob2 >= THRESHOLD

    added_raw = built2 & (~built1)
    removed_raw = built1 & (~built2)

    added = keep_large_components(added_raw, MIN_SIZE)
    removed = keep_large_components(removed_raw, MIN_SIZE)

    persistent = built1 & built2

    # -------------------------------------------------
    # PIXEL + PERCENTAGE METRICS
    # -------------------------------------------------

    total_pixels = prob1.size

    added_pixels = int(added.sum())
    removed_pixels = int(removed.sum())

    changed = added | removed
    changed_pixels = int(changed.sum())

    added_percent = (added_pixels / total_pixels) * 100
    removed_percent = (removed_pixels / total_pixels) * 100
    change_percent = (changed_pixels / total_pixels) * 100

    # -------------------------------------------------
    # REAL-WORLD AREA METRICS
    # -------------------------------------------------

    pixel_area_m2, utm_epsg = get_hrnet_pixel_area_m2(
        t2["tif_path"],
        prob2.shape,
    )

    added_area_m2 = added_pixels * pixel_area_m2
    removed_area_m2 = removed_pixels * pixel_area_m2
    total_change_area_m2 = changed_pixels * pixel_area_m2

    added_area_km2 = added_area_m2 / 1_000_000
    removed_area_km2 = removed_area_m2 / 1_000_000
    total_change_area_km2 = total_change_area_m2 / 1_000_000

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    output_dir = OUTPUT_ROOT / aoi_id
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = np.zeros(
        (prob1.shape[0], prob1.shape[1], 3),
        dtype=np.uint8,
    )

    # White = persistent buildings
    overlay[persistent] = [255, 255, 255]

    # Green = added buildings
    overlay[added] = [0, 255, 0]

    # Red = removed buildings
    overlay[removed] = [255, 0, 0]

    overlay_path = output_dir / "final_change_overlay.png"

    Image.fromarray(overlay).save(overlay_path)

    Image.fromarray(
        added.astype(np.uint8) * 255
    ).save(
        output_dir / "added_buildings.png"
    )

    Image.fromarray(
        removed.astype(np.uint8) * 255
    ).save(
        output_dir / "removed_buildings.png"
    )

    summary = {
        "query": query,
        "semantic_score": score,
        "aoi_id": aoi_id,
        "location": {
            "city": best["city"],
            "state": best["state"],
            "country": best["country"],
            "latitude": best.get("latitude"),
            "longitude": best.get("longitude"),
        },
        "t1": t1["date"],
        "t2": t2["date"],
        "threshold": THRESHOLD,
        "min_size": MIN_SIZE,
        "added_pixels": added_pixels,
        "removed_pixels": removed_pixels,
        "changed_pixels": changed_pixels,
        "added_percent": added_percent,
        "removed_percent": removed_percent,
        "change_percent": change_percent,
        "hrnet_pixel_area_m2": pixel_area_m2,
        "area_crs_epsg": utm_epsg,
        "added_area_m2": added_area_m2,
        "removed_area_m2": removed_area_m2,
        "total_change_area_m2": total_change_area_m2,
        "added_area_km2": added_area_km2,
        "removed_area_km2": removed_area_km2,
        "total_change_area_km2": total_change_area_km2,
        "overlay": str(overlay_path),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------

    print("\n========== CHANGE RESULT ==========")
    print("Added pixels:", added_pixels)
    print("Removed pixels:", removed_pixels)

    print("Added %:", round(added_percent, 3))
    print("Removed %:", round(removed_percent, 3))
    print("Total change %:", round(change_percent, 3))

    print("Added Area:", round(added_area_km2, 4), "km²")
    print("Removed Area:", round(removed_area_km2, 4), "km²")
    print("Total Change Area:", round(total_change_area_km2, 4), "km²")
    print("Area CRS: EPSG:", utm_epsg)

    print("Overlay:", overlay_path)

    print("\n========== PIPELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
