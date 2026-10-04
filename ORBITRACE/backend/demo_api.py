from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import open_clip
import torch
import rasterio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rasterio.transform import xy
from rasterio.warp import transform, transform_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "ORBITRACE_DATA"

INDEX_PATH = DATA_ROOT / "faiss" / "sn7_160_orbitrace.index"
META_PATH = DATA_ROOT / "faiss" / "sn7_160_orbitrace_metadata.json"
PREVIEW_ROOT = DATA_ROOT / "sn7_80_previews"
RAPID_ROOT = DATA_ROOT / "rapid_change_all"
PIPELINE_RESULTS_ROOT = DATA_ROOT / "pipeline_results"
LOCATION_CACHE_PATH = DATA_ROOT / "faiss" / "sn7_80_locations.json"

MODEL_PATH = next((REPO_ROOT / "checkpoints").rglob("RemoteCLIP-RN50.pt"), None)

app = FastAPI(title="ORBITRACE Demo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if PREVIEW_ROOT.exists():
    app.mount("/preview-assets", StaticFiles(directory=str(PREVIEW_ROOT)), name="preview-assets")
if RAPID_ROOT.exists():
    app.mount("/rapid-assets", StaticFiles(directory=str(RAPID_ROOT)), name="rapid-assets")
if PIPELINE_RESULTS_ROOT.exists():
    app.mount("/pipeline-assets", StaticFiles(directory=str(PIPELINE_RESULTS_ROOT)), name="pipeline-assets")

INDEX = None
META = None
MODEL = None
TOKENIZER = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def _load_once():
    global INDEX, META, MODEL, TOKENIZER

    if INDEX is None:
        if not INDEX_PATH.exists():
            raise RuntimeError(f"Missing FAISS index: {INDEX_PATH}")
        INDEX = faiss.read_index(str(INDEX_PATH))

    if META is None:
        if not META_PATH.exists():
            raise RuntimeError(f"Missing metadata: {META_PATH}")
        META = json.loads(META_PATH.read_text(encoding="utf-8"))

    if MODEL is None:
        if MODEL_PATH is None or not MODEL_PATH.exists():
            raise RuntimeError("RemoteCLIP-RN50.pt not found under checkpoints")

        model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained=None)
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in ckpt.items()
            }
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        MODEL = model
        TOKENIZER = open_clip.get_tokenizer("RN50")


def _scene_preview_url(rec: dict[str, Any]) -> str | None:
    name = f'{rec["aoi_id"]}__{rec["temporal_role"]}__{rec["date"]}.jpg'
    path = PREVIEW_ROOT / name
    return f"/preview-assets/{name}" if path.exists() else None


def _ground_pixel_area_m2(rec: dict[str, Any]) -> tuple[float | None, int | None]:
    """Estimate one source pixel's real ground area in the AOI's local UTM CRS."""
    try:
        tif_path = Path(rec.get("path", ""))
        if not tif_path.exists():
            return None, None

        with rasterio.open(tif_path) as src:
            if src.crs is None or src.width < 2 or src.height < 2:
                return None, None

            row = src.height // 2
            col = src.width // 2
            x0, y0 = xy(src.transform, row, col, offset="center")
            xr, yr = xy(src.transform, row, col + 1, offset="center")
            xd, yd = xy(src.transform, row + 1, col, offset="center")

            lon_arr, lat_arr = transform(src.crs, "EPSG:4326", [x0], [y0])
            lon = float(lon_arr[0])
            lat = float(lat_arr[0])
            zone = max(1, min(60, int((lon + 180) // 6) + 1))
            utm_epsg = (32600 if lat >= 0 else 32700) + zone

            xs, ys = transform(
                src.crs,
                f"EPSG:{utm_epsg}",
                [x0, xr, xd],
                [y0, yr, yd],
            )
            width_m = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
            height_m = math.hypot(xs[2] - xs[0], ys[2] - ys[0])
            area = width_m * height_m
            return (float(area), int(utm_epsg)) if area > 0 else (None, int(utm_epsg))
    except Exception:
        return None, None


def _pipeline_enrichment(aoi: str) -> dict[str, Any] | None:
    """Read richer saved pipeline output when it already exists for this AOI."""
    path = PIPELINE_RESULTS_ROOT / aoi / "summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ci = data.get("change_interpretation")
        if isinstance(ci, str):
            change_type = ci
            direction = None
            pattern = None
        elif isinstance(ci, dict):
            change_type = ci.get("change_type") or ci.get("type") or ci.get("label")
            direction = ci.get("direction")
            pattern = ci.get("pattern")
        else:
            change_type = None
            direction = None
            pattern = None

        return {
            "available": True,
            "location": data.get("location"),
            "change_type": change_type,
            "direction": direction or data.get("direction"),
            "pattern": pattern or data.get("pattern"),
            "change_percent": data.get("change_percent"),
            "total_change_area_km2": data.get("total_change_area_km2"),
            "added_area_km2": data.get("added_area_km2"),
            "removed_area_km2": data.get("removed_area_km2"),
            "added_percent": data.get("added_percent"),
            "removed_percent": data.get("removed_percent"),
            "reliability": data.get("reliability"),
            "area_crs_epsg": data.get("area_crs_epsg"),
            "threshold": data.get("threshold"),
            "min_size": data.get("min_size"),
        }
    except Exception:
        return None


def _load_location_cache() -> dict[str, Any]:
    try:
        if LOCATION_CACHE_PATH.exists():
            data = json.loads(LOCATION_CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _location_info(aoi: str, lat: float | None, lon: float | None, rich: dict[str, Any] | None) -> dict[str, Any]:
    if rich and isinstance(rich.get("location"), dict):
        loc = rich["location"]
        city = loc.get("city")
        state = loc.get("state")
        country = loc.get("country")
        if any([city, state, country]):
            label = ", ".join(str(x) for x in [city, state, country] if x)
            return {
                "city": city,
                "state": state,
                "country": country,
                "label": label,
                "source": "validated_pipeline",
            }

    cached = _load_location_cache().get(aoi)
    if isinstance(cached, dict):
        city = cached.get("city")
        state = cached.get("state")
        country = cached.get("country")
        label = cached.get("label") or ", ".join(str(x) for x in [city, state, country] if x)
        if label:
            return {
                "city": city,
                "state": state,
                "country": country,
                "label": label,
                "source": "local_reverse_geocode_cache",
            }

    coord_label = (
        f"{lat:.5f}, {lon:.5f}"
        if lat is not None and lon is not None
        else "Unknown location"
    )
    return {
        "city": None,
        "state": None,
        "country": None,
        "label": coord_label,
        "source": "coordinates_only",
    }


def _rapid_result(rec: dict[str, Any]) -> dict[str, Any] | None:
    split = rec["split"]
    aoi = rec["aoi_id"]
    path = RAPID_ROOT / split / aoi / "result.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    base = f"/rapid-assets/{split}/{aoi}"

    changed_pixels = data.get("changed_pixels")
    pixel_area_m2, area_crs_epsg = _ground_pixel_area_m2(rec)
    changed_area_m2 = None
    if pixel_area_m2 is not None and changed_pixels is not None:
        try:
            changed_area_m2 = float(changed_pixels) * float(pixel_area_m2)
        except (TypeError, ValueError):
            changed_area_m2 = None

    return {
        "analysis_mode": data.get("analysis_mode", "RAPID_CHANGE_SCREENING"),
        "change_type": "GENERAL_SURFACE_CHANGE",
        "change_percent": data.get("change_percent"),
        "change_level": data.get("change_level"),
        "changed_pixels": changed_pixels,
        "stable_pixels": data.get("stable_pixels"),
        "pixel_area_m2": round(pixel_area_m2, 6) if pixel_area_m2 is not None else None,
        "area_crs_epsg": area_crs_epsg,
        "changed_area_m2": round(changed_area_m2, 3) if changed_area_m2 is not None else None,
        "changed_area_km2": round(changed_area_m2 / 1_000_000, 6) if changed_area_m2 is not None else None,
        "changed_area_hectares": round(changed_area_m2 / 10_000, 3) if changed_area_m2 is not None else None,
        "interpretation": data.get("interpretation"),
        "building_specific": data.get("building_specific", False),
        "t1_date": data.get("t1_date"),
        "t2_date": data.get("t2_date"),
        "t1_preview_url": f"{base}/t1_preview.jpg",
        "t2_preview_url": f"{base}/t2_preview.jpg",
        "overlay_url": f"{base}/change_overlay.png",
        "heatmap_url": f"{base}/change_heatmap.png",
        "heatmap_meaning": "Normalized T1/T2 pixel-difference intensity; brighter pixels mean stronger visual difference. It is not confidence.",
        "note": data.get("note"),
    }


def _center_latlon(rec: dict[str, Any]) -> tuple[float | None, float | None]:
    try:
        bounds = rec.get("bounds")
        crs = rec.get("crs")
        if not bounds or not crs:
            return None, None
        left, bottom, right, top = transform_bounds(crs, "EPSG:4326", *bounds)
        return round((bottom + top) / 2.0, 6), round((left + right) / 2.0, 6)
    except Exception:
        return None, None


def _unique_aoi_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    _load_once()

    q = query.strip()
    if len(q) < 2:
        raise HTTPException(status_code=422, detail="Query too short")

    top_k = max(1, min(int(top_k), 10))

    with torch.inference_mode():
        text = MODEL.encode_text(TOKENIZER([q]))
        text = text / text.norm(dim=-1, keepdim=True)

    # Search more scene rows, then collapse T1/T2 duplicates to unique AOIs.
    raw_k = min(max(top_k * 8, 40), int(INDEX.ntotal))
    D, I = INDEX.search(text.cpu().numpy().astype("float32"), raw_k)

    results = []
    seen = set()

    for score, idx in zip(D[0], I[0]):
        rec = META["records"][int(idx)]
        aoi = rec["aoi_id"]
        if aoi in seen:
            continue
        seen.add(aoi)

        lat, lon = _center_latlon(rec)
        rapid = _rapid_result(rec)
        rich = _pipeline_enrichment(aoi)
        location = _location_info(aoi, lat, lon, rich)

        results.append(
            {
                "rank": len(results) + 1,
                "score": round(float(score), 4),
                "aoi_id": aoi,
                "split": rec["split"],
                "matched_scene": {
                    "temporal_role": rec["temporal_role"],
                    "date": rec["date"],
                    "preview_url": _scene_preview_url(rec),
                },
                "coordinates": {"lat": lat, "lon": lon},
                "location": location,
                "location_label": location.get("label"),
                "analysis_ready": rapid is not None,
                "analysis": rapid,
                "validated_pipeline": rich,
            }
        )

        if len(results) >= top_k:
            break

    return results


@app.get("/")
def root():
    return {
        "system": "ORBITRACE",
        "status": "ready",
        "semantic_index": "RemoteCLIP RN50 + FAISS",
        "archive": "80 AOIs / 160 observations",
        "fallback_analysis": "precomputed rapid T1/T2 change screening",
    }


@app.get("/api/v1/demo/status")
def status():
    try:
        _load_once()
        rapid_count = len(list(RAPID_ROOT.rglob("result.json"))) if RAPID_ROOT.exists() else 0
        return {
            "ready": True,
            "faiss_vectors": int(INDEX.ntotal),
            "metadata_records": len(META["records"]),
            "rapid_change_results": rapid_count,
            "location_cache_ready": LOCATION_CACHE_PATH.exists(),
            "remoteclip_checkpoint": str(MODEL_PATH),
        }
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/v1/demo/search")
def search(body: SearchRequest):
    return {
        "query": body.query,
        "results": _unique_aoi_search(body.query, body.top_k),
    }


@app.get("/api/v1/demo/search")
def search_get(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=10),
):
    return {
        "query": q,
        "results": _unique_aoi_search(q, top_k),
    }
