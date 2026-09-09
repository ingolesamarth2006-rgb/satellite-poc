import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from PIL import Image

AOI = "L15-1669E-1160N_6679_3549_13"

base = Path("dataset/spacenet7_test") / AOI
out = Path("integration_data/spacenet7_validation")
out.mkdir(parents=True, exist_ok=True)

for date in ["2017_07", "2019_09"]:

    tif = base / "images" / f"global_monthly_{date}_mosaic_{AOI}.tif"
    geojson = base / "labels" / f"global_monthly_{date}_mosaic_{AOI}_Buildings.geojson"

    with rasterio.open(tif) as ds:
        with open(geojson, "r") as f:
            data = json.load(f)

        geometries = []
        skipped = 0

        for feature in data["features"]:
            geom = feature.get("geometry")

            # Skip null / invalid geometry records
            if not geom:
                skipped += 1
                continue

            try:
                geom_3857 = transform_geom(
                    "OGC:CRS84",
                    ds.crs,
                    geom,
                    precision=6
                )

                geometries.append((geom_3857, 1))

            except Exception:
                skipped += 1

        mask = rasterize(
            geometries,
            out_shape=(ds.height, ds.width),
            transform=ds.transform,
            fill=0,
            dtype=np.uint8
        )

    Image.fromarray(mask * 255).save(
        out / f"gt_buildings_{date}.png"
    )

    print(
        date,
        "GT BUILDING AREA:",
        round(mask.mean() * 100, 2),
        "%",
        "| valid geometries:",
        len(geometries),
        "| skipped:",
        skipped
    )

print("GT MASKS CREATED")
