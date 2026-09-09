from pathlib import Path
import csv
import json
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

DATA_ROOT = Path(r"C:\ORBITRACE_DATA\mini_archive")
PREVIEW_ROOT = Path(r"C:\ORBITRACE_DATA\mini_previews")
OUTPUT_JSON = Path(r"C:\ORBITRACE_DATA\mini_catalog.json")
OUTPUT_CSV = Path(r"C:\ORBITRACE_DATA\mini_catalog.csv")

PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)


def reverse_geocode(lat, lon):
    params = urllib.parse.urlencode({
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 14,
        "addressdetails": 1,
        "accept-language": "en"
    })

    url = f"https://nominatim.openstreetmap.org/reverse?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ORBITRACE/0.1"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        a = data.get("address", {})

        return {
            "display_name": data.get("display_name", ""),
            "village": a.get("village") or a.get("town") or "",
            "city": a.get("city") or a.get("municipality") or "",
            "state": a.get("state") or "",
            "country": a.get("country") or "",
            "country_code": a.get("country_code") or ""
        }

    except Exception as e:
        print("Geocoding warning:", e)
        return {
            "display_name": "",
            "village": "",
            "city": "",
            "state": "",
            "country": "",
            "country_code": ""
        }


catalog = []
location_cache = {}

tif_files = sorted(DATA_ROOT.rglob("*.tif"))

print("Found TIFFs:", len(tif_files))

for tif_path in tif_files:

    aoi_id = tif_path.parent.name
    date = tif_path.stem

    print("\nProcessing:", aoi_id, date)

    with rasterio.open(tif_path) as src:

        bounds4326 = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds
        )

        left, bottom, right, top = bounds4326

        latitude = (bottom + top) / 2
        longitude = (left + right) / 2

        # SpaceNet RGB preview
        arr = src.read([3, 2, 1])
        rgb = np.moveaxis(arr, 0, -1)

        preview_dir = PREVIEW_ROOT / aoi_id
        preview_dir.mkdir(parents=True, exist_ok=True)

        preview_path = preview_dir / f"{date}.png"
        Image.fromarray(rgb).save(preview_path)

        if aoi_id not in location_cache:
            location_cache[aoi_id] = reverse_geocode(
                latitude,
                longitude
            )

        location = location_cache[aoi_id]

        row = {
            "aoi_id": aoi_id,
            "date": date,
            "tif_path": str(tif_path),
            "preview_path": str(preview_path),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "latitude": latitude,
            "longitude": longitude,
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
            **location
        }

        catalog.append(row)


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=catalog[0].keys()
    )
    writer.writeheader()
    writer.writerows(catalog)


print("\n==============================")
print("CATALOG COMPLETE")
print("Images:", len(catalog))
print("AOIs:", len(set(x["aoi_id"] for x in catalog)))
print("JSON:", OUTPUT_JSON)
print("CSV :", OUTPUT_CSV)
print("Previews:", PREVIEW_ROOT)
print("==============================")
