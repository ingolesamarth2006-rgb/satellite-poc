import json
from pathlib import Path
from collections import defaultdict

CATALOG = Path(r"C:\ORBITRACE_DATA\mini_catalog.json")

with open(CATALOG, "r", encoding="utf-8") as f:
    catalog = json.load(f)

groups = defaultdict(list)

for item in catalog:
    groups[item["aoi_id"]].append(item)

print("\n========== ORBITRACE TEMPORAL PAIRS ==========")

for aoi_id, observations in groups.items():
    observations.sort(key=lambda x: x["date"])

    t1 = observations[0]
    t2 = observations[-1]

    print("\nAOI:", aoi_id)
    print("Location:", t1["city"], t1["state"], t1["country"])
    print("T1:", t1["date"])
    print("T2:", t2["date"])
    print("T1 path:", t1["tif_path"])
    print("T2 path:", t2["tif_path"])
