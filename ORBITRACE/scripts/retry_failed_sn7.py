from __future__ import annotations

import csv
import json
import runpy
import time
from pathlib import Path

PROJECT_SCRIPT = Path(__file__).with_name("download_sn7_80_aoi.py")
MANIFEST = Path(r"C:\ORBITRACE_DATA\sn7_80_aoi\manifest.csv")
OUT_ROOT = Path(r"C:\ORBITRACE_DATA\sn7_80_aoi")

MAX_ROUNDS = 3
WAIT_BETWEEN_AOIS = 2


def load_downloader_namespace():
    if not PROJECT_SCRIPT.exists():
        raise FileNotFoundError(PROJECT_SCRIPT)
    return runpy.run_path(str(PROJECT_SCRIPT))


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    fields = [
        "split", "aoi", "t1_date", "t2_date",
        "t1_path", "t2_path",
        "t1_valid_fraction", "t2_valid_fraction",
        "width", "height", "bands", "crs",
        "temporal_gap_months", "status"
    ]
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    ns = load_downloader_namespace()
    process_aoi = ns["process_aoi"]
    rows = read_manifest()

    for round_no in range(1, MAX_ROUNDS + 1):
        failed_idx = [i for i, r in enumerate(rows) if r.get("status") != "OK"]
        if not failed_idx:
            break

        print(f"\n[ROUND {round_no}/{MAX_ROUNDS}] Retrying {len(failed_idx)} failed AOIs only")

        for n, idx in enumerate(failed_idx, 1):
            r = rows[idx]
            split, aoi = r["split"], r["aoi"]
            print(f"[{n:02d}/{len(failed_idx)}] {split}/{aoi}")

            try:
                new_row = process_aoi(split, aoi)
                rows[idx] = new_row
                print(f"      {new_row['status']} | T1={new_row['t1_date']} | T2={new_row['t2_date']}")
            except KeyboardInterrupt:
                write_manifest(rows)
                print("\n[STOP] Interrupted. Progress saved.")
                return 130
            except Exception as exc:
                rows[idx]["status"] = f"ERROR_{type(exc).__name__}: {exc}"
                print(f"      {rows[idx]['status']}")

            write_manifest(rows)
            time.sleep(WAIT_BETWEEN_AOIS)

        ok = sum(1 for r in rows if r.get("status") == "OK")
        print(f"[ROUND {round_no}] OK={ok}/80  Remaining={80-ok}")
        if ok == 80:
            break

        time.sleep(5)

    ok = sum(1 for r in rows if r.get("status") == "OK")
    errors = 80 - ok
    print(f"\n[DONE] OK={ok}/80  Errors={errors}")
    print(f"[DONE] Final selected GeoTIFF target={ok*2}")
    return 0 if ok == 80 else 3


if __name__ == "__main__":
    raise SystemExit(main())
