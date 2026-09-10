from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio

S3_ROOT = "s3://spacenet-dataset/spacenet/SN7_buildings"
OUT_ROOT = Path(r"C:\ORBITRACE_DATA\sn7_80_aoi")
CANDIDATES_PER_SIDE = 3
MIN_MONTH_GAP = 12
MAX_RETRIES = 3

DATE_RE = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic_(.+)\.tif$")


@dataclass
class RemoteImage:
    split: str
    aoi: str
    filename: str
    date: str
    s3_uri: str
    size_bytes: int


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def aws_ls(uri: str) -> List[str]:
    cp = run(["aws", "s3", "ls", uri, "--no-sign-request"])
    return [ln.rstrip() for ln in cp.stdout.splitlines() if ln.strip()]


def list_aois(split: str) -> List[str]:
    lines = aws_ls(f"{S3_ROOT}/{split}/")
    aois = []
    for ln in lines:
        parts = ln.split()
        if parts and parts[0] == "PRE":
            aois.append(parts[-1].rstrip("/"))
    return sorted(aois)


def parse_remote_images(split: str, aoi: str) -> List[RemoteImage]:
    uri = f"{S3_ROOT}/{split}/{aoi}/images_masked/"
    lines = aws_ls(uri)
    items: List[RemoteImage] = []

    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        filename = parts[-1]
        m = DATE_RE.match(filename)
        if not m:
            continue
        yyyy, mm, _ = m.groups()
        date = f"{yyyy}_{mm}"
        items.append(
            RemoteImage(
                split=split,
                aoi=aoi,
                filename=filename,
                date=date,
                s3_uri=f"{uri}{filename}",
                size_bytes=size,
            )
        )
    return sorted(items, key=lambda x: x.date)


def month_index(date: str) -> int:
    y, m = map(int, date.split("_"))
    return y * 12 + (m - 1)


def download_with_retry(uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cp = run(["aws", "s3", "cp", uri, str(dest), "--no-sign-request"])
            if dest.exists() and dest.stat().st_size > 0:
                return
        except subprocess.CalledProcessError as exc:
            last_err = exc.stderr or str(exc)
        time.sleep(attempt * 2)

    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {uri}\n{last_err}")


def raster_quality(path: Path) -> Tuple[float, int, int, int, str]:
    """
    Conservative usability score:
    valid_fraction = fraction of pixels where at least one band is non-zero and finite.
    This does NOT claim cloud detection; it only detects heavily masked/empty imagery.
    """
    with rasterio.open(path) as src:
        arr = src.read(masked=False)
        finite = np.isfinite(arr).all(axis=0)
        nonzero = np.any(arr != 0, axis=0)
        valid = finite & nonzero
        valid_fraction = float(valid.mean())
        return valid_fraction, src.width, src.height, src.count, str(src.crs)


def choose_best(
    candidates: List[RemoteImage],
    temp_dir: Path,
    prefer: str,
) -> Tuple[RemoteImage, Path, float, Tuple[int, int, int, str]]:
    """Choose highest-quality candidate with deterministic date tie-break.

    prefer='earliest' -> for T1, equal-quality ties choose the earliest date.
    prefer='latest'   -> for T2, equal-quality ties choose the latest date.
    """
    scored = []

    for item in candidates:
        local = temp_dir / item.filename
        download_with_retry(item.s3_uri, local)
        try:
            valid_fraction, w, h, bands, crs = raster_quality(local)
        except Exception:
            valid_fraction, w, h, bands, crs = -1.0, 0, 0, 0, ""
        scored.append((valid_fraction, item, local, (w, h, bands, crs)))

    if prefer == "earliest":
        scored.sort(key=lambda x: (-x[0], x[1].date))
    elif prefer == "latest":
        scored.sort(key=lambda x: (-x[0], x[1].date), reverse=False)
        # Among equal quality, newest date first.
        best_quality = scored[0][0]
        tied = [x for x in scored if x[0] == best_quality]
        tied.sort(key=lambda x: x[1].date, reverse=True)
        best = tied[0]
        return best[1], best[2], best[0], best[3]
    else:
        raise ValueError("prefer must be 'earliest' or 'latest'")

    best = scored[0]
    return best[1], best[2], best[0], best[3]


def save_manifest(rows: List[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    json_path = OUT_ROOT / "manifest.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = OUT_ROOT / "manifest.csv"
    fields = [
        "split", "aoi", "t1_date", "t2_date",
        "t1_path", "t2_path",
        "t1_valid_fraction", "t2_valid_fraction",
        "width", "height", "bands", "crs",
        "temporal_gap_months", "status"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_aoi(split: str, aoi: str) -> dict:
    images = parse_remote_images(split, aoi)
    if len(images) < 2:
        return {
            "split": split, "aoi": aoi, "t1_date": "", "t2_date": "",
            "t1_path": "", "t2_path": "",
            "t1_valid_fraction": "", "t2_valid_fraction": "",
            "width": "", "height": "", "bands": "", "crs": "",
            "temporal_gap_months": "", "status": "ERROR_NOT_ENOUGH_IMAGES"
        }

    early_candidates = images[: min(CANDIDATES_PER_SIDE, len(images))]
    late_candidates = images[-min(CANDIDATES_PER_SIDE, len(images)):]

    aoi_dir = OUT_ROOT / split / aoi
    temp_dir = aoi_dir / "_candidates"
    temp_dir.mkdir(parents=True, exist_ok=True)

    t1, t1_local, q1, meta1 = choose_best(early_candidates, temp_dir, prefer="earliest")
    t2, t2_local, q2, meta2 = choose_best(late_candidates, temp_dir, prefer="latest")

    gap = month_index(t2.date) - month_index(t1.date)
    if gap < MIN_MONTH_GAP:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "split": split, "aoi": aoi, "t1_date": t1.date, "t2_date": t2.date,
            "t1_path": "", "t2_path": "",
            "t1_valid_fraction": round(q1, 6), "t2_valid_fraction": round(q2, 6),
            "width": meta1[0], "height": meta1[1], "bands": meta1[2], "crs": meta1[3],
            "temporal_gap_months": gap, "status": "ERROR_TEMPORAL_GAP_TOO_SMALL"
        }

    if meta1 != meta2:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "split": split, "aoi": aoi, "t1_date": t1.date, "t2_date": t2.date,
            "t1_path": "", "t2_path": "",
            "t1_valid_fraction": round(q1, 6), "t2_valid_fraction": round(q2, 6),
            "width": meta1[0], "height": meta1[1], "bands": meta1[2], "crs": meta1[3],
            "temporal_gap_months": gap, "status": "ERROR_RASTER_METADATA_MISMATCH"
        }

    final_t1 = aoi_dir / f"T1_{t1.date}.tif"
    final_t2 = aoi_dir / f"T2_{t2.date}.tif"

    if final_t1.exists():
        final_t1.unlink()
    if final_t2.exists():
        final_t2.unlink()

    shutil.move(str(t1_local), str(final_t1))
    shutil.move(str(t2_local), str(final_t2))
    shutil.rmtree(temp_dir, ignore_errors=True)

    w, h, bands, crs = meta1
    return {
        "split": split,
        "aoi": aoi,
        "t1_date": t1.date,
        "t2_date": t2.date,
        "t1_path": str(final_t1),
        "t2_path": str(final_t2),
        "t1_valid_fraction": round(q1, 6),
        "t2_valid_fraction": round(q2, 6),
        "width": w,
        "height": h,
        "bands": bands,
        "crs": crs,
        "temporal_gap_months": gap,
        "status": "OK"
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    splits = [("train", 60), ("test_public", 20)]
    all_targets = []

    for split, expected in splits:
        aois = list_aois(split)
        if len(aois) != expected:
            print(f"[STOP] {split}: expected {expected} AOIs, found {len(aois)}")
            return 2
        all_targets.extend((split, aoi) for aoi in aois)

    print(f"[OK] AOIs confirmed: {len(all_targets)}")
    print("[INFO] Final target: exactly 2 selected GeoTIFFs per AOI = 160 files")
    print("[INFO] Labels are never downloaded.")
    print("[INFO] Candidate images are temporary; only selected T1/T2 remain.")
    print("[INFO] T1/T2 width, height, bands and CRS must match within each AOI.")
    print("[INFO] AOIs may differ in shape globally; dimensions are recorded in manifest for later padding/preprocessing.")
    print()

    rows: List[dict] = []
    for idx, (split, aoi) in enumerate(all_targets, 1):
        print(f"[{idx:02d}/80] {split}/{aoi}")
        try:
            row = process_aoi(split, aoi)
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
            save_manifest(rows)
            return 130
        except Exception as exc:
            row = {
                "split": split, "aoi": aoi, "t1_date": "", "t2_date": "",
                "t1_path": "", "t2_path": "",
                "t1_valid_fraction": "", "t2_valid_fraction": "",
                "width": "", "height": "", "bands": "", "crs": "",
                "temporal_gap_months": "", "status": f"ERROR_{type(exc).__name__}: {exc}"
            }
        rows.append(row)
        save_manifest(rows)
        print(f"      {row['status']} | T1={row['t1_date']} | T2={row['t2_date']}")

    ok = [r for r in rows if r["status"] == "OK"]
    print()
    print(f"[DONE] AOIs OK: {len(ok)}/80")
    print(f"[DONE] Final selected GeoTIFF count expected: {len(ok) * 2}")
    print(f"[DONE] Manifest: {OUT_ROOT / 'manifest.csv'}")

    return 0 if len(ok) == 80 else 3


if __name__ == "__main__":
    raise SystemExit(main())
