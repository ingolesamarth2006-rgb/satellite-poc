"""
ORBITRACE AI pipeline service.

This module is intentionally separate from FastAPI app.py.

Current POC integration strategy:
- Reuse the already validated ORBITRACE CLI pipeline.
- Execute it with the same Python environment as the backend.
- Read the generated summary.json / evidence_card.json.
- Return a clean JSON object to FastAPI.
- Handle HRNet cache-miss safely without inventing a change result.

This avoids rewriting the working AI pipeline during the SIH demo build.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent
ORBITRACE_DIR = BACKEND_DIR.parent

DATA_ROOT = Path(
    os.getenv("ORBITRACE_DATA_ROOT", r"C:\ORBITRACE_DATA")
)

PIPELINE_SCRIPT = ORBITRACE_DIR / "scripts" / "test_full_pipeline.py"
RESULTS_ROOT = DATA_ROOT / "pipeline_results"

PIPELINE_TIMEOUT_SECONDS = int(
    os.getenv("ORBITRACE_PIPELINE_TIMEOUT", "300")
)


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------

# The current POC pipeline loads heavy ML assets and writes result files.
# For a live demo, one analysis at a time is safer and deterministic.
_PIPELINE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AOI_RE = re.compile(
    r"^AOI:\s*(?P<aoi>[A-Za-z0-9_-]+)\s*$",
    re.MULTILINE,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {"data": data}

    except (OSError, json.JSONDecodeError):
        return None


def _extract_aoi_id(stdout: str) -> str | None:
    match = _AOI_RE.search(stdout)
    return match.group("aoi") if match else None


def _artifact_paths(aoi_id: str) -> dict[str, str]:
    result_dir = RESULTS_ROOT / aoi_id

    candidates = {
        "overlay": result_dir / "final_change_overlay.png",
        "added_mask": result_dir / "added_buildings.png",
        "removed_mask": result_dir / "removed_buildings.png",
        "summary": result_dir / "summary.json",
        "evidence_card": result_dir / "evidence_card.json",
    }

    result: dict[str, str] = {}

    for name, path in candidates.items():
        if path.exists():
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                result[name] = f"/artifacts/{aoi_id}/{path.name}"
            else:
                result[name] = str(path)

    return result


def _build_cache_miss_response(
    query: str,
    aoi_id: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "status": "review",
        "analysis_status": "HRNET_CACHE_MISS",
        "query": query,
        "aoi_id": aoi_id,
        "message": (
            "Semantic retrieval and temporal pairing succeeded, "
            "but no cached HRNet probability-map pair is available "
            "for the selected AOI."
        ),
        "change_result": None,
        "reliability": {
            "gate_status": "REVIEW",
            "reason": "HRNET_CACHE_MISS",
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


# ---------------------------------------------------------------------------
# Public service function
# ---------------------------------------------------------------------------

def analyze_query(query: str) -> dict[str, Any]:
    """
    Execute the current validated ORBITRACE pipeline for one query.

    Returns a JSON-compatible dictionary for FastAPI.
    """

    clean_query = " ".join(query.split()).strip()

    if len(clean_query) < 3:
        raise ValueError("Query must contain at least 3 characters.")

    if not PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"ORBITRACE pipeline script not found: {PIPELINE_SCRIPT}"
        )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        clean_query,
    ]

    started = time.perf_counter()

    with _PIPELINE_LOCK:
        try:
            completed = subprocess.run(
                command,
                cwd=str(ORBITRACE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=PIPELINE_TIMEOUT_SECONDS,
                check=False,
            )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ORBITRACE analysis timed out after "
                f"{PIPELINE_TIMEOUT_SECONDS} seconds."
            ) from exc

    elapsed_seconds = time.perf_counter() - started

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        error_tail = "\n".join(
            (stderr or stdout).splitlines()[-20:]
        )

        raise RuntimeError(
            "ORBITRACE pipeline failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"Last output:\n{error_tail}"
        )

    aoi_id = _extract_aoi_id(stdout)

    # Safe abstention: never fabricate change metrics when HRNet data is absent.
    if "HRNET CACHE MISS" in stdout:
        return _build_cache_miss_response(
            query=clean_query,
            aoi_id=aoi_id,
            elapsed_seconds=elapsed_seconds,
        )

    if not aoi_id:
        raise RuntimeError(
            "Pipeline completed but no AOI id could be identified."
        )

    result_dir = RESULTS_ROOT / aoi_id

    summary = _read_json(result_dir / "summary.json")
    evidence = _read_json(result_dir / "evidence_card.json")

    if summary is None:
        raise RuntimeError(
            "Pipeline completed but summary.json was not generated "
            f"for AOI {aoi_id}."
        )

    return {
        "status": "success",
        "analysis_status": "COMPLETE",
        "query": clean_query,
        "aoi_id": aoi_id,
        "summary": summary,
        "evidence_card": evidence,
        "artifacts": _artifact_paths(aoi_id),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


# ---------------------------------------------------------------------------
# Local smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    test_query = " ".join(sys.argv[1:]).strip()

    if not test_query:
        test_query = "forest with winding roads"

    pprint.pp(analyze_query(test_query))
