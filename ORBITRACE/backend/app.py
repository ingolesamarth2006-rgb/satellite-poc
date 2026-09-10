"""
ORBITRACE FastAPI backend entrypoint.

Purpose:
- Stable API surface for the frontend.
- Health/readiness checks for ORBITRACE assets.
- Safe access to generated result artifacts.
- A clean /analyze contract that will call backend.services.pipeline_service
  once the AI service adapter is wired.

Run from the ORBITRACE directory:
    uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
ORBITRACE_DIR = BACKEND_DIR.parent
PROJECT_ROOT = ORBITRACE_DIR.parent

DATA_ROOT = Path(
    os.getenv("ORBITRACE_DATA_ROOT", r"C:\ORBITRACE_DATA")
)

FAISS_INDEX_PATH = DATA_ROOT / "faiss" / "mini_orbitrace.index"
FAISS_METADATA_PATH = DATA_ROOT / "faiss" / "mini_orbitrace_metadata.json"
CATALOG_PATH = DATA_ROOT / "mini_catalog.json"
HRNET_ROOT = DATA_ROOT / "hrnet"
PIPELINE_RESULTS_ROOT = DATA_ROOT / "pipeline_results"

API_VERSION = "0.2.0"
API_PREFIX = "/api/v1"

AOI_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("ORBITRACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("orbitrace.api")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Natural-language satellite search / analysis query.",
        examples=["forest with winding roads"],
    )


class APIMessage(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asset_status() -> dict[str, dict[str, Any]]:
    """Return lightweight readiness information without loading ML models."""

    assets = {
        "faiss_index": FAISS_INDEX_PATH,
        "faiss_metadata": FAISS_METADATA_PATH,
        "catalog": CATALOG_PATH,
        "hrnet_cache": HRNET_ROOT,
        "pipeline_results": PIPELINE_RESULTS_ROOT,
    }

    result: dict[str, dict[str, Any]] = {}

    for name, path in assets.items():
        result[name] = {
            "exists": path.exists(),
            "path": str(path),
        }

    return result


def _demo_ready(assets: dict[str, dict[str, Any]]) -> bool:
    """Critical assets required for the current ORBITRACE POC."""

    required = (
        "faiss_index",
        "faiss_metadata",
        "catalog",
        "hrnet_cache",
    )

    return all(assets[name]["exists"] for name in required)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {"data": data}

    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read JSON file: %s", path)
        return None


def _validate_aoi_id(aoi_id: str) -> None:
    if not AOI_ID_PATTERN.fullmatch(aoi_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_aoi_id",
                "message": "AOI id contains unsupported characters.",
            },
        )


def _artifact_url(request: Request, aoi_id: str, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/artifacts/{aoi_id}/{filename}"


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Keep startup lightweight.

    Heavy AI models are intentionally not loaded here. The pipeline service
    owns AI initialization so the API can still expose health/status routes
    even if an AI asset is temporarily unavailable.
    """

    PIPELINE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ORBITRACE API v%s", API_VERSION)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Data root: %s", DATA_ROOT)

    yield

    logger.info("Stopping ORBITRACE API")


app = FastAPI(
    title="ORBITRACE API",
    version=API_VERSION,
    description=(
        "Semantic satellite retrieval, temporal pairing, "
        "building-change analysis, reliability, and evidence APIs."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# Explicit origins can be added through:
# ORBITRACE_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
configured_origins = [
    origin.strip()
    for origin in os.getenv("ORBITRACE_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request metadata middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    return response


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": "validation_error",
            "message": "Request validation failed.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled API error | %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": "internal_error",
            "message": "ORBITRACE encountered an internal server error.",
        },
    )


# ---------------------------------------------------------------------------
# Static generated artifacts
# ---------------------------------------------------------------------------

# Examples:
# /artifacts/<AOI>/final_change_overlay.png
# /artifacts/<AOI>/added_buildings.png
# /artifacts/<AOI>/removed_buildings.png
app.mount(
    "/artifacts",
    StaticFiles(
        directory=str(PIPELINE_RESULTS_ROOT),
        check_dir=False,
    ),
    name="artifacts",
)


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    return {
        "system": "ORBITRACE",
        "api_version": API_VERSION,
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": API_PREFIX,
    }


@app.get("/health", tags=["System"])
def health():
    """Fast liveness endpoint for browser/demo checks."""

    return {
        "status": "ok",
        "service": "ORBITRACE API",
        "version": API_VERSION,
    }


@app.get(f"{API_PREFIX}/status", tags=["System"])
def system_status():
    """
    Detailed readiness endpoint.

    This checks file availability only; it does not run RemoteCLIP,
    FAISS search, or HRNet inference.
    """

    assets = _asset_status()
    ready = _demo_ready(assets)

    return {
        "status": "ready" if ready else "degraded",
        "ready_for_demo": ready,
        "version": API_VERSION,
        "assets": assets,
    }


@app.get(f"{API_PREFIX}/results/{{aoi_id}}", tags=["Results"])
def get_result(aoi_id: str, request: Request):
    """
    Return the latest saved ORBITRACE result for an AOI.

    Works with the summary.json / evidence_card.json files already produced
    by the current validated POC pipeline.
    """

    _validate_aoi_id(aoi_id)

    result_dir = PIPELINE_RESULTS_ROOT / aoi_id

    if not result_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "result_not_found",
                "message": f"No saved ORBITRACE result found for AOI {aoi_id}.",
            },
        )

    summary = _read_json(result_dir / "summary.json")
    evidence = _read_json(result_dir / "evidence_card.json")

    if summary is None and evidence is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "result_files_missing",
                "message": (
                    "The AOI result folder exists, but no readable "
                    "summary/evidence JSON was found."
                ),
            },
        )

    artifact_files = {
        "overlay": "final_change_overlay.png",
        "added_mask": "added_buildings.png",
        "removed_mask": "removed_buildings.png",
    }

    artifacts = {
        key: _artifact_url(request, aoi_id, filename)
        for key, filename in artifact_files.items()
        if (result_dir / filename).exists()
    }

    return {
        "status": "success",
        "aoi_id": aoi_id,
        "summary": summary,
        "evidence_card": evidence,
        "artifacts": artifacts,
    }


@app.post(f"{API_PREFIX}/analyze", tags=["Analysis"])
async def analyze(payload: AnalyzeRequest):
    """
    Stable frontend contract for ORBITRACE analysis.

    The heavy AI implementation lives in:
        backend/services/pipeline_service.py

    Keeping the AI adapter outside app.py prevents this API entrypoint from
    becoming tightly coupled to RemoteCLIP / FAISS / HRNet implementation
    details and makes debugging substantially safer.
    """

    service_file = BACKEND_DIR / "services" / "pipeline_service.py"

    if not service_file.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "pipeline_service_not_wired",
                "message": (
                    "API is healthy, but the ORBITRACE AI pipeline service "
                    "has not been wired yet."
                ),
            },
        )

    try:
        # Lazy import keeps /health and /status available even if an AI
        # dependency later has an initialization problem.
        from backend.services.pipeline_service import analyze_query

        result = await run_in_threadpool(
            analyze_query,
            payload.query.strip(),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Pipeline analysis failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "analysis_failed",
                "message": str(exc),
            },
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "invalid_pipeline_response",
                "message": "Pipeline service did not return a JSON-compatible object.",
            },
        )

    return result


# ---------------------------------------------------------------------------
# Backward-compatible aliases (useful during frontend prototyping)
# ---------------------------------------------------------------------------

@app.get("/result/{aoi_id}", include_in_schema=False)
def result_alias(aoi_id: str, request: Request):
    return get_result(aoi_id, request)


@app.post("/analyze", include_in_schema=False)
async def analyze_alias(payload: AnalyzeRequest):
    return await analyze(payload)


# ---------------------------------------------------------------------------
# Optional direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("ORBITRACE_API_HOST", "127.0.0.1"),
        port=int(os.getenv("ORBITRACE_API_PORT", "8000")),
        log_level=os.getenv("ORBITRACE_LOG_LEVEL", "info").lower(),
    )
