<div align="center">

🛰️ ORBITRACE

Semantic Retrieval & Multi-Temporal Change Intelligence for Satellite Imagery

Search satellite imagery with natural language. Discover the right AOI. Compare dates. Detect urban change. Produce geospatial evidence.








</div>

🌍 What is ORBITRACE?

ORBITRACE is an AI-powered satellite intelligence platform designed to make large satellite archives easier to search, understand and compare.

Instead of manually browsing thousands of images, an analyst can write a query such as:

“Find urban regions with recent construction.”

ORBITRACE then:

understands the query using RemoteCLIP

retrieves semantically relevant satellite observations

resolves AOI, date and geographic location

selects temporal observations for comparison

analyzes built-up change using a SpaceNet-7 HRNet-W48 branch

produces visual and quantitative evidence

⚡ Core Pipeline

flowchart LR
    A["Natural Language Query"] --> B["RemoteCLIP RN50"]
    B --> C["FAISS Vector Search"]
    C --> D["Relevant AOI / Observation"]
    D --> E["Geospatial Metadata"]
    E --> F["Temporal Pairing"]
    F --> G["HRNet-W48 Change Engine"]
    G --> H["Added / Removed Buildings"]
    H --> I["Map + Overlay + Metrics + Evidence"]

🎯 Why ORBITRACE?

Satellite data is abundant, but finding the right scene and extracting useful change evidence is still time-consuming.

Traditional workflows often require an analyst to:

manually locate relevant images

inspect metadata separately

identify matching dates

run change-detection tools independently

interpret outputs manually

ORBITRACE connects these steps into one searchable geospatial intelligence workflow.

✨ Current Capabilities

Capability

Status

Natural-language satellite retrieval

✅ Working

RemoteCLIP RN50 embeddings

✅ Working

Semantic similarity ranking

✅ Working

FAISS vector retrieval

✅ POC tested

GeoTIFF metadata extraction

✅ Working

CRS / bounds / lat-lon extraction

✅ Working

Reverse geocoding

✅ Working

RGB preview generation

✅ Working

Multi-date AOI catalog

✅ Working

HRNet-W48 building-change branch

✅ Validated POC

Added / removed building masks

✅ Working

Local frozen HRNet post-processing

✅ Working

Unified backend API

🚧 In progress

Frontend integration

🚧 In progress

Full archive scale-up

🚧 Planned

🧠 AI / ML Stack

1. RemoteCLIP — Semantic Satellite Retrieval

RemoteCLIP maps both text and satellite imagery into the same embedding space.

"forest with winding roads"
          │
          ▼
     RemoteCLIP
          │
          ▼
     Text Embedding
          │
          ├──────── similarity ────────┐
          │                            │
Satellite Image → RemoteCLIP → Image Embedding

This allows ORBITRACE to retrieve imagery by meaning, not just filenames or tags.

2. FAISS — Fast Vector Search

For a large archive, image embeddings are generated once and stored in a FAISS index.

Satellite Archive
      ↓
RemoteCLIP Embeddings
      ↓
FAISS Index
      ↑
Query Embedding

This avoids encoding every image again for every query.

3. HRNet-W48 — Built-Up Change Detection

The construction-change branch is based on the SpaceNet-7 1-lxastro0 HRNet-W48 solution, selected because it is native to the same multi-temporal urban-development domain.

Current frozen POC configuration:

Scale                : 3×
Inference patch       : 512 × 512
Patch overlap         : None
Building threshold    : 0.35
Minimum component     : 100 pixels

The system derives:

persistent built-up regions

added buildings

removed buildings

change overlays

pixel-level change statistics

The current metrics are limited POC validation results, not production accuracy claims.

📊 Validation Snapshot

Development AOI

L15-1669E-1160N_6679_3549_13

Metric

Added Buildings

Precision

0.4626

Recall

0.6137

F1

0.5276

IoU

0.3583

Unseen AOI

L15-1615E-1206N_6460_3366_13

Using the same frozen parameters with no retuning:

Metric

Added Buildings

Precision

0.4598

Recall

0.6170

F1

0.5269

IoU

0.3577

This close development-vs-unseen performance is encouraging for the current prototype configuration.

🗺️ Geospatial Intelligence

Each GeoTIFF observation can be converted into structured metadata:

{
  "aoi_id": "L15-1670E-1159N_6681_3552_13",
  "date": "2018_09",
  "crs": "EPSG:3857",
  "latitude": 23.2211,
  "longitude": 113.6206,
  "city": "Guangzhou City",
  "state": "Guangdong",
  "country": "China"
}

ORBITRACE can therefore return not only an image, but also where it is, when it was captured and what changed.

🧪 Mini End-to-End Test Archive

The current integration test uses:

3 AOIs
×
2 temporal observations
=
6 real GeoTIFF satellite scenes

Local heavy data is intentionally stored outside OneDrive/Git:

C:\ORBITRACE_DATA\

Example:

C:\ORBITRACE_DATA
├── mini_archive
├── mini_previews
├── hrnet
└── mini_catalog.json

🏗️ Project Structure

satellite-poc/
│
├── core/
│   ├── semantic_search.py
│   ├── faiss_index.py
│   ├── geospatial.py
│   ├── pipeline.py
│   └── change_detection*.py
│
├── ORBITRACE/
│   ├── backend/
│   │   ├── api/
│   │   ├── change_detection/
│   │   ├── geospatial/
│   │   ├── semantic_search/
│   │   └── services/
│   │
│   ├── config/
│   ├── data/
│   ├── frontend/
│   ├── models/
│   ├── outputs/
│   └── scripts/
│       ├── build_mini_catalog.py
│       ├── test_mini_semantic.py
│       └── test_hrnet_local.py
│
├── images/
├── integration_data/
└── tests...

🚀 Quick Start

1. Create / activate environment

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install core dependencies

pip install torch torchvision
pip install open-clip-torch
pip install faiss-cpu
pip install rasterio pillow numpy

3. Build the mini metadata catalog

cd ORBITRACE
python scripts\build_mini_catalog.py

4. Run semantic retrieval test

python scripts\test_mini_semantic.py "forest with winding roads"

5. Run local HRNet post-processing test

python scripts\test_hrnet_local.py

🔎 Example User Journey

USER
"Find areas with new construction near dense urban regions"

        ↓

REMOTECLIP
Understands semantic intent

        ↓

FAISS
Returns top matching satellite observations

        ↓

GEOSPATIAL CATALOG
AOI + date + city + coordinates

        ↓

TEMPORAL PAIRING
Older observation vs newer observation

        ↓

HRNET-W48
Built-up probability maps

        ↓

CHANGE ENGINE
Added / Removed / Persistent buildings

        ↓

ORBITRACE RESULT
Map + before/after + overlay + metrics + evidence

🏆 Winner-Level Features — Locked Roadmap

1. Earliest Evidence Discovery

Automatically determine the first observation where a meaningful change becomes visible.

2. Evidence Timeline

Display all available temporal observations as an interactive development timeline.

3. Explainable Semantic Search

Show why a satellite result matched the analyst's query instead of only returning a similarity score.

4. Change Reliability Gate

Evaluate image-pair quality before trusting a change result:

Registration Quality       GOOD
Cloud Obstruction          LOW
Observation Compatibility  GOOD
Change Confidence          HIGH

5. Evidence Card / Investigation Report

Generate a structured intelligence card containing:

location

source

AOI

T1 / T2 dates

change type

affected area

model / parameters

confidence

before / after / overlay

💡 Additional Future Capability

Similar-Site Discovery

After finding one significant change event:

“Find other locations with similar development patterns.”

This combines semantic retrieval with temporal change intelligence across the entire indexed archive.

📦 Dataset Strategy

ORBITRACE currently uses SpaceNet-7 for multi-temporal urban-development imagery.

Publicly accessible archive discovered during development:

Train AOIs       : 60
Public Test AOIs : 20
---------------------
Accessible AOIs  : 80

The system is designed so scaling from a mini archive to the full accessible archive becomes a batch indexing / inference operation, not a rewrite of the application.

🔐 Repository Policy

Heavy data and model artifacts should stay outside Git:

.venv/
__pycache__/
*.pyc
*.pt
*.pth
*.ckpt
*.npy
*.tif
*.tiff
external/
dataset/
outputs/

The repository should contain code, configuration and lightweight metadata, while raw satellite imagery and model outputs remain in local storage.

🛠️ Technology Stack

Layer

Technology

Semantic Vision-Language Model

RemoteCLIP RN50

Vector Search

FAISS

Building Change Analysis

HRNet-W48

Geospatial Processing

Rasterio

Image Processing

Pillow / NumPy

ML Runtime

PyTorch

Metadata Catalog

JSON / CSV → SQLite planned

API

FastAPI planned

Frontend

Web dashboard planned

Data Source

SpaceNet-7

📌 Current Development Focus

Semantic Retrieval     ✅
Geospatial Catalog     ✅
HRNet Local Adapter    ✅
        ↓
FAISS Finalization
        ↓
Automatic T1/T2 Pairing
        ↓
Unified ORBITRACE Pipeline
        ↓
Backend API
        ↓
Frontend
        ↓
Winner-Level Intelligence Features

⚠️ Important Note

ORBITRACE is currently a research / hackathon prototype.

Validation numbers shown in this repository are based on a limited development and unseen-AOI POC experiment and must not be interpreted as production-grade accuracy.

<div align="center">

🛰️ ORBITRACE

Search the Earth by meaning. Track how it changes over time.

Built for intelligent satellite-image retrieval, temporal analysis and geospatial evidence generation.

</div>
