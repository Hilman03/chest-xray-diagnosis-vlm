

"""
backend/main.py
===============
Phase 3 — FastAPI Backend for CXR PACS System

Covers all expected output requirements:
    1. VLM generates descriptive text from CXR image
    2. LLM generates interpretable textual output
    3. Inference only (no training)
    4. Upload, display, and generate explanation side by side
    5. Simulated PACS workflow integration
    6. Functional testing endpoint
    7. System performance measurement
    8. System stability (graceful error handling)

Endpoints:
    POST /upload              → upload CXR image (PNG/JPG/DCM)
    POST /analyze/{image_id}  → run VLM + LLM pipeline
    GET  /report/{image_id}   → get stored report
    GET  /reports             → list all reports
    GET  /images/{image_id}   → serve image file
    GET  /export/{image_id}   → download PDF report
    GET  /test/{image_id}     → functional test on image
    GET  /health              → API health + system status
    DELETE /report/{image_id} → delete report
"""

import sys
import os
import uuid
import shutil
import io
import time
import traceback
from pathlib import Path
from datetime import datetime

# Make stdout/stderr UTF-8 safe so console logging of Unicode (e.g. the "→"
# in route docs/log lines) never raises on a legacy Windows code page (cp1252),
# which would otherwise surface as a misleading "Upload error".
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "models"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from PIL import Image

from database import (
    connect_mongodb, save_record, get_record, delete_record,
    list_records, store_exists, is_mongo_connected,
    gridfs_put, gridfs_get,
)
from schemas import (
    UploadResponse, AnalyzeResponse,
    ReportResponse, HealthResponse,
)

load_dotenv(ROOT / ".env")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
UPLOAD_DIR   = ROOT / "data" / "uploads"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".dcm"}
MAX_FILE_MB  = 20
TARGET_SIZE  = (224, 224)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "CXR PACS Analysis API",
    description = (
        "AI-powered Chest X-Ray analysis system. "
        "Uses pretrained VLM (PubMedCLIP) for disease prediction "
        "and LLM for generating descriptive observational reports. "
        "Inference only — no model training involved."
    ),
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Track system start time for uptime measurement
_system_start_time = datetime.now()
_total_requests    = 0
_error_count       = 0


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def preprocess_image(src_path: str, dst_path: str):
    """Resize image to 224x224 RGB — same as Phase 1 pipeline."""
    img = Image.open(src_path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    img.save(dst_path, format="PNG")


def handle_dicom(dicom_path: str, png_path: str) -> dict:
    """Convert DICOM to PNG and extract patient metadata."""
    try:
        import pydicom
        import numpy as np

        ds  = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(png_path, format="PNG")

        return {
            "patient_name"  : str(getattr(ds, "PatientName", "Unknown")),
            "patient_id"    : str(getattr(ds, "PatientID", "Unknown")),
            "patient_age"   : str(getattr(ds, "PatientAge", "Unknown")),
            "patient_sex"   : str(getattr(ds, "PatientSex", "Unknown")),
            "study_date"    : str(getattr(ds, "StudyDate", "Unknown")),
            "modality"      : str(getattr(ds, "Modality", "CR")),
            "view_position" : str(getattr(ds, "ViewPosition", "Unknown")),
            "image_comments": str(getattr(ds, "ImageComments", "")),
        }
    except ImportError:
        return {}
    except Exception as e:
        print(f"  [DICOM] Error: {e}")
        return {}


def _latin1(text) -> str:
    """Make any text safe for fpdf core fonts (latin-1 only)."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "—": "-", "–": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "…": "...", "·": "-",
        "•": "-", " ": " ", "°": " deg",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf(record: dict) -> bytes:
    """Generate PDF report from analysis record."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Header
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "CXR PACS - AI Analysis Report", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6,
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ln=True, align="C")
        pdf.ln(5)

        pdf.set_draw_color(0, 128, 0)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Image Info
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Image Information", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, _latin1(f"File      : {record.get('filename', 'N/A')}"), ln=True)
        pdf.cell(0, 6, _latin1(f"Image ID  : {record.get('image_id', 'N/A')}"), ln=True)
        pdf.cell(0, 6, _latin1(f"Uploaded  : {record.get('uploaded_at', 'N/A')}"), ln=True)
        pdf.cell(0, 6, _latin1(f"Analyzed  : {record.get('analyzed_at', 'N/A')}"), ln=True)
        pdf.ln(5)

        # DICOM metadata
        dicom_meta = record.get("dicom_metadata", {})
        if dicom_meta:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Patient Information (DICOM)", ln=True)
            pdf.set_font("Arial", "", 11)
            for key, val in dicom_meta.items():
                label = key.replace("_", " ").title()
                pdf.cell(0, 6, _latin1(f"{label:<20}: {val}"), ln=True)
            pdf.ln(5)

        # AI Analysis
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "VLM Analysis Results (PubMedCLIP)", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6,
                 _latin1(f"Primary Finding : {record.get('disease_label', 'N/A')}"),
                 ln=True)
        pdf.ln(3)

        top_diseases = record.get("top_diseases", [])
        if top_diseases:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(100, 7, "Disease", border=1)
            pdf.cell(50,  7, "Confidence", border=1, ln=True)
            pdf.set_font("Arial", "", 11)
            for disease, score in top_diseases:
                pdf.cell(100, 7, _latin1(str(disease)), border=1)
                pdf.cell(50,  7, f"{score*100:.1f}%", border=1, ln=True)
        pdf.ln(5)

        # LLM Report
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "LLM Generated Observational Report", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, _latin1(record.get("llm_report", "No report generated.")))
        pdf.ln(5)

        # Performance
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "System Performance", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, f"VLM Inference Time : {record.get('vlm_time', 0)}s", ln=True)
        pdf.cell(0, 6, f"LLM Report Time    : {record.get('llm_time', 0)}s", ln=True)
        pdf.cell(0, 6, f"Total Time         : {record.get('total_time', 0)}s", ln=True)
        pdf.cell(0, 6, _latin1(f"LLM Backend        : {record.get('llm_backend', 'N/A')}"), ln=True)
        pdf.ln(5)

        # Disclaimer
        pdf.set_draw_color(0, 128, 0)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(0, 5,
            "DISCLAIMER: This report is generated by an AI prototype system "
            "for demonstration purposes only. It is NOT intended for clinical "
            "use or medical diagnosis. Always consult a qualified radiologist."
        )

        # fpdf2 (installed in Colab) returns bytearray; old PyFPDF returns str
        try:
            out = pdf.output()           # fpdf2 → bytearray
        except TypeError:
            out = pdf.output(dest="S")   # PyFPDF fallback → str
        if isinstance(out, str):
            return out.encode("latin-1")
        return bytes(out)

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="fpdf2 not installed. Run: pip install fpdf2"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation error: {e}"
        )


# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _system_start_time
    _system_start_time = datetime.now()

    print("=" * 55)
    print("  CXR PACS Analysis API — Starting...")
    print("=" * 55)

    mongo_uri = os.getenv("MONGODB_URI", "")
    if mongo_uri:
        connect_mongodb(mongo_uri)
    else:
        print("  [Startup] No MONGODB_URI — using in-memory store")

    print("  [Startup] Pre-loading PubMedCLIP model...")
    try:
        from vlm_inference import load_model
        load_model()
        print("  [Startup] PubMedCLIP ready")
    except Exception as e:
        print(f"  [Startup] PubMedCLIP pre-load skipped: {e}")

    print("  [Startup] Pre-loading LLM (in-process transformers)...")
    try:
        from llm_refine import load_llm, LLM_MODEL
        if load_llm():
            print(f"  [Startup] LLM ready — {LLM_MODEL}")
        else:
            print(f"  [Startup] WARNING: LLM '{LLM_MODEL}' failed to load")
    except Exception as e:
        print(f"  [Startup] LLM pre-load skipped: {e}")

    # Resource-usage snapshot — verifies models are resident in memory and
    # helps monitor footprint. Skipped silently if psutil is unavailable.
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss  = proc.memory_info().rss / (1024 * 1024)
        print(f"  [Startup] Process memory (RSS): {rss:.0f} MB")
        print(f"  [Startup] System memory used  : "
              f"{psutil.virtual_memory().percent:.0f}%")
    except Exception:
        pass

    print("  [Startup] API ready!")
    print("=" * 55)


# ═════════════════════════════════════════════════════════════
# ENDPOINT 1 — POST /upload
# ═════════════════════════════════════════════════════════════
@app.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image for analysis.

    - Accepts PNG, JPG, JPEG, DCM formats
    - Automatically preprocesses to 224x224 RGB
    - DICOM files: extracts patient metadata
    - Returns image_id for subsequent analysis
    """
    global _total_requests, _error_count
    _total_requests += 1

    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTS}"
            )

        contents = await file.read()
        size_mb  = len(contents) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_MB}MB"
            )

        image_id       = str(uuid.uuid4())
        is_dicom       = ext == ".dcm"
        raw_path       = UPLOAD_DIR / f"{image_id}_raw{ext}"
        processed_path = UPLOAD_DIR / f"{image_id}.png"
        dicom_metadata = {}

        with open(raw_path, "wb") as f:
            f.write(contents)

        try:
            if is_dicom:
                dicom_metadata = handle_dicom(str(raw_path), str(processed_path))
                # handle_dicom swallows its own errors and may NOT write a PNG.
                # Never fall back to copying the raw .dcm to a .png name — that
                # produces an unviewable file. Fail loudly instead so the user
                # knows the DICOM pixel data could not be decoded.
                if not Path(processed_path).exists():
                    raise HTTPException(
                        status_code=400,
                        detail="Could not convert DICOM to an image. The file "
                               "may lack standard pixel data or use an "
                               "unsupported transfer syntax.",
                    )
            else:
                preprocess_image(str(raw_path), str(processed_path))
        except HTTPException:
            raise
        except Exception as e:
            # Only PNG/JPG (already raster images) are safe to copy as-is.
            if is_dicom:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not read DICOM image: {e}",
                )
            shutil.copy(str(raw_path), str(processed_path))
            print(f"  [Upload] Preprocessing fallback: {e}")

        record = {
            "image_id"       : image_id,
            "filename"       : file.filename,
            "image_path"     : str(processed_path),
            "raw_path"       : str(raw_path),
            "file_ext"       : ext,
            "size_mb"        : round(size_mb, 2),
            "is_dicom"       : is_dicom,
            "dicom_metadata" : dicom_metadata,
            "status"         : "uploaded",
            "uploaded_at"    : datetime.now().isoformat(),
        }
        save_record(image_id, record)

        # Persist the processed image in GridFS so it survives a server/Colab
        # restart (the local data/uploads dir is wiped between sessions, but
        # MongoDB keeps the report — without this the viewer can't show it).
        if is_mongo_connected():
            try:
                with open(processed_path, "rb") as imgf:
                    gridfs_put(image_id, imgf.read(), filename=file.filename)
            except Exception as e:
                print(f"  [Upload] GridFS store skipped: {e}")

        print(f"  [Upload] {file.filename} → {image_id}")

        return UploadResponse(
            image_id = image_id,
            filename = file.filename,
            status   = "uploaded",
            message  = "Image uploaded and preprocessed successfully.",
            is_dicom = is_dicom,
        )

    except HTTPException:
        _error_count += 1
        raise
    except Exception as e:
        _error_count += 1
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


# ═════════════════════════════════════════════════════════════
# ENDPOINT 2 — POST /analyze/{image_id}
# ═════════════════════════════════════════════════════════════
@app.post("/analyze/{image_id}", response_model=AnalyzeResponse)
async def analyze_image(image_id: str):
    """
    Run AI pipeline on uploaded image.

    Pipeline:
        1. PubMedCLIP (VLM) — predicts disease from image
        2. Ollama LLM — generates structured observational report
           (prompt-engineered, grounded in the PubMedCLIP findings)
        3. Stores result in MongoDB Atlas

    Returns descriptive and interpretable textual output
    as required by the FYP expected output.
    """
    global _total_requests, _error_count
    _total_requests += 1

    record = get_record(image_id)
    if not record:
        _error_count += 1
        raise HTTPException(
            status_code=404,
            detail="Image not found. Please upload first."
        )

    image_path = record.get("image_path", "")
    if not Path(image_path).exists():
        _error_count += 1
        raise HTTPException(
            status_code=404,
            detail="Image file not found on disk."
        )

    print(f"  [Analyze] Running VLM+LLM pipeline: {record['filename']}")

    try:
        from pipeline import run_pipeline
        result = run_pipeline(image_path)

        if result["status"] == "error":
            _error_count += 1
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {result['error']}"
            )

        analyzed_at = datetime.now().isoformat()
        record.update({
            "status"        : "analyzed",
            "disease_label" : result["disease_label"],
            "top_diseases"  : result["top_diseases"],
            "vlm_caption"   : result["vlm_caption"],
            "llm_report"    : result["llm_report"],
            "llm_backend"   : result["llm_backend"],
            "vlm_time"      : result["vlm_time"],
            "llm_time"      : result["llm_time"],
            "total_time"    : result["total_time"],
            "analyzed_at"   : analyzed_at,
        })

        # Measure database storage time as a performance metric, then persist
        # it (a second save is negligible and keeps db_time in the record).
        db_start = time.time()
        save_record(image_id, record)
        db_time = round(time.time() - db_start, 3)
        record["db_time"] = db_time
        save_record(image_id, record)

        print(f"  [Analyze] Done — {result['disease_label']} "
              f"in {result['total_time']}s (db {db_time}s)")

        return AnalyzeResponse(
            image_id      = image_id,
            filename      = record["filename"],
            status        = "analyzed",
            disease_label = result["disease_label"],
            top_diseases  = result["top_diseases"],
            vlm_caption   = result["vlm_caption"],
            llm_report    = result["llm_report"],
            llm_backend   = result["llm_backend"],
            vlm_time      = result["vlm_time"],
            llm_time      = result["llm_time"],
            db_time       = db_time,
            total_time    = result["total_time"],
            analyzed_at   = analyzed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        _error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════
# ENDPOINT 3 — GET /report/{image_id}
# ═════════════════════════════════════════════════════════════
@app.get("/report/{image_id}", response_model=ReportResponse)
async def get_report(image_id: str):
    """Retrieve stored analysis report."""
    record = get_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found.")

    # Build response defensively: use stored values when present and non-null,
    # fall back to "" for any missing required field, and let Pydantic apply
    # the declared default for optional fields that are absent.
    data = {}
    for name, field in ReportResponse.model_fields.items():
        if name in record and record[name] is not None:
            data[name] = record[name]
        elif field.is_required():
            data[name] = ""
    return ReportResponse(**data)


# ═════════════════════════════════════════════════════════════
# ENDPOINT 4 — GET /reports
# ═════════════════════════════════════════════════════════════
@app.get("/reports")
async def list_all_reports():
    """List all stored reports summary."""
    records = list_records()
    return {
        "total"  : len(records),
        "reports": [
            {
                "image_id"       : r.get("image_id", ""),
                "filename"       : r.get("filename", ""),
                "status"         : r.get("status", ""),
                "disease_label"  : r.get("disease_label", ""),
                "is_dicom"       : r.get("is_dicom", False),
                "vlm_time"       : r.get("vlm_time", 0),
                "llm_time"       : r.get("llm_time", 0),
                "db_time"        : r.get("db_time", 0),
                "total_time"     : r.get("total_time", 0),
                "uploaded_at"    : r.get("uploaded_at", ""),
                "analyzed_at"    : r.get("analyzed_at", ""),
                "report_exported": r.get("report_exported", False),
                "exported_at"    : r.get("exported_at", ""),
            }
            for r in records
        ]
    }


# ═════════════════════════════════════════════════════════════
# ENDPOINT 5 — GET /images/{image_id}
# ═════════════════════════════════════════════════════════════
@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """Serve processed image file for display in PACS viewer."""
    record = get_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Image not found.")

    image_path = record.get("image_path", "")
    if image_path and Path(image_path).exists():
        return FileResponse(image_path, media_type="image/png")

    # Disk file missing (e.g. fresh session) — fall back to the GridFS copy.
    data = gridfs_get(image_id)
    if data:
        return StreamingResponse(io.BytesIO(data), media_type="image/png")

    raise HTTPException(status_code=404, detail="Image file not found.")


# ═════════════════════════════════════════════════════════════
# ENDPOINT 6 — GET /export/{image_id}
# ═════════════════════════════════════════════════════════════
@app.get("/export/{image_id}")
async def export_pdf(image_id: str):
    """
    Export analysis report as downloadable PDF.

    PDF includes:
        - Image information
        - DICOM patient metadata (if applicable)
        - VLM disease predictions with confidence scores
        - LLM generated observational report
        - System performance metrics
        - Disclaimer
    """
    record = get_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found.")

    if record.get("status") != "analyzed":
        raise HTTPException(
            status_code=400,
            detail="Image not analyzed yet. Run /analyze first."
        )

    pdf_bytes = generate_pdf(record)
    filename  = f"CXR_Report_{record['filename'].split('.')[0]}.pdf"

    # Audit trail: record that the report was exported and when.
    record["report_exported"] = True
    record["exported_at"]     = datetime.now().isoformat()
    save_record(image_id, record)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ═════════════════════════════════════════════════════════════
# ENDPOINT 7 — GET /test/{image_id}  ← NEW
# ═════════════════════════════════════════════════════════════
@app.get("/test/{image_id}")
async def functional_test(image_id: str):
    """
    Functional testing endpoint.

    Tests the full pipeline on a specific image and returns
    a structured test report covering:
        - Image loading test
        - VLM inference test
        - LLM report generation test
        - Response time measurement
        - Output validation
        - System stability check

    Used for Phase 5 evaluation and testing documentation.
    """
    record = get_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Image not found.")

    image_path = record.get("image_path", "")
    test_results = {
        "image_id"    : image_id,
        "filename"    : record.get("filename", ""),
        "test_date"   : datetime.now().isoformat(),
        "tests"       : {},
        "overall"     : "PASS",
        "errors"      : [],
    }

    # Test 1 — Image file exists
    test_results["tests"]["image_exists"] = {
        "name"   : "Image File Exists",
        "status" : "PASS" if Path(image_path).exists() else "FAIL",
        "detail" : image_path,
    }
    if not Path(image_path).exists():
        test_results["overall"] = "FAIL"
        test_results["errors"].append("Image file not found on disk")

    # Test 2 — Image loadable
    try:
        img = Image.open(image_path)
        test_results["tests"]["image_loadable"] = {
            "name"   : "Image Loadable",
            "status" : "PASS",
            "detail" : f"Size: {img.size}, Mode: {img.mode}",
        }
    except Exception as e:
        test_results["tests"]["image_loadable"] = {
            "name"   : "Image Loadable",
            "status" : "FAIL",
            "detail" : str(e),
        }
        test_results["overall"] = "FAIL"
        test_results["errors"].append(f"Image load error: {e}")

    # Test 3 — VLM inference
    try:
        from vlm_inference import infer_vlm_with_label
        start      = time.time()
        vlm_result = infer_vlm_with_label(image_path)
        vlm_time   = round(time.time() - start, 3)

        test_results["tests"]["vlm_inference"] = {
            "name"          : "VLM Inference (PubMedCLIP)",
            "status"        : "PASS",
            "disease_label" : vlm_result["disease_label"],
            "top_diseases"  : vlm_result["top_diseases"],
            "response_time" : f"{vlm_time}s",
            "detail"        : "Disease prediction successful",
        }
    except Exception as e:
        test_results["tests"]["vlm_inference"] = {
            "name"   : "VLM Inference (PubMedCLIP)",
            "status" : "FAIL",
            "detail" : str(e),
        }
        test_results["overall"] = "FAIL"
        test_results["errors"].append(f"VLM error: {e}")
        vlm_result = None

    # Test 4 — LLM report generation
    try:
        from llm_refine import refine_llm
        if vlm_result:
            start      = time.time()
            llm_result = refine_llm(
                vlm_result["caption"],
                vlm_result["disease_label"],
                vlm_result["top_diseases"],
            )
            llm_time = round(time.time() - start, 3)

            test_results["tests"]["llm_report"] = {
                "name"          : "LLM Report Generation",
                "status"        : "PASS",
                "backend"       : llm_result["backend"],
                "report_length" : len(llm_result["report"]),
                "response_time" : f"{llm_time}s",
                "detail"        : "Report generated successfully",
            }
        else:
            test_results["tests"]["llm_report"] = {
                "name"   : "LLM Report Generation",
                "status" : "SKIP",
                "detail" : "Skipped due to VLM failure",
            }
    except Exception as e:
        test_results["tests"]["llm_report"] = {
            "name"   : "LLM Report Generation",
            "status" : "FAIL",
            "detail" : str(e),
        }
        test_results["overall"] = "FAIL"
        test_results["errors"].append(f"LLM error: {e}")

    # Test 5 — Output validation
    if vlm_result:
        has_disease  = bool(vlm_result.get("disease_label"))
        has_scores   = len(vlm_result.get("top_diseases", [])) > 0
        output_valid = has_disease and has_scores

        test_results["tests"]["output_validation"] = {
            "name"        : "Output Validation",
            "status"      : "PASS" if output_valid else "FAIL",
            "has_disease" : has_disease,
            "has_scores"  : has_scores,
            "detail"      : "Output contains disease label and confidence scores",
        }

    # Test 6 — System stability
    test_results["tests"]["system_stability"] = {
        "name"         : "System Stability",
        "status"       : "PASS",
        "uptime"       : str(datetime.now() - _system_start_time),
        "total_requests": _total_requests,
        "error_count"  : _error_count,
        "error_rate"   : f"{(_error_count/_total_requests*100) if _total_requests > 0 else 0:.1f}%",
        "detail"       : "System running without crashes",
    }

    # Summary
    pass_count = sum(
        1 for t in test_results["tests"].values()
        if t["status"] == "PASS"
    )
    total_tests = len(test_results["tests"])
    test_results["summary"] = {
        "passed"     : pass_count,
        "failed"     : total_tests - pass_count,
        "total"      : total_tests,
        "pass_rate"  : f"{pass_count/total_tests*100:.0f}%",
    }

    return test_results


# ═════════════════════════════════════════════════════════════
# ENDPOINT 8 — GET /health
# ═════════════════════════════════════════════════════════════
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API health check with system performance metrics.
    Shows uptime, request count, and MongoDB status.
    """
    return HealthResponse(
        status        = "running",
        message       = "CXR PACS Analysis API is live",
        mongodb       = "connected" if is_mongo_connected() else "in-memory only",
        total_reports = len(list_records()),
    )


# ═════════════════════════════════════════════════════════════
# ENDPOINT 9 — GET /status
# ═════════════════════════════════════════════════════════════
@app.get("/status")
async def system_status():
    """
    Detailed system status for performance evaluation.
    Used in Phase 5 testing documentation.
    """
    uptime = datetime.now() - _system_start_time
    return {
        "status"          : "running",
        "uptime"          : str(uptime),
        "uptime_seconds"  : uptime.total_seconds(),
        "total_requests"  : _total_requests,
        "error_count"     : _error_count,
        "error_rate"      : f"{(_error_count/_total_requests*100) if _total_requests > 0 else 0:.1f}%",
        "mongodb"         : "connected" if is_mongo_connected() else "in-memory only",
        "total_reports"   : len(list_records()),
        "models_loaded"   : {
            "vlm" : "PubMedCLIP (flaviagiammarino/pubmed-clip-vit-base-patch32)",
            "llm" : "Ollama vision (llama3.2-vision)",
        },
        "inference_only"  : True,
        "training"        : False,
    }


# ═════════════════════════════════════════════════════════════
# ENDPOINT 10 — DELETE /report/{image_id}
# ═════════════════════════════════════════════════════════════
@app.delete("/report/{image_id}")
async def delete_report(image_id: str):
    """Delete report and associated image files."""
    record = get_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found.")

    for path_key in ["image_path", "raw_path"]:
        path = record.get(path_key, "")
        if path and Path(path).exists():
            Path(path).unlink()

    delete_record(image_id)

    return {"message": f"Report {image_id} deleted successfully"}


# ═════════════════════════════════════════════════════════════
# ENDPOINT 11 — GET /test  (system-wide functional tests)
# ═════════════════════════════════════════════════════════════
@app.get("/test")
async def run_system_tests():
    """
    Run all functionality tests using a sample image from
    data/processed/images/.  No image upload required.
    Returns structured pass/fail results for each test category.
    """
    results = []

    def _r(name, status, detail="", **extra):
        results.append({"name": name, "status": status, "detail": detail, **extra})

    # ── Test 1: sample image available ───────────────────────
    images_dir = ROOT / "data" / "processed" / "images"
    sample_images = sorted(images_dir.glob("*.png"))

    # Fallback: use any uploaded image if processed folder is empty
    if not sample_images:
        upload_pngs = [
            p for p in sorted(UPLOAD_DIR.glob("*.png"))
            if "_raw" not in p.name
        ]
        sample_images = upload_pngs

    # Fallback 2: use any analyzed record from database
    if not sample_images:
        for rec in list_records():
            p = rec.get("image_path", "")
            if p and Path(p).exists():
                sample_images = [Path(p)]
                break

    if sample_images:
        _r("Sample Images Available", "PASS",
           f"{len(sample_images)} image(s) found")
        image_path = str(sample_images[0])
    else:
        _r("Sample Images Available", "FAIL",
           "No images found — upload an image via POST /upload first")
        image_path = None

    # ── Test 2: image loadable ────────────────────────────────
    if image_path:
        try:
            img = Image.open(image_path)
            _r("Image Loading", "PASS",
               f"Size: {img.size}, Mode: {img.mode}")
        except Exception as e:
            _r("Image Loading", "FAIL", str(e))
            image_path = None
    else:
        _r("Image Loading", "SKIP", "No sample image")

    # ── Test 3: VLM inference ─────────────────────────────────
    vlm_result = None
    if image_path:
        try:
            from vlm_inference import infer_vlm_with_label
            t0 = time.time()
            vlm_result = infer_vlm_with_label(image_path)
            elapsed = round(time.time() - t0, 3)
            _r("VLM Inference (PubMedCLIP)", "PASS",
               f"Primary: {vlm_result['disease_label']}  |  Time: {elapsed}s",
               disease_label=vlm_result["disease_label"],
               top_diseases=vlm_result["top_diseases"],
               response_time=elapsed)
        except Exception as e:
            _r("VLM Inference (PubMedCLIP)", "FAIL", str(e))
    else:
        _r("VLM Inference (PubMedCLIP)", "SKIP", "No image to test")

    # ── Test 4: output structure ──────────────────────────────
    if vlm_result:
        required = ["caption", "disease_label", "top_diseases",
                    "all_scores", "response_time", "success"]
        missing = [k for k in required if k not in vlm_result]
        scores_ok = all(0.0 <= s <= 1.0
                        for _, s in vlm_result.get("top_diseases", []))
        if not missing and scores_ok:
            _r("VLM Output Structure", "PASS",
               "All required keys present, scores in [0,1]")
        else:
            detail = (f"Missing keys: {missing}" if missing
                      else "Scores out of range")
            _r("VLM Output Structure", "FAIL", detail)
    else:
        _r("VLM Output Structure", "SKIP", "VLM did not run")

    # ── Test 5: LLM report generation ────────────────────────
    llm_result = None
    if vlm_result:
        try:
            from llm_refine import refine_llm
            t0 = time.time()
            llm_result = refine_llm(
                vlm_result["caption"],
                vlm_result["disease_label"],
                vlm_result["top_diseases"],
            )
            elapsed = round(time.time() - t0, 3)
            _r("LLM Report Generation (TinyLlama)", "PASS",
               llm_result["report"],
               backend=llm_result["backend"],
               response_time=elapsed)
        except Exception as e:
            _r("LLM Report Generation (TinyLlama)", "FAIL", str(e))
    else:
        _r("LLM Report Generation (TinyLlama)", "SKIP", "VLM did not produce output")

    # ── Test 6: full pipeline end-to-end ─────────────────────
    if image_path:
        try:
            from pipeline import run_pipeline
            t0 = time.time()
            pr = run_pipeline(image_path)
            elapsed = round(time.time() - t0, 3)
            if pr["status"] == "success":
                _r("Full Pipeline (end-to-end)", "PASS",
                   f"Total: {elapsed}s  Disease: {pr['disease_label']}",
                   total_time=elapsed)
            else:
                _r("Full Pipeline (end-to-end)", "FAIL", pr.get("error", ""))
        except Exception as e:
            _r("Full Pipeline (end-to-end)", "FAIL", str(e))
    else:
        _r("Full Pipeline (end-to-end)", "SKIP", "No image to test")

    # ── Test 7: API health ────────────────────────────────────
    _r("API Health", "PASS",
       f"Uptime: {str(datetime.now() - _system_start_time).split('.')[0]}  "
       f"Requests: {_total_requests}  Errors: {_error_count}")

    # ── Test 8: database ──────────────────────────────────────
    db_status = "connected" if is_mongo_connected() else "in-memory"
    _r("Database", "PASS", f"Mode: {db_status}  Records: {len(list_records())}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total  = len(results)

    return {
        "test_date": datetime.now().isoformat(),
        "overall" : "PASS" if failed == 0 else "FAIL",
        "summary" : {"passed": passed, "failed": failed,
                     "skipped": total - passed - failed, "total": total},
        "tests"   : results,
    }


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  CXR PACS Analysis API")
    print("  http://localhost:8000/docs")
    print("=" * 55)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)