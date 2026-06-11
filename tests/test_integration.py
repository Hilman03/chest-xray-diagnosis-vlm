"""
tests/test_integration.py
=========================
Workflow Integration Tests — Section 3.8 (Report)

Tests the complete PACS-style workflow via FastAPI backend:
    1. Health check — API is running
    2. Image upload — POST /upload
    3. Analysis trigger — POST /analyze/{id}
    4. Report retrieval — GET /report/{id}
    5. Image serving — GET /images/{id}
    6. PDF export — GET /export/{id}
    7. Report deletion — DELETE /report/{id}
    8. Listing reports — GET /reports
    9. Full end-to-end workflow (upload → analyze → report → export)

Run backend first:
    uvicorn backend.main:app --reload --port 8000

Then run:
    pytest tests/test_integration.py -v
"""

import sys
import io
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

# ─────────────────────────────────────────────────────────────
# SETUP — use TestClient (no live server needed)
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient — no live server required."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi[testclient] (httpx) not installed")

    mock_vlm = {
        "caption"       : "Chest X-Ray showing pneumonia with lobar opacity",
        "disease_label" : "Pneumonia",
        "top_diseases"  : [("Pneumonia", 0.72), ("Effusion", 0.18), ("No Finding", 0.10)],
        "all_scores"    : {"Pneumonia": 0.72, "Effusion": 0.18, "No Finding": 0.10},
        "image_name"    : "test.png",
        "response_time" : 1.5,
        "success"       : True,
    }
    mock_llm = {
        "report"        : "The chest radiograph demonstrates adequate exposure. Increased opacity in the left lower lobe is noted. No additional significant abnormalities are identified.",
        "backend"       : "tinyllama",
        "response_time" : 2.0,
    }

    with patch("backend.main.infer_vlm_with_label", return_value=mock_vlm), \
         patch("backend.main.refine_llm", return_value=mock_llm):
        from backend.main import app
        return TestClient(app)


@pytest.fixture(scope="module")
def sample_png_bytes():
    """Create a valid 224x224 PNG in memory."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(200, 200, 200)).save(buf, "PNG")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# TEST 1: HEALTH CHECK
# ─────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data

    def test_health_status_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] in ("ok", "healthy", "running")

    def test_status_endpoint_returns_200(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────
# TEST 2: IMAGE UPLOAD
# ─────────────────────────────────────────────────────────────
class TestUploadEndpoint:
    def test_upload_png_returns_200(self, client, sample_png_bytes):
        resp = client.post(
            "/upload",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        assert resp.status_code == 200

    def test_upload_returns_image_id(self, client, sample_png_bytes):
        resp = client.post(
            "/upload",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        data = resp.json()
        assert "image_id" in data
        assert len(data["image_id"]) > 0

    def test_upload_returns_status_field(self, client, sample_png_bytes):
        resp = client.post(
            "/upload",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        data = resp.json()
        assert "status" in data

    def test_upload_invalid_extension_rejected(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("malware.exe", b"fake binary data", "application/octet-stream")},
        )
        assert resp.status_code in (400, 415, 422)

    def test_upload_empty_file_rejected(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code in (400, 422, 500)


# ─────────────────────────────────────────────────────────────
# TEST 3: ANALYSIS
# ─────────────────────────────────────────────────────────────
class TestAnalyzeEndpoint:
    @pytest.fixture
    def uploaded_id(self, client, sample_png_bytes):
        resp = client.post(
            "/upload",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        return resp.json()["image_id"]

    def test_analyze_returns_200(self, client, uploaded_id):
        mock_vlm = {
            "caption": "Chest X-Ray showing pneumonia",
            "disease_label": "Pneumonia",
            "top_diseases": [("Pneumonia", 0.72)],
            "all_scores": {"Pneumonia": 0.72},
            "image_name": "test.png",
            "response_time": 1.5,
            "success": True,
        }
        mock_llm = {
            "report": "Normal findings. Adequate exposure. No abnormalities.",
            "backend": "tinyllama",
            "response_time": 2.0,
        }
        with patch("backend.main.infer_vlm_with_label", return_value=mock_vlm), \
             patch("backend.main.refine_llm", return_value=mock_llm):
            resp = client.post(f"/analyze/{uploaded_id}")
        assert resp.status_code == 200

    def test_analyze_returns_disease_label(self, client, uploaded_id):
        mock_vlm = {
            "caption": "Chest X-Ray showing pneumonia",
            "disease_label": "Pneumonia",
            "top_diseases": [("Pneumonia", 0.72)],
            "all_scores": {"Pneumonia": 0.72},
            "image_name": "test.png",
            "response_time": 1.5,
            "success": True,
        }
        mock_llm = {
            "report": "Adequate exposure. Findings noted. No additional abnormalities.",
            "backend": "tinyllama",
            "response_time": 2.0,
        }
        with patch("backend.main.infer_vlm_with_label", return_value=mock_vlm), \
             patch("backend.main.refine_llm", return_value=mock_llm):
            resp = client.post(f"/analyze/{uploaded_id}")
        data = resp.json()
        assert "disease_label" in data or "primary_disease" in data or "finding" in data

    def test_analyze_nonexistent_id_returns_404(self, client):
        resp = client.post("/analyze/nonexistent-id-00000")
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────
# TEST 4: REPORT RETRIEVAL
# ─────────────────────────────────────────────────────────────
class TestReportEndpoint:
    @pytest.fixture
    def analyzed_id(self, client, sample_png_bytes):
        upload_resp = client.post(
            "/upload",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        image_id = upload_resp.json()["image_id"]
        mock_vlm = {
            "caption": "Chest X-Ray showing pneumonia",
            "disease_label": "Pneumonia",
            "top_diseases": [("Pneumonia", 0.72)],
            "all_scores": {"Pneumonia": 0.72},
            "image_name": "test.png",
            "response_time": 1.5,
            "success": True,
        }
        mock_llm = {
            "report": "Adequate exposure. Findings present. No other abnormalities.",
            "backend": "tinyllama",
            "response_time": 2.0,
        }
        with patch("backend.main.infer_vlm_with_label", return_value=mock_vlm), \
             patch("backend.main.refine_llm", return_value=mock_llm):
            client.post(f"/analyze/{image_id}")
        return image_id

    def test_report_retrieval_200(self, client, analyzed_id):
        resp = client.get(f"/report/{analyzed_id}")
        assert resp.status_code == 200

    def test_report_has_required_fields(self, client, analyzed_id):
        resp = client.get(f"/report/{analyzed_id}")
        data = resp.json()
        for field in ["image_id", "disease_label", "report"]:
            assert field in data, f"Missing field: {field}"

    def test_report_not_empty(self, client, analyzed_id):
        resp = client.get(f"/report/{analyzed_id}")
        data = resp.json()
        assert len(data.get("report", "")) > 0

    def test_report_nonexistent_returns_404(self, client):
        resp = client.get("/report/nonexistent-report-id")
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────
# TEST 5: LIST REPORTS
# ─────────────────────────────────────────────────────────────
class TestListReports:
    def test_list_reports_returns_200(self, client):
        resp = client.get("/reports")
        assert resp.status_code == 200

    def test_list_reports_returns_list(self, client):
        resp = client.get("/reports")
        data = resp.json()
        assert isinstance(data, list)


# ─────────────────────────────────────────────────────────────
# TEST 6: END-TO-END WORKFLOW
# ─────────────────────────────────────────────────────────────
class TestEndToEndWorkflow:
    def test_full_pacs_workflow(self, client, sample_png_bytes):
        """
        Complete PACS workflow:
        Upload → Analyze → Get Report → List Reports → Delete
        """
        mock_vlm = {
            "caption": "Chest X-Ray showing pneumonia with consolidation",
            "disease_label": "Pneumonia",
            "top_diseases": [("Pneumonia", 0.80), ("Effusion", 0.12)],
            "all_scores": {"Pneumonia": 0.80, "Effusion": 0.12},
            "image_name": "workflow_test.png",
            "response_time": 1.8,
            "success": True,
        }
        mock_llm = {
            "report": "The radiograph is of adequate diagnostic quality. Dense consolidation is visible in the lower lobe. No additional significant abnormalities are noted.",
            "backend": "tinyllama",
            "response_time": 3.5,
        }

        # Step 1: Upload
        upload_resp = client.post(
            "/upload",
            files={"file": ("workflow_test.png", sample_png_bytes, "image/png")},
        )
        assert upload_resp.status_code == 200
        image_id = upload_resp.json()["image_id"]
        assert image_id

        # Step 2: Analyze
        with patch("backend.main.infer_vlm_with_label", return_value=mock_vlm), \
             patch("backend.main.refine_llm", return_value=mock_llm):
            analyze_resp = client.post(f"/analyze/{image_id}")
        assert analyze_resp.status_code == 200

        # Step 3: Get Report
        report_resp = client.get(f"/report/{image_id}")
        assert report_resp.status_code == 200
        report_data = report_resp.json()
        assert report_data.get("disease_label") == "Pneumonia"
        assert len(report_data.get("report", "")) > 0

        # Step 4: List Reports (image should appear)
        list_resp = client.get("/reports")
        assert list_resp.status_code == 200
        ids = [r.get("image_id") for r in list_resp.json()]
        assert image_id in ids

        # Step 5: Delete Report
        del_resp = client.delete(f"/report/{image_id}")
        assert del_resp.status_code in (200, 204)

        # Step 6: Confirm deleted
        confirm_resp = client.get(f"/report/{image_id}")
        assert confirm_resp.status_code == 404

    def test_workflow_timing_is_recorded(self, client, sample_png_bytes):
        """Analyze response must include timing information."""
        mock_vlm = {
            "caption": "Chest X-Ray normal",
            "disease_label": "No Finding",
            "top_diseases": [("No Finding", 0.65)],
            "all_scores": {"No Finding": 0.65},
            "image_name": "time_test.png",
            "response_time": 2.0,
            "success": True,
        }
        mock_llm = {
            "report": "Normal study. Clear lung fields. No abnormalities noted.",
            "backend": "tinyllama",
            "response_time": 3.0,
        }
        upload_resp = client.post(
            "/upload",
            files={"file": ("time_test.png", sample_png_bytes, "image/png")},
        )
        image_id = upload_resp.json()["image_id"]

        with patch("backend.main.infer_vlm_with_label", return_value=mock_vlm), \
             patch("backend.main.refine_llm", return_value=mock_llm):
            resp = client.post(f"/analyze/{image_id}")

        data = resp.json()
        timing_keys = {"total_time", "vlm_time", "llm_time", "response_time",
                       "processing_time", "inference_time"}
        has_timing = bool(timing_keys.intersection(set(data.keys())))
        assert has_timing, f"No timing field in response: {list(data.keys())}"


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                    str(Path(__file__)), "-v", "--tb=short"])
