"""
tests/test_performance.py
=========================
System Performance Tests — Section 3.8 (Report)

Covers:
    1. Response time — pipeline completes within acceptable limits
    2. Consistency — same image produces same disease label every run
    3. VLM-only speed benchmark
    4. Output stability across repeated inferences

Run:
    pytest tests/test_performance.py -v
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

# Acceptable response time thresholds (seconds)
VLM_MAX_TIME   = 60.0    # PubMedCLIP on CPU
LLM_MAX_TIME   = 120.0   # TinyLlama on CPU
TOTAL_MAX_TIME = 180.0   # Full pipeline on CPU (generous for cold start)
TOTAL_GPU_TIME = 30.0    # Full pipeline on CUDA


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_image_path():
    images = sorted((ROOT / "data" / "processed" / "images").glob("*.png"))
    if not images:
        pytest.skip("No processed images — run scripts/preprocess.py first")
    return str(images[0])


@pytest.fixture
def mock_vlm_result():
    return {
        "caption"       : "Chest X-Ray showing pneumonia with lobar opacity",
        "disease_label" : "Pneumonia",
        "top_diseases"  : [("Pneumonia", 0.72), ("Effusion", 0.18), ("No Finding", 0.10)],
        "all_scores"    : {"Pneumonia": 0.72, "Effusion": 0.18, "No Finding": 0.10},
        "image_name"    : "test.png",
        "response_time" : 1.5,
        "success"       : True,
    }


@pytest.fixture
def mock_llm_result():
    return {
        "report"        : "The chest radiograph demonstrates adequate exposure. Increased opacity in the left lower lobe is noted. No additional significant abnormalities are identified.",
        "backend"       : "tinyllama",
        "response_time" : 2.0,
    }


# ─────────────────────────────────────────────────────────────
# TEST 1: RESPONSE TIME
# ─────────────────────────────────────────────────────────────
class TestResponseTime:
    def test_vlm_inference_time_recorded(self, sample_image_path):
        """VLM records its own response_time."""
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        assert result["response_time"] > 0

    def test_vlm_inference_within_limit(self, sample_image_path):
        """PubMedCLIP inference must complete within VLM_MAX_TIME."""
        from models.vlm_inference import infer_vlm_with_label
        t0     = time.time()
        result = infer_vlm_with_label(sample_image_path)
        elapsed = time.time() - t0
        assert elapsed <= VLM_MAX_TIME, (
            f"VLM took {elapsed:.1f}s — exceeds {VLM_MAX_TIME}s limit"
        )

    def test_pipeline_total_time_recorded(self, sample_image_path,
                                           mock_vlm_result, mock_llm_result):
        """Pipeline records total_time in result."""
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            result = run_pipeline(sample_image_path)
        assert result["total_time"] > 0

    def test_pipeline_vlm_time_positive(self, sample_image_path,
                                         mock_vlm_result, mock_llm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            result = run_pipeline(sample_image_path)
        assert result["vlm_time"] >= 0

    def test_pipeline_llm_time_positive(self, sample_image_path,
                                         mock_vlm_result, mock_llm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            result = run_pipeline(sample_image_path)
        assert result["llm_time"] >= 0

    def test_total_time_within_limit(self, sample_image_path,
                                      mock_vlm_result, mock_llm_result):
        """Full pipeline (mocked) must complete well within limit."""
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            t0     = time.time()
            result = run_pipeline(sample_image_path)
            elapsed = time.time() - t0
        assert elapsed <= TOTAL_MAX_TIME, (
            f"Pipeline took {elapsed:.1f}s — exceeds {TOTAL_MAX_TIME}s limit"
        )


# ─────────────────────────────────────────────────────────────
# TEST 2: CONSISTENCY — same image → same disease label
# ─────────────────────────────────────────────────────────────
class TestOutputConsistency:
    def test_vlm_same_image_same_label(self, sample_image_path):
        """
        PubMedCLIP is deterministic (no sampling) — the same image
        must produce the same primary disease label every time.
        """
        from models.vlm_inference import infer_vlm_with_label
        result1 = infer_vlm_with_label(sample_image_path)
        result2 = infer_vlm_with_label(sample_image_path)
        assert result1["disease_label"] == result2["disease_label"], (
            f"Inconsistent labels: '{result1['disease_label']}' vs '{result2['disease_label']}'"
        )

    def test_vlm_same_image_same_top_disease(self, sample_image_path):
        """Top-1 disease must be identical across two runs."""
        from models.vlm_inference import infer_vlm_with_label
        r1 = infer_vlm_with_label(sample_image_path)
        r2 = infer_vlm_with_label(sample_image_path)
        assert r1["top_diseases"][0][0] == r2["top_diseases"][0][0]

    def test_vlm_scores_stable(self, sample_image_path):
        """Confidence scores must be identical (deterministic model)."""
        from models.vlm_inference import infer_vlm_with_label
        r1 = infer_vlm_with_label(sample_image_path)
        r2 = infer_vlm_with_label(sample_image_path)
        for disease in r1["all_scores"]:
            diff = abs(r1["all_scores"][disease] - r2["all_scores"][disease])
            assert diff < 0.001, f"Score for {disease} changed: {diff}"

    def test_pipeline_consistent_disease_label(self, sample_image_path,
                                                mock_vlm_result, mock_llm_result):
        """
        Pipeline run twice on same image → same disease_label.
        (Uses mock to isolate consistency from LLM stochasticity.)
        """
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            r1 = run_pipeline(sample_image_path)
            r2 = run_pipeline(sample_image_path)
        assert r1["disease_label"] == r2["disease_label"]

    def test_pipeline_consistent_status(self, sample_image_path,
                                         mock_vlm_result, mock_llm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            r1 = run_pipeline(sample_image_path)
            r2 = run_pipeline(sample_image_path)
        assert r1["status"] == r2["status"] == "success"


# ─────────────────────────────────────────────────────────────
# TEST 3: MULTI-IMAGE STABILITY — 5 different images, no crash
# ─────────────────────────────────────────────────────────────
class TestMultiImageStability:
    def test_five_images_all_succeed(self):
        """System processes 5 different images without crashing."""
        images = sorted((ROOT / "data" / "processed" / "images").glob("*.png"))[:5]
        if len(images) < 2:
            pytest.skip("Need at least 2 processed images")

        from models.vlm_inference import infer_vlm_with_label
        errors = []
        for img_path in images:
            try:
                result = infer_vlm_with_label(str(img_path))
                assert result["success"] is True
            except Exception as e:
                errors.append(f"{img_path.name}: {e}")

        assert not errors, f"Errors on: {errors}"

    def test_five_images_response_times_reasonable(self):
        """Each image's VLM inference should stay within time limit."""
        images = sorted((ROOT / "data" / "processed" / "images").glob("*.png"))[:5]
        if not images:
            pytest.skip("No processed images")

        from models.vlm_inference import infer_vlm_with_label
        for img_path in images:
            result = infer_vlm_with_label(str(img_path))
            assert result["response_time"] <= VLM_MAX_TIME, (
                f"{img_path.name} took {result['response_time']}s"
            )


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                    str(Path(__file__)), "-v", "--tb=short"])
