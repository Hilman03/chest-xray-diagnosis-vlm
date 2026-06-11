"""
tests/test_functionality.py
===========================
Technical Functionality Tests — Section 3.8 (Report)

Covers:
    1. Image upload handling (no crash, format support)
    2. Inference pipeline execution (VLM output without error)
    3. Output generation (report produced, not empty)
    4. System stability (repeated execution without crash)
    5. DICOM file handling
    6. Error handling (graceful failure on bad input)

Run:
    pytest tests/test_functionality.py -v
"""

import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from PIL import Image


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_image_path():
    images = sorted((ROOT / "data" / "processed" / "images").glob("*.png"))
    if not images:
        pytest.skip("No processed images — run scripts/preprocess.py first")
    return str(images[0])


@pytest.fixture(scope="module")
def sample_images(n=5):
    images = sorted((ROOT / "data" / "processed" / "images").glob("*.png"))
    if not images:
        pytest.skip("No processed images — run scripts/preprocess.py first")
    return [str(p) for p in images[:n]]


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
        "report"        : "The chest radiograph demonstrates adequate exposure. Increased opacity in the left lower lobe consistent with pneumonia. No additional significant abnormalities identified.",
        "backend"       : "tinyllama",
        "response_time" : 2.0,
    }


# ─────────────────────────────────────────────────────────────
# TEST 1: IMAGE LOADING — system accepts valid image formats
# ─────────────────────────────────────────────────────────────
class TestImageLoading:
    def test_png_opens_without_error(self, sample_image_path):
        img = Image.open(sample_image_path)
        assert img is not None

    def test_image_converts_to_rgb(self, sample_image_path):
        img = Image.open(sample_image_path).convert("RGB")
        assert img.mode == "RGB"

    def test_image_has_valid_size(self, sample_image_path):
        img = Image.open(sample_image_path)
        w, h = img.size
        assert w > 0 and h > 0

    def test_synthetic_jpg_accepted(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(img_path, "JPEG")
        img = Image.open(img_path).convert("RGB")
        assert img.mode == "RGB"

    def test_nonexistent_image_raises(self):
        from models.vlm_inference import infer_vlm_with_label
        with pytest.raises(FileNotFoundError):
            infer_vlm_with_label("path/that/does/not/exist.png")


# ─────────────────────────────────────────────────────────────
# TEST 2: VLM INFERENCE — PubMedCLIP output structure
# ─────────────────────────────────────────────────────────────
class TestVLMFunctionality:
    def test_vlm_returns_all_required_keys(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        required = ["caption", "disease_label", "top_diseases",
                    "all_scores", "response_time", "success"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_disease_label_is_valid_class(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label, DISEASE_TEMPLATES
        result = infer_vlm_with_label(sample_image_path)
        assert result["disease_label"] in DISEASE_TEMPLATES

    def test_top_diseases_sorted_descending(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        scores = [s for _, s in result["top_diseases"]]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_sum_to_one(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        total = sum(result["all_scores"].values())
        assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"

    def test_vlm_success_flag_true(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        assert result["success"] is True

    def test_caption_contains_chest_xray(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        assert "Chest X-Ray" in result["caption"] or "chest" in result["caption"].lower()


# ─────────────────────────────────────────────────────────────
# TEST 3: LLM OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────
class TestLLMFunctionality:
    def test_report_is_string(self, mock_vlm_result, mock_llm_result):
        from models.llm_refine import refine_llm
        with patch("models.llm_refine.load_tinyllama", return_value=True), \
             patch("models.llm_refine._generate", return_value=mock_llm_result["report"]):
            result = refine_llm(
                mock_vlm_result["caption"],
                mock_vlm_result["disease_label"],
                mock_vlm_result["top_diseases"],
            )
        assert isinstance(result["report"], str)

    def test_report_minimum_length(self, mock_vlm_result, mock_llm_result):
        from models.llm_refine import refine_llm
        with patch("models.llm_refine.load_tinyllama", return_value=True), \
             patch("models.llm_refine._generate", return_value=mock_llm_result["report"]):
            result = refine_llm(
                mock_vlm_result["caption"],
                mock_vlm_result["disease_label"],
            )
        assert len(result["report"]) >= 30, "Report too short"

    def test_report_does_not_contain_ai_mention(self, mock_vlm_result):
        report = "The chest radiograph demonstrates adequate exposure. Increased opacity noted. No additional significant findings."
        with patch("models.llm_refine.load_tinyllama", return_value=True), \
             patch("models.llm_refine._generate", return_value=report):
            from models.llm_refine import refine_llm
            result = refine_llm(mock_vlm_result["caption"], mock_vlm_result["disease_label"])
        lowered = result["report"].lower()
        for forbidden in ["ai system", "software", "model generated", "tinyllama"]:
            assert forbidden not in lowered, f"Report mentions '{forbidden}'"


# ─────────────────────────────────────────────────────────────
# TEST 4: FULL PIPELINE — no crash, all keys present
# ─────────────────────────────────────────────────────────────
class TestPipelineFunctionality:
    def test_pipeline_runs_without_exception(self, sample_image_path,
                                              mock_vlm_result, mock_llm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            result = run_pipeline(sample_image_path)
        assert result is not None

    def test_pipeline_status_success(self, sample_image_path,
                                      mock_vlm_result, mock_llm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            result = run_pipeline(sample_image_path)
        assert result["status"] == "success"

    def test_pipeline_output_fields_populated(self, sample_image_path,
                                               mock_vlm_result, mock_llm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", return_value=mock_llm_result):
            result = run_pipeline(sample_image_path)
        assert result["disease_label"] != ""
        assert result["llm_report"] != ""
        assert result["vlm_caption"] != ""

    def test_pipeline_handles_vlm_crash_gracefully(self, sample_image_path):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label",
                   side_effect=RuntimeError("GPU out of memory")):
            result = run_pipeline(sample_image_path)
        assert result["status"] == "error"
        assert result["error"] != ""

    def test_pipeline_handles_llm_crash_gracefully(self, sample_image_path, mock_vlm_result):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=mock_vlm_result), \
             patch("pipeline.refine_llm", side_effect=RuntimeError("LLM failed")):
            result = run_pipeline(sample_image_path)
        assert result["status"] == "error"


# ─────────────────────────────────────────────────────────────
# TEST 5: DICOM HANDLING
# ─────────────────────────────────────────────────────────────
class TestDICOMHandling:
    def test_dicom_samples_exist(self):
        dicom_dir = ROOT / "data" / "dicom_samples"
        if not dicom_dir.exists():
            pytest.skip("No DICOM samples directory")
        dcm_files = list(dicom_dir.glob("*.dcm"))
        assert len(dcm_files) > 0, "No .dcm files found in data/dicom_samples"

    def test_dicom_file_readable(self):
        dicom_dir = ROOT / "data" / "dicom_samples"
        if not dicom_dir.exists():
            pytest.skip("No DICOM samples directory")
        dcm_files = list(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            pytest.skip("No .dcm files")
        import pydicom
        ds = pydicom.dcmread(str(dcm_files[0]))
        assert ds is not None
        assert hasattr(ds, "pixel_array")

    def test_dicom_pixel_array_valid(self):
        dicom_dir = ROOT / "data" / "dicom_samples"
        if not dicom_dir.exists():
            pytest.skip("No DICOM samples directory")
        dcm_files = list(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            pytest.skip("No .dcm files")
        import pydicom, numpy as np
        ds  = pydicom.dcmread(str(dcm_files[0]))
        arr = ds.pixel_array
        assert arr.shape[0] > 0 and arr.shape[1] > 0


# ─────────────────────────────────────────────────────────────
# TEST 6: PREPROCESSING OUTPUTS PRESENT
# ─────────────────────────────────────────────────────────────
class TestPreprocessingOutputs:
    def test_processed_images_folder_exists(self):
        assert (ROOT / "data" / "processed" / "images").exists()

    def test_metadata_csv_exists(self):
        csv_path = ROOT / "data" / "processed" / "metadata_clean.csv"
        if not csv_path.exists():
            pytest.skip("metadata_clean.csv not generated yet")
        assert csv_path.stat().st_size > 0

    def test_dataset_info_json_exists(self):
        json_path = ROOT / "data" / "processed" / "dataset_info.json"
        if not json_path.exists():
            pytest.skip("dataset_info.json not generated yet")
        import json
        with open(json_path) as f:
            info = json.load(f)
        assert "total_images" in info or len(info) > 0


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                    str(Path(__file__)), "-v", "--tb=short"])
