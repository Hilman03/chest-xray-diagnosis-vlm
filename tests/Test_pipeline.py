"""
tests/test_pipeline.py
======================
Unit tests for PubMedCLIP + LLaMA pipeline.

Run directly with VS Code OR terminal:
    pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from unittest.mock import patch


@pytest.fixture
def sample_image_path():
    images = sorted((ROOT / "data" / "processed" / "images").glob("*.png"))
    if not images:
        pytest.skip("No processed images — run phase1_preprocess.py first")
    return str(images[0])


@pytest.fixture
def sample_caption():
    return "Chest X-Ray showing pneumonia with lobar opacity and consolidation"


@pytest.fixture
def sample_label():
    return "Pneumonia"


@pytest.fixture
def sample_top_diseases():
    return [("Pneumonia", 0.72), ("Effusion", 0.18), ("No Finding", 0.10)]


# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────
class TestPrompts:
    def test_report_contains_caption(self, sample_caption, sample_label):
        from models.prompts import get_report_prompt
        assert sample_caption in get_report_prompt(sample_caption, sample_label)

    def test_report_contains_disease(self, sample_caption, sample_label):
        from models.prompts import get_report_prompt
        assert sample_label in get_report_prompt(sample_caption, sample_label)

    def test_report_contains_scores(self, sample_caption, sample_label,
                                    sample_top_diseases):
        from models.prompts import get_report_prompt
        prompt = get_report_prompt(sample_caption, sample_label, sample_top_diseases)
        assert "72.0%" in prompt or "72%" in prompt

    def test_ollama_has_inst_tags(self, sample_caption, sample_label):
        from models.prompts import get_ollama_prompt
        prompt = get_ollama_prompt(sample_caption, sample_label)
        assert "[INST]" in prompt and "[/INST]" in prompt

    def test_system_prompt_not_empty(self):
        from models.prompts import SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 20


# ─────────────────────────────────────────────────────────────
# VLM
# ─────────────────────────────────────────────────────────────
class TestVLMInference:
    def test_missing_image_raises(self):
        from models.vlm_inference import infer_vlm_with_label
        with pytest.raises(FileNotFoundError):
            infer_vlm_with_label("nonexistent/image.png")

    def test_returns_all_keys(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        for key in ["caption", "disease_label", "top_diseases",
                    "all_scores", "response_time", "success"]:
            assert key in result

    def test_caption_is_string(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 0

    def test_disease_label_not_empty(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        assert len(result["disease_label"]) > 0

    def test_top_diseases_is_list(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        assert isinstance(result["top_diseases"], list)
        assert len(result["top_diseases"]) > 0

    def test_scores_between_0_and_1(self, sample_image_path):
        from models.vlm_inference import infer_vlm_with_label
        result = infer_vlm_with_label(sample_image_path)
        for _, score in result["top_diseases"]:
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────
# LLM (mocked)
# ─────────────────────────────────────────────────────────────
class TestLLMRefine:
    def test_returns_dict(self, sample_caption, sample_label, sample_top_diseases):
        from models.llm_refine import refine_llm
        mock_out = [{"generated_text": "Good image. Pneumonia findings. No other issues."}]
        with patch("models.llm_refine._is_ollama_running", return_value=False), \
             patch("models.llm_refine._load_hf_pipeline"), \
             patch("models.llm_refine._hf_pipeline", return_value=mock_out):
            result = refine_llm(sample_caption, sample_label, sample_top_diseases)
        assert "report" in result
        assert "backend" in result
        assert "response_time" in result

    def test_backend_valid(self, sample_caption, sample_label):
        from models.llm_refine import refine_llm
        mock_out = [{"generated_text": "Test output."}]
        with patch("models.llm_refine._is_ollama_running", return_value=False), \
             patch("models.llm_refine._load_hf_pipeline"), \
             patch("models.llm_refine._hf_pipeline", return_value=mock_out):
            result = refine_llm(sample_caption, sample_label)
        assert result["backend"] in ("ollama", "huggingface")


# ─────────────────────────────────────────────────────────────
# PIPELINE (mocked)
# ─────────────────────────────────────────────────────────────
class TestPipeline:
    def _mock_vlm(self):
        return {
            "caption"       : "Chest X-Ray showing pneumonia with opacity",
            "disease_label" : "Pneumonia",
            "top_diseases"  : [("Pneumonia", 0.72), ("Effusion", 0.18)],
            "all_scores"    : {"Pneumonia": 0.72, "Effusion": 0.18},
            "image_name"    : "test.png",
            "response_time" : 2.0,
            "success"       : True,
        }

    def _mock_llm(self):
        return {
            "report"        : "Good image. Pneumonia findings visible. High confidence.",
            "backend"       : "ollama",
            "response_time" : 1.0,
        }

    def test_has_all_keys(self, sample_image_path):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=self._mock_vlm()), \
             patch("pipeline.refine_llm", return_value=self._mock_llm()):
            result = run_pipeline(sample_image_path)
        for key in ["image_path", "image_name", "disease_label", "top_diseases",
                    "all_scores", "vlm_caption", "llm_report", "llm_backend",
                    "vlm_time", "llm_time", "total_time", "status", "error"]:
            assert key in result

    def test_success_status(self, sample_image_path):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=self._mock_vlm()), \
             patch("pipeline.refine_llm", return_value=self._mock_llm()):
            result = run_pipeline(sample_image_path)
        assert result["status"] == "success"

    def test_disease_stored(self, sample_image_path):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=self._mock_vlm()), \
             patch("pipeline.refine_llm", return_value=self._mock_llm()):
            result = run_pipeline(sample_image_path)
        assert result["disease_label"] == "Pneumonia"

    def test_error_handling(self):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label",
                   side_effect=Exception("Model failed")):
            result = run_pipeline("any/image.png")
        assert result["status"] == "error"
        assert "Model failed" in result["error"]

    def test_total_time_positive(self, sample_image_path):
        from pipeline import run_pipeline
        with patch("pipeline.infer_vlm_with_label", return_value=self._mock_vlm()), \
             patch("pipeline.refine_llm", return_value=self._mock_llm()):
            result = run_pipeline(sample_image_path)
        assert result["total_time"] > 0


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                    str(Path(__file__)), "-v", "--tb=short"])