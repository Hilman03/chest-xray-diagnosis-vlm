"""
pipeline.py
===========
Full Inference Pipeline — PubMedCLIP + LLaMA

Flow:
    CXR Image
        |
        ├─> PubMedCLIP  → disease predictions (image-text matching)
        |
        └─> LLaMA       → structured observational report

Can be run directly with VS Code Run button OR terminal.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import time
from models.vlm_inference import infer_vlm_with_label
from models.llm_refine import refine_llm


def run_pipeline(image_path: str) -> dict:
    """
    Full pipeline: CXR image -> PubMedCLIP -> LLaMA report

    Returns:
        {
            "image_path"    : str,
            "image_name"    : str,
            "disease_label" : str,   primary predicted disease
            "top_diseases"  : list,  top diseases with scores
            "all_scores"    : dict,  all disease scores
            "vlm_caption"   : str,   disease description
            "llm_report"    : str,   final structured report
            "llm_backend"   : str,
            "vlm_time"      : float,
            "llm_time"      : float,
            "total_time"    : float,
            "status"        : str,
            "error"         : str,
        }
    """
    start  = time.time()
    result = {
        "image_path"    : str(image_path),
        "image_name"    : Path(image_path).name,
        "disease_label" : "",
        "top_diseases"  : [],
        "all_scores"    : {},
        "vlm_caption"   : "",
        "llm_report"    : "",
        "llm_backend"   : "",
        "vlm_time"      : 0.0,
        "llm_time"      : 0.0,
        "total_time"    : 0.0,
        "status"        : "success",
        "error"         : "",
    }

    try:
        # Step 1 — PubMedCLIP disease prediction
        print(f"  [Pipeline] Step 1/2 — PubMedCLIP inference...")
        vlm_start   = time.time()
        vlm_result  = infer_vlm_with_label(str(image_path))
        vlm_elapsed = round(time.time() - vlm_start, 3)

        # A failed VLM returns success=False with an "Unknown" label — treat
        # it as an error instead of generating a bogus report downstream.
        if not vlm_result.get("success", False):
            raise RuntimeError(
                "PubMedCLIP inference failed — model unavailable or "
                "prediction error. Check that the model loaded correctly."
            )

        result["vlm_caption"]   = vlm_result["caption"]
        result["disease_label"] = vlm_result["disease_label"]
        result["top_diseases"]  = vlm_result["top_diseases"]
        result["all_scores"]    = vlm_result["all_scores"]
        result["vlm_time"]      = vlm_elapsed

        print(f"  [Pipeline] Primary Disease : {vlm_result['disease_label']}")
        print(f"  [Pipeline] Step 1 done in {vlm_elapsed}s")

        # Step 2 — LLM report generation
        print(f"  [Pipeline] Step 2/2 — LLM report generation...")
        llm_result = refine_llm(
            vlm_result["caption"],
            vlm_result["disease_label"],
            vlm_result["top_diseases"],
            image_path=str(image_path),
        )

        result["llm_report"]  = llm_result["report"]
        result["llm_backend"] = llm_result["backend"]
        result["llm_time"]    = llm_result["response_time"]
        print(f"  [Pipeline] Step 2 done in {llm_result['response_time']}s")

    except Exception as e:
        import traceback
        # Some exceptions carry an empty str(e), which hid the real cause
        # before — always include the exception type, and print the full
        # traceback so the failing line is visible in the logs.
        detail = f"{type(e).__name__}: {e}".strip().rstrip(":").strip()
        result["status"] = "error"
        result["error"]  = detail or type(e).__name__
        print(f"  [Pipeline] ERROR: {result['error']}")
        traceback.print_exc()

    result["total_time"] = round(time.time() - start, 3)
    print(f"  [Pipeline] Total: {result['total_time']}s")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Full Pipeline — PubMedCLIP + LLaMA")
    print("=" * 60)

    images_dir = ROOT / "data" / "processed" / "images"
    images     = sorted(images_dir.glob("*.png"))

    if not images:
        print(f"  No images found. Run scripts/preprocess.py first.")
        sys.exit(1)

    test_image = str(images[0])
    print(f"  Image : {Path(test_image).name}\n")

    result = run_pipeline(test_image)

    print("\n" + "=" * 60)
    print("  PIPELINE RESULT")
    print("=" * 60)
    print(f"  Status         : {result['status']}")
    print(f"  Image          : {result['image_name']}")
    print(f"  Primary Disease: {result['disease_label']}")
    print(f"\n  Top Predictions:")
    for disease, score in result["top_diseases"]:
        bar = "█" * int(score * 40)
        print(f"    {disease:<25} : {bar:<40} {score*100:.1f}%")
    print(f"\n  Description    : {result['vlm_caption']}")
    print(f"  LLM Backend    : {result['llm_backend']}")
    print(f"  VLM Time       : {result['vlm_time']}s")
    print(f"  LLM Time       : {result['llm_time']}s")
    print(f"  Total Time     : {result['total_time']}s")
    print(f"\n  Generated Report:")
    print(f"  {result['llm_report']}")
    print("=" * 60)