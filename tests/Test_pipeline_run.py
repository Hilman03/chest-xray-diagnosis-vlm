"""
tests/test_pipeline_run.py
==========================
Run BiomedCLIP pipeline on 10 sample images.
Output : tests/sample_outputs.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
from datetime import datetime
from pipeline import run_pipeline

IMAGES_DIR  = ROOT / "data" / "processed" / "images"
OUTPUT_FILE = ROOT / "tests" / "sample_outputs.json"
N_SAMPLES   = 10
TARGET_TIME = 15.0


def run_sample_test():
    print("=" * 60)
    print("  10 Image Sample Test — BiomedCLIP + LLaMA")
    print("=" * 60)

    all_images = sorted(IMAGES_DIR.glob("*.png"))

    if not all_images:
        print(f"  ERROR: No images in {IMAGES_DIR}")
        print(f"  Run scripts/preprocess.py first.")
        sys.exit(1)

    samples       = all_images[:N_SAMPLES]
    results       = []
    total_times   = []
    vlm_times     = []
    llm_times     = []
    success_count = 0
    error_count   = 0

    print(f"  Testing {len(samples)} images\n")

    for i, img_path in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {img_path.name}")
        print("-" * 50)

        result    = run_pipeline(str(img_path))
        time_flag = "OVER TARGET" if result["total_time"] > TARGET_TIME else "OK"

        total_times.append(result["total_time"])
        vlm_times.append(result["vlm_time"])
        llm_times.append(result["llm_time"])

        if result["status"] == "success":
            success_count += 1
        else:
            error_count += 1

        print(f"  Disease : {result['disease_label']}")
        print(f"  Time    : {result['total_time']}s [{time_flag}]")
        print(f"  Report  : {result['llm_report'][:80]}...")

        results.append({
            "index"        : i,
            "image_name"   : img_path.name,
            "disease_label": result["disease_label"],
            "top_diseases" : result["top_diseases"],
            "vlm_caption"  : result["vlm_caption"],
            "llm_report"   : result["llm_report"],
            "llm_backend"  : result["llm_backend"],
            "vlm_time"     : result["vlm_time"],
            "llm_time"     : result["llm_time"],
            "total_time"   : result["total_time"],
            "status"       : result["status"],
            "error"        : result["error"],
            "time_flag"    : time_flag,
        })

    avg_total   = round(sum(total_times) / len(total_times), 3)
    avg_vlm     = round(sum(vlm_times)   / len(vlm_times),   3)
    avg_llm     = round(sum(llm_times)   / len(llm_times),   3)
    over_target = sum(1 for t in total_times if t > TARGET_TIME)

    summary = {
        "test_date"          : datetime.now().isoformat(),
        "approach"           : "BiomedCLIP + LLaMA",
        "model"              : "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "total_images_tested": len(samples),
        "success_count"      : success_count,
        "error_count"        : error_count,
        "target_time_per_img": TARGET_TIME,
        "images_over_target" : over_target,
        "timing": {
            "avg_total_time": avg_total,
            "avg_vlm_time"  : avg_vlm,
            "avg_llm_time"  : avg_llm,
            "max_total_time": round(max(total_times), 3),
            "min_total_time": round(min(total_times), 3),
        },
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  SAMPLE TEST COMPLETE")
    print("=" * 60)
    print(f"  Images tested  : {len(samples)}")
    print(f"  Successful     : {success_count}")
    print(f"  Errors         : {error_count}")
    print(f"  Avg total time : {avg_total}s")
    print(f"  Over target    : {over_target}/{len(samples)}")
    print(f"  Saved to       : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_sample_test()