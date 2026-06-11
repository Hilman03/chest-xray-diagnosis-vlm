"""
tests/run_all_tests.py
======================
Master Test Runner — generates a structured evaluation report
matching Section 3.8 of the Final Report.

Runs all three test suites and saves results to:
    tests/test_report.json   (machine-readable)
    tests/test_report.txt    (human-readable summary)

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --quick   (skip real model tests)
"""

import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TESTS = [
    {
        "id"    : "T1",
        "name"  : "Technical Functionality",
        "file"  : "tests/test_functionality.py",
        "desc"  : "Image upload, VLM inference, LLM output, DICOM handling",
    },
    {
        "id"    : "T2",
        "name"  : "System Performance",
        "file"  : "tests/test_performance.py",
        "desc"  : "Response time, output consistency, multi-image stability",
    },
    {
        "id"    : "T3",
        "name"  : "Workflow Integration",
        "file"  : "tests/test_integration.py",
        "desc"  : "API endpoints: upload → analyze → report → export",
    },
    {
        "id"    : "T4",
        "name"  : "Unit Tests",
        "file"  : "tests/test_pipeline.py",
        "desc"  : "Prompts, VLM output keys, LLM mocked, pipeline logic",
    },
]


def run_pytest(test_file: str, quick: bool = False) -> dict:
    """Run pytest on a file and return parsed results."""
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v", "--tb=short",
        "--no-header",
        "--json-report", "--json-report-file=-",   # requires pytest-json-report
    ]

    # Fallback: run without JSON plugin and parse output
    cmd_simple = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v", "--tb=short", "--no-header",
    ]

    t0  = time.time()
    try:
        proc = subprocess.run(
            cmd_simple,
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        elapsed = round(time.time() - t0, 2)
        output  = proc.stdout + proc.stderr

        # Parse summary line  e.g. "5 passed, 2 failed in 3.45s"
        passed = failed = skipped = errors = 0
        for line in output.splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                import re
                nums = re.findall(r'(\d+)\s+(passed|failed|error|skipped)', line)
                for count, status in nums:
                    if status == "passed":   passed   = int(count)
                    elif status == "failed": failed   = int(count)
                    elif status == "error":  errors   = int(count)
                    elif status == "skipped":skipped  = int(count)

        total   = passed + failed + errors
        success = proc.returncode == 0

        return {
            "passed"  : passed,
            "failed"  : failed,
            "errors"  : errors,
            "skipped" : skipped,
            "total"   : total,
            "success" : success,
            "time_s"  : elapsed,
            "output"  : output[-3000:],   # last 3000 chars
        }

    except Exception as e:
        return {
            "passed": 0, "failed": 0, "errors": 1,
            "skipped": 0, "total": 1, "success": False,
            "time_s": round(time.time() - t0, 2),
            "output": str(e),
        }


def print_banner():
    print()
    print("=" * 65)
    print("  CXR DIAGNOSIS SYSTEM — EVALUATION TEST SUITE")
    print("  CSP650 Final Year Project — Muhammad Nurhilman")
    print("=" * 65)
    print(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Root   : {ROOT}")
    print("=" * 65)
    print()


def run_all(quick: bool = False) -> dict:
    print_banner()

    results   = []
    total_p   = total_f = total_e = total_s = 0
    t_start   = time.time()

    for test in TESTS:
        print(f"  [{test['id']}] {test['name']}")
        print(f"       {test['desc']}")
        print(f"       File: {test['file']}")
        print()

        res = run_pytest(test["file"], quick)

        total_p += res["passed"]
        total_f += res["failed"]
        total_e += res["errors"]
        total_s += res["skipped"]

        status_icon = "✅" if res["success"] else "❌"
        print(f"       {status_icon}  Passed: {res['passed']}  "
              f"Failed: {res['failed']}  "
              f"Errors: {res['errors']}  "
              f"Skipped: {res['skipped']}  "
              f"({res['time_s']}s)")
        print()

        results.append({**test, **res})

    total_time  = round(time.time() - t_start, 2)
    overall_ok  = total_f == 0 and total_e == 0

    print("=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Total Tests   : {total_p + total_f + total_e}")
    print(f"  Passed        : {total_p}")
    print(f"  Failed        : {total_f}")
    print(f"  Errors        : {total_e}")
    print(f"  Skipped       : {total_s}")
    print(f"  Duration      : {total_time}s")
    print(f"  Overall       : {'✅ PASS' if overall_ok else '❌ FAIL'}")
    print("=" * 65)

    # ── Evaluation criteria table (matches report Section 3.8) ──
    print()
    print("  EVALUATION CRITERIA (Section 3.8)")
    print("  " + "-" * 60)
    criteria = [
        ("Technical Functionality", "Image upload, inference, output generation", "T1"),
        ("System Performance",      "Response time, consistency",                 "T2"),
        ("Workflow Integration",    "Upload → Analyze → Report → Export",         "T3"),
    ]
    for criterion, description, test_id in criteria:
        match = next((r for r in results if r["id"] == test_id), None)
        if match:
            status = "PASS" if match["success"] else "FAIL"
            print(f"  {criterion:<28} [{status}]  {description}")
    print()

    # ── Save results ─────────────────────────────────────────────
    summary = {
        "project"        : "CXR Diagnosis System — VLM + LLM Pipeline",
        "author"         : "Muhammad Nurhilman Bin Mohd Rozalee",
        "student_id"     : "2024814584",
        "test_date"      : datetime.now().isoformat(),
        "overall_pass"   : overall_ok,
        "total_passed"   : total_p,
        "total_failed"   : total_f,
        "total_errors"   : total_e,
        "total_skipped"  : total_s,
        "total_time_s"   : total_time,
        "test_suites"    : results,
        "evaluation_criteria": {
            "technical_functionality": next(
                (r["success"] for r in results if r["id"] == "T1"), False),
            "system_performance": next(
                (r["success"] for r in results if r["id"] == "T2"), False),
            "workflow_integration": next(
                (r["success"] for r in results if r["id"] == "T3"), False),
        },
    }

    json_out = ROOT / "tests" / "test_report.json"
    txt_out  = ROOT / "tests" / "test_report.txt"

    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(txt_out, "w") as f:
        f.write("CXR DIAGNOSIS SYSTEM — EVALUATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Author     : Muhammad Nurhilman Bin Mohd Rozalee\n")
        f.write(f"Student ID : 2024814584\n")
        f.write(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Overall Result : {'PASS' if overall_ok else 'FAIL'}\n")
        f.write(f"Total Tests    : {total_p + total_f + total_e}\n")
        f.write(f"Passed         : {total_p}\n")
        f.write(f"Failed         : {total_f}\n")
        f.write(f"Errors         : {total_e}\n")
        f.write(f"Duration       : {total_time}s\n\n")
        f.write("EVALUATION CRITERIA (Section 3.8)\n")
        f.write("-" * 65 + "\n")
        for criterion, description, test_id in criteria:
            match = next((r for r in results if r["id"] == test_id), None)
            if match:
                status = "PASS" if match["success"] else "FAIL"
                f.write(f"{criterion:<30} [{status}]\n")
                f.write(f"  {description}\n")
                f.write(f"  Passed: {match['passed']}  "
                        f"Failed: {match['failed']}  "
                        f"Time: {match['time_s']}s\n\n")

    print(f"  Reports saved:")
    print(f"    {json_out}")
    print(f"    {txt_out}")
    print()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all evaluation tests")
    parser.add_argument("--quick", action="store_true",
                        help="Skip tests that require real model loading")
    args = parser.parse_args()
    summary = run_all(quick=args.quick)
    sys.exit(0 if summary["overall_pass"] else 1)
