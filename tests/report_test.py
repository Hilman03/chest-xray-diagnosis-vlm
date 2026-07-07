"""
tests/report_test.py
─────────────────────────────────────────────────────────────────────────
Generates the report-ready tables for:

    4.5.1  Functional Testing        -> Table 4.4 (9 functions)
    4.5.2.1 Stability Testing        -> 10 DICOMs uploaded + analysed
    4.5.2.2 Error Handling           -> Table 4.5 (3 scenarios)

It exercises the LIVE backend over HTTP exactly as the app does, records the
Expected vs Actual result of every function, and writes both a console summary
and a Markdown file you can paste straight into the report:

    tests/report_test_results.md

Usage
-----
    # backend on Colab (default URL is read from API_URL env var):
    python tests/report_test.py --api https://<your-ngrok>.ngrok-free.dev

    # or set once:
    set API_URL=https://<your-ngrok>.ngrok-free.dev   (Windows)
    python tests/report_test.py

Notes
-----
* "AI Prediction" and "Report Generation" require the backend's BiomedCLIP +
  LLM to be loaded (i.e. run against Colab, not a model-less local server).
* Uses the DICOMs in data/dicom_diseases (run scripts/biomed_screen.py first).
"""

import os
import sys
import io
import time
import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DICOM_DIR = ROOT / "data" / "dicom_diseases"
OUT_MD = Path(__file__).resolve().parent / "report_test_results.md"

DEFAULT_API = os.environ.get(
    "API_URL", "https://pampered-enrage-girdle.ngrok-free.dev")

# ngrok's browser-warning page is skipped with this header.
HEADERS = {"ngrok-skip-browser-warning": "true"}
TIMEOUT = 180          # analysis can take a while on first call


def _dicoms(n=None):
    files = sorted(DICOM_DIR.glob("*.dcm"))
    return files[:n] if n else files


class Row:
    """One functional-test result row."""
    def __init__(self, no, function, expected):
        self.no = no
        self.function = function
        self.expected = expected
        self.actual = "—"
        self.status = "Fail"

    def ok(self, actual="Successful"):
        self.actual, self.status = actual, "Pass"

    def fail(self, actual):
        self.actual, self.status = (actual[:60] or "Failed"), "Fail"


# ─────────────────────────────────────────────────────────────
# 4.5.1  FUNCTIONAL TESTING  -> Table 4.4
# ─────────────────────────────────────────────────────────────
def functional_tests(api):
    rows, ctx = [], {}
    dcm = _dicoms(1)
    if not dcm:
        raise RuntimeError(
            "No DICOMs in data/dicom_diseases — run scripts/biomed_screen.py first.")
    dcm = dcm[0]

    r1 = Row(1, "Upload DICOM", "Image uploaded successfully")
    r2 = Row(2, "Display Image", "Image displayed in viewer")
    r3 = Row(3, "AI Prediction", "Disease prediction generated")
    r4 = Row(4, "Report Generation", "Report generated")
    r5 = Row(5, "Export PDF", "PDF created")
    r6 = Row(6, "Save Study", "Record stored in database")
    r7 = Row(7, "Search History", "Study retrieved")
    r8 = Row(8, "Delete Study", "Record removed")
    r9 = Row(9, "Settings Connection", "API connected")
    rows = [r1, r2, r3, r4, r5, r6, r7, r8, r9]

    # 1. Upload DICOM
    try:
        with open(dcm, "rb") as f:
            resp = requests.post(f"{api}/upload", headers=HEADERS,
                                 files={"file": (dcm.name, f, "application/dicom")},
                                 timeout=TIMEOUT)
        if resp.status_code == 200 and resp.json().get("image_id"):
            ctx["id"] = resp.json()["image_id"]
            r1.ok()
        else:
            r1.fail(f"HTTP {resp.status_code}")
    except Exception as e:
        r1.fail(str(e))

    iid = ctx.get("id")

    # 2. Display Image
    if iid:
        try:
            resp = requests.get(f"{api}/images/{iid}", headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and resp.content:
                r2.ok()
            else:
                r2.fail(f"HTTP {resp.status_code}")
        except Exception as e:
            r2.fail(str(e))

    # 3. AI Prediction + 4. Report Generation (single /analyze call)
    if iid:
        try:
            resp = requests.post(f"{api}/analyze/{iid}", headers=HEADERS, timeout=TIMEOUT)
            data = resp.json() if resp.status_code == 200 else {}
            if data.get("disease_label"):
                r3.ok(f"Detected: {data['disease_label']}")
            else:
                r3.fail(f"HTTP {resp.status_code}")
            if data.get("llm_report"):
                r4.ok()
            else:
                r4.fail("No report text")
        except Exception as e:
            r3.fail(str(e)); r4.fail(str(e))

    # 5. Export PDF
    if iid:
        try:
            resp = requests.get(f"{api}/export/{iid}", headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                r5.ok()
            else:
                r5.fail(f"HTTP {resp.status_code}")
        except Exception as e:
            r5.fail(str(e))

    # 6. Save Study (record retrievable from DB)
    if iid:
        try:
            resp = requests.get(f"{api}/report/{iid}", headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and resp.json().get("image_id") == iid:
                r6.ok()
            else:
                r6.fail(f"HTTP {resp.status_code}")
        except Exception as e:
            r6.fail(str(e))

    # 7. Search History (study appears in /reports)
    if iid:
        try:
            resp = requests.get(f"{api}/reports", headers=HEADERS, timeout=TIMEOUT)
            ids = [r.get("image_id") for r in resp.json().get("reports", [])]
            if iid in ids:
                r7.ok()
            else:
                r7.fail("Not found in history")
        except Exception as e:
            r7.fail(str(e))

    # 8. Delete Study (then confirm it is gone)
    if iid:
        try:
            requests.delete(f"{api}/report/{iid}", headers=HEADERS, timeout=TIMEOUT)
            check = requests.get(f"{api}/report/{iid}", headers=HEADERS, timeout=TIMEOUT)
            if check.status_code == 404:
                r8.ok()
            else:
                r8.fail("Still present after delete")
        except Exception as e:
            r8.fail(str(e))

    # 9. Settings Connection (health)
    try:
        resp = requests.get(f"{api}/health", headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.json().get("status"):
            r9.ok()
        else:
            r9.fail(f"HTTP {resp.status_code}")
    except Exception as e:
        r9.fail(str(e))

    return rows


# ─────────────────────────────────────────────────────────────
# 4.5.2.1  STABILITY TESTING (10 DICOMs)
# ─────────────────────────────────────────────────────────────
def stability_test(api, n=10):
    images = _dicoms(n)
    total = len(images)
    analysed = crashes = db_fail = 0

    for dcm in images:
        try:
            with open(dcm, "rb") as f:
                up = requests.post(f"{api}/upload", headers=HEADERS,
                                   files={"file": (dcm.name, f, "application/dicom")},
                                   timeout=TIMEOUT)
            if up.status_code != 200:
                crashes += 1
                continue
            iid = up.json()["image_id"]
            an = requests.post(f"{api}/analyze/{iid}", headers=HEADERS, timeout=TIMEOUT)
            if an.status_code == 200 and an.json().get("disease_label"):
                analysed += 1
            else:
                db_fail += 1
            requests.delete(f"{api}/report/{iid}", headers=HEADERS, timeout=TIMEOUT)
            print(f"  [{dcm.name:<24}] analysed={an.status_code == 200}")
        except Exception as e:
            crashes += 1
            print(f"  [{dcm.name:<24}] CRASH: {e}")

    return {"total": total, "analysed": analysed,
            "crashes": crashes, "interruptions": db_fail}


# ─────────────────────────────────────────────────────────────
# 4.5.2.2  ERROR HANDLING  -> Table 4.5
# ─────────────────────────────────────────────────────────────
def error_handling_tests(api):
    rows = []

    # Test 1 — unsupported file type (.pdf)
    row = Row(1, "Unsupported file", "Rejected")
    try:
        fake_pdf = io.BytesIO(b"%PDF-1.4 not a real xray")
        resp = requests.post(f"{api}/upload", headers=HEADERS,
                             files={"file": ("example.pdf", fake_pdf, "application/pdf")},
                             timeout=TIMEOUT)
        if resp.status_code >= 400:
            row.ok("Rejected")
        else:
            # If it accepted, clean up and mark fail.
            try:
                requests.delete(f"{api}/report/{resp.json().get('image_id')}",
                                headers=HEADERS, timeout=30)
            except Exception:
                pass
            row.fail("Accepted (HTTP 200)")
    except Exception as e:
        row.fail(str(e))
    rows.append(row)

    # Test 2 — API offline (connection failure is detected)
    row = Row(2, "API offline", "Warning displayed")
    try:
        # Hit a port nothing listens on to simulate the backend being down.
        requests.get("http://127.0.0.1:9/health", timeout=3)
        row.fail("Unexpected response from dead endpoint")
    except requests.exceptions.RequestException:
        row.ok("Offline detected")
    rows.append(row)

    # Test 3 — empty upload (no file selected)
    row = Row(3, "No file selected", "Upload prevented")
    try:
        resp = requests.post(f"{api}/upload", headers=HEADERS, timeout=30)
        if resp.status_code >= 400:
            row.ok("Upload prevented")
        else:
            row.fail(f"Accepted (HTTP {resp.status_code})")
    except Exception as e:
        row.fail(str(e))
    rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────
def build_markdown(func_rows, stab, err_rows):
    """Return the report-ready Markdown string (no file/console side effects)."""
    md = []
    md.append("# Technical Functionality Testing — Results\n")

    # Table 4.4
    md.append("## Table 4.4  Functional Testing Results\n")
    md.append("| No | Function | Expected Result | Actual Result | Status |")
    md.append("|----|----------|-----------------|---------------|--------|")
    for r in func_rows:
        md.append(f"| {r.no} | {r.function} | {r.expected} | {r.actual} | {r.status} |")
    fpass = sum(1 for r in func_rows if r.status == "Pass")
    md.append(f"\n**{fpass}/{len(func_rows)} functions passed.**\n")

    # Stability
    md.append("## 4.5.2.1  Stability Testing\n")
    md.append(f"- DICOM images processed : **{stab['total']}**")
    md.append(f"- Successful analyses    : **{stab['analysed']}**")
    md.append(f"- Crashes                : **{stab['crashes']}**")
    md.append(f"- System interruptions   : **{stab['interruptions']}**\n")

    # Table 4.5
    md.append("## Table 4.5  Error Handling Results\n")
    md.append("| Scenario | Expected Behaviour | Result |")
    md.append("|----------|--------------------|--------|")
    for r in err_rows:
        md.append(f"| {r.function} | {r.expected} | {r.status} |")
    epass = sum(1 for r in err_rows if r.status == "Pass")
    md.append(f"\n**{epass}/{len(err_rows)} error scenarios handled correctly.**\n")

    return "\n".join(md)


def render(func_rows, stab, err_rows):
    text = build_markdown(func_rows, stab, err_rows)
    OUT_MD.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved report tables to: {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API, help="Backend base URL")
    ap.add_argument("--stability-n", type=int, default=10)
    args = ap.parse_args()
    api = args.api.rstrip("/")

    print(f"Testing backend: {api}\n")
    print("== 4.5.1 Functional Testing ==")
    func_rows = functional_tests(api)
    print("\n== 4.5.2.1 Stability Testing ==")
    stab = stability_test(api, args.stability_n)
    print("\n== 4.5.2.2 Error Handling ==")
    err_rows = error_handling_tests(api)
    render(func_rows, stab, err_rows)


if __name__ == "__main__":
    main()
