"""
biomed_screen.py
─────────────────────────────────────────────────────────────────────────
Same screening as scripts/pubmed_screen.py, but using BiomedCLIP (with
prompt-ensembling) instead of PubMedCLIP. Lets us compare which diseases each
zero-shot VLM can actually detect.

Outputs:
  data/biomed_screen_report.txt   ← what BiomedCLIP can / cannot detect

Run (slow on CPU — intended for background):
    python scripts/biomed_screen.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.vlm_inference import predict_diseases  # noqa: E402

DATA     = ROOT / "data"
RAW_DIR  = DATA / "raw"
CSV_PATH = RAW_DIR / "Data_Entry_2017.csv"
REPORT   = DATA / "biomed_screen_report.txt"
DICOM_DIR = DATA / "dicom_diseases"      # DICOMs are (re)generated here

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]
MAX_CANDIDATES   = 25   # images tested per disease
KEEP_PER_DISEASE = 3    # DICOMs written per disease from TOP-1 passes


def write_dicom(row, png_path, out_path, label, number):
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID    = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    try:
        age_str = f"{int(float(row.get('Patient Age', 0))):03d}Y"
    except (ValueError, TypeError):
        age_str = "000Y"
    ds.PatientName  = f"PATIENT_{row['Patient ID']}"
    ds.PatientID    = str(row["Patient ID"])
    ds.PatientAge   = age_str
    ds.PatientSex   = str(row.get("Patient Gender", "O")).strip() or "O"
    ds.StudyDate    = datetime.now().strftime("%Y%m%d")
    ds.StudyTime    = datetime.now().strftime("%H%M%S")
    ds.StudyInstanceUID  = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID    = ds.file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID       = SecondaryCaptureImageStorage
    ds.Modality          = "CR"
    ds.ViewPosition      = str(row.get("View Position", "")).strip()
    ds.SeriesNumber      = number
    ds.SeriesDescription = label
    ds.StudyDescription  = f"{label} (BiomedCLIP-verified)"
    ds.ImageComments     = str(row["Finding Labels"])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    pydicom.dcmwrite(str(out_path), ds, write_like_original=False)


def build_image_index():
    index = {}
    for img in RAW_DIR.glob("images_*/*.png"):
        index[img.name] = img
    for img in RAW_DIR.glob("images_*/images/*.png"):
        index.setdefault(img.name, img)
    return index


def main():
    df = pd.read_csv(CSV_PATH)
    idx = build_image_index()
    print(f"Indexed {len(idx)} images. Testing up to {MAX_CANDIDATES}/disease.\n")

    # Fresh DICOM output: clear any previous set so only verified studies remain.
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    for old in DICOM_DIR.glob("*.dcm"):
        old.unlink()

    detectable, not_detectable = [], []
    lines = ["BiomedCLIP screening report (prompt-ensembled)",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Tested up to {MAX_CANDIDATES} single-label images per disease.",
             "=" * 70]
    written_n = 0

    for disease in DISEASES:
        solo = df[df["Finding Labels"] == disease]
        cands = [(r, idx[r["Image Index"]]) for _, r in solo.iterrows()
                 if r["Image Index"] in idx][:MAX_CANDIDATES]

        top1 = top3 = tested = 0
        kept = []
        for row, png in cands:
            res = predict_diseases(str(png))
            if not res.get("success"):
                continue
            tested += 1
            names = [d for d, _ in res["top_diseases"]]
            if res["primary_label"] == disease:
                top1 += 1
                if len(kept) < KEEP_PER_DISEASE:
                    kept.append((row, png))
            if disease in names:
                top3 += 1

        # Write DICOMs only for studies BiomedCLIP detects correctly (top-1).
        for row, png in kept:
            written_n += 1
            write_dicom(row, png, DICOM_DIR / f"{disease.lower()}_{written_n}.dcm",
                        disease, written_n)

        status = "DETECTABLE" if top1 > 0 else ("WEAK" if top3 > 0 else "NOT DETECTABLE")
        (detectable if top1 > 0 else not_detectable).append(disease)
        line = (f"{disease:<20} {status:<15} top1={top1}/{tested}  "
                f"top3={top3}/{tested}  kept={len(kept)}")
        print(line)
        lines.append(line)

    lines += ["=" * 70,
              f"DETECTABLE (top-1 hit)     : {', '.join(detectable) or 'none'}",
              f"NOT detectable (top-1 miss): {', '.join(not_detectable) or 'none'}",
              f"DICOMs written to {DICOM_DIR} : {written_n}"]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-4:]))
    print(f"\nReport saved: {REPORT}")


if __name__ == "__main__":
    main()
