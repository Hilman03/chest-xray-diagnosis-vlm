"""
diseases_to_dicom.py
─────────────────────────────────────────────────────────────────────────
Create ONE DICOM (.dcm) file per disease class in the NIH ChestX-ray14
dataset, numbered in order:

    pneumothorax_1.dcm, atelectasis_2.dcm, ...   (well, in label order:
    atelectasis_1.dcm, cardiomegaly_2.dcm, ... pneumothorax_14.dcm)

For each disease the script picks one chest x-ray that is labelled with
that disease (preferring an image whose ONLY finding is that disease, so
the example is "clean"), reads the matching PNG from data/raw/images_*,
and writes a proper DICOM with patient metadata + the finding label
embedded in the ImageComments tag — so the app's DICOM reader can pick
it up.

Run from the project root:
    python scripts/diseases_to_dicom.py
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage


# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent / "data"
RAW_DIR   = BASE_DIR / "raw"
CSV_PATH  = RAW_DIR / "Data_Entry_2017.csv"
OUT_DIR   = BASE_DIR / "dicom_diseases"

# 14 pathologies (we skip "No Finding" — it is not a disease).
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]


def build_image_index():
    """Map every available PNG filename -> its full path on disk."""
    index = {}
    for folder in sorted(RAW_DIR.glob("images_*")):
        for img in folder.glob("*.png"):
            index[img.name] = img
    # Some layouts nest under images_XXX/images/
    for img in RAW_DIR.glob("images_*/images/*.png"):
        index.setdefault(img.name, img)
    return index


def pick_image(df, disease, img_index):
    """Return (row, png_path) for one image showing `disease`, or None."""
    # NIH labels are pipe-joined, e.g. "Cardiomegaly|Effusion".
    has = df["Finding Labels"].str.contains(rf"(?:^|\|){disease}(?:\||$)",
                                            regex=True, na=False)
    cand = df[has]

    # Prefer an image whose ONLY finding is this disease (cleanest example).
    solo = cand[cand["Finding Labels"] == disease]
    for subset in (solo, cand):
        for _, row in subset.iterrows():
            png = img_index.get(row["Image Index"])
            if png is not None:
                return row, png
    return None


def write_dicom(row, png_path, out_path, label, number):
    img_arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)

    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID    = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    age = str(row.get("Patient Age", "")).strip()
    try:
        age_str = f"{int(float(age)):03d}Y"
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
    ds.StudyDescription  = f"{label} example"
    # Embed the ground-truth finding so the app/DICOM reader can surface it.
    ds.ImageComments     = str(row["Finding Labels"])

    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows        = img_arr.shape[0]
    ds.Columns     = img_arr.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored    = 8
    ds.HighBit       = 7
    ds.PixelRepresentation = 0
    ds.PixelData     = img_arr.tobytes()

    pydicom.dcmwrite(str(out_path), ds, write_like_original=False)


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    img_index = build_image_index()
    print(f"Indexed {len(img_index)} PNG images under {RAW_DIR}")

    made, missing = 0, []
    for number, disease in enumerate(DISEASES, start=1):
        picked = pick_image(df, disease, img_index)
        if picked is None:
            missing.append(disease)
            print(f"  [skip] {disease:<18} no matching image available locally")
            continue

        row, png = picked
        slug = disease.lower().replace(" ", "_")
        out  = OUT_DIR / f"{slug}_{number}.dcm"
        write_dicom(row, png, out, disease, number)
        made += 1
        print(f"  [ok]   {disease:<18} -> {out.name}   (from {row['Image Index']}, "
              f"labels: {row['Finding Labels']})")

    print(f"\nDone. {made} DICOM files written to {OUT_DIR}")
    if missing:
        print(f"No local image found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
