"""
preprocess.py
=============
Phase 1 - Dataset & Preprocessing for NIH ChestX-ray14

Expected folder structure:
    data/
        raw/
            images_001/
            images_002/
            ...
            images_012/
            Data_Entry_2017.csv
            BBox_List_2017.csv

Run from project root:
    python scripts/preprocess.py

Install dependencies first:
    pip install -r requirements.txt
"""

import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, UnidentifiedImageError
from pathlib import Path
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: opencv-python not installed. Run: pip install opencv-python")
    print("Falling back to PIL-only processing.\n")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\hilma\OneDrive\Documents\DEGREE\SEM 6\CSP650\CODE\data")
RAW_DIR       = BASE_DIR / "raw"
CSV_ENTRY     = RAW_DIR / "Data_Entry_2017.csv"
CSV_BBOX      = RAW_DIR / "BBox_List_2017.csv"
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_IMAGES = PROCESSED_DIR / "images"
DICOM_DIR     = BASE_DIR / "dicom_samples"

TARGET_SIZE   = (224, 224)
MAX_VALID_AGE = 120

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Image processing parameters
GAUSSIAN_BLUR_RADIUS  = 1.0
MEDIAN_FILTER_SIZE    = 3
BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
CLAHE_CLIP_LIMIT      = 2.0
CLAHE_TILE_GRID_SIZE  = (8, 8)
SHARPEN_RADIUS        = 2
SHARPEN_PERCENT       = 150
SHARPEN_THRESHOLD     = 3

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "No Finding", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
def setup_logging():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase1")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(PROCESSED_DIR / "preprocessing_log.txt", mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ─────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────
def clean_previous_outputs():
    print("=" * 60)
    print("  Cleaning previous outputs...")
    print("=" * 60)

    if OUTPUT_IMAGES.exists():
        shutil.rmtree(OUTPUT_IMAGES)
        print(f"  Deleted : {OUTPUT_IMAGES}")

    if DICOM_DIR.exists():
        shutil.rmtree(DICOM_DIR)
        print(f"  Deleted : {DICOM_DIR}")

    for old_file in [
        PROCESSED_DIR / "metadata_clean.csv",
        PROCESSED_DIR / "dataset_info.json",
        PROCESSED_DIR / "preprocessing_log.txt",
    ]:
        if old_file.exists():
            old_file.unlink()
            print(f"  Deleted : {old_file.name}")

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    print("  Fresh output folders created\n")


# ═════════════════════════════════════════════════════════════
# STEP 1 - Load CSVs
# ═════════════════════════════════════════════════════════════
def load_csvs(log):
    log.info("=" * 60)
    log.info("STEP 1 - Loading CSV files")
    log.info("=" * 60)

    df = pd.read_csv(CSV_ENTRY)
    log.info(f"  Data_Entry_2017  raw shape : {df.shape}")

    df = df.rename(columns={
        "Image Index"                : "image_index",
        "Finding Labels"             : "finding_labels",
        "Follow-up #"                : "follow_up_num",
        "Patient ID"                 : "patient_id",
        "Patient Age"                : "patient_age",
        "Patient Gender"             : "patient_gender",
        "View Position"              : "view_position",
        "OriginalImage[Width"        : "img_width",
        "Height]"                    : "img_height",
        "OriginalImagePixelSpacing[x": "pixel_spacing_x",
        "y]"                         : "pixel_spacing_y",
    })
    df = df.drop(columns=["Unnamed: 11"], errors="ignore")

    bb = pd.read_csv(CSV_BBOX)
    log.info(f"  BBox_List_2017   raw shape : {bb.shape}")

    bb = bb.rename(columns={
        "Image Index"  : "image_index",
        "Finding Label": "finding_label",
        "Bbox [x"      : "bbox_x",
        "y"            : "bbox_y",
        "w"            : "bbox_w",
        "h]"           : "bbox_h",
    })
    bb = bb.drop(columns=["Unnamed: 6", "Unnamed: 7", "Unnamed: 8"], errors="ignore")

    log.info("  CSVs loaded successfully")
    return df, bb


# ═════════════════════════════════════════════════════════════
# STEP 2 - Clean metadata
# ═════════════════════════════════════════════════════════════
def clean_metadata(df, log):
    log.info("=" * 60)
    log.info("STEP 2 - Cleaning metadata")
    log.info("=" * 60)
    report = {}

    n_before = len(df)
    df = df.drop_duplicates(subset=["image_index"])
    n_dup = n_before - len(df)
    log.info(f"  Duplicates removed         : {n_dup}")
    report["duplicates_removed"] = n_dup

    bad_age = df["patient_age"] > MAX_VALID_AGE
    log.info(f"  Implausible ages (>{MAX_VALID_AGE})  : {bad_age.sum()} -> set NaN")
    df.loc[bad_age, "patient_age"] = np.nan
    report["bad_ages_nulled"] = int(bad_age.sum())

    df["patient_gender"] = df["patient_gender"].str.strip().str.upper()
    invalid_g = ~df["patient_gender"].isin({"M", "F"})
    df.loc[invalid_g, "patient_gender"] = np.nan
    log.info(f"  Invalid gender values      : {invalid_g.sum()} -> set NaN")

    df["view_position"] = df["view_position"].str.strip().str.upper()
    log.info(f"  View positions             : {df['view_position'].value_counts().to_dict()}")

    bad_sp = (df["pixel_spacing_x"] <= 0) | (df["pixel_spacing_y"] <= 0)
    df = df[~bad_sp]
    log.info(f"  Invalid pixel spacing rows : {bad_sp.sum()} -> dropped")

    df["labels_list"] = df["finding_labels"].apply(
        lambda x: [l.strip() for l in str(x).split("|")]
    )

    for label in ALL_LABELS:
        col = label.lower().replace(" ", "_").replace("-", "_")
        df[col] = df["labels_list"].apply(lambda lst: int(label in lst))

    log.info(f"  Clean metadata shape       : {df.shape}")
    log.info("  Label distribution:")
    for label in ALL_LABELS:
        col = label.lower().replace(" ", "_").replace("-", "_")
        log.info(f"    {label:<22}: {df[col].sum():>6,}")

    return df, report


# ═════════════════════════════════════════════════════════════
# STEP 3 - Build image index
# ═════════════════════════════════════════════════════════════
def build_image_index(log):
    log.info("=" * 60)
    log.info("STEP 3 - Scanning images_001 to images_012")
    log.info("=" * 60)

    index = {}
    for folder_num in range(1, 13):
        folder_name = f"images_{folder_num:03d}"
        folder_path = RAW_DIR / folder_name

        if not folder_path.exists():
            log.warning(f"  Not found - skipping : {folder_path}")
            continue

        pngs = list(folder_path.glob("*.png"))
        log.info(f"  {folder_name}  ->  {len(pngs):,} images")

        for fpath in pngs:
            index[fpath.name] = str(fpath)

    log.info(f"  Total images indexed : {len(index):,}")
    return index


# ═════════════════════════════════════════════════════════════
# STEP 4 - Match metadata to disk
# ═════════════════════════════════════════════════════════════
def match_images(df, image_index, log):
    log.info("=" * 60)
    log.info("STEP 4 - Matching metadata to disk images")
    log.info("=" * 60)

    df = df.copy()
    df["image_path"] = df["image_index"].map(image_index)

    n_found   = df["image_path"].notnull().sum()
    n_missing = df["image_path"].isnull().sum()
    log.info(f"  Matched to disk            : {n_found:,}")
    log.info(f"  Missing on disk            : {n_missing:,}")

    df = df[df["image_path"].notnull()].copy()
    log.info(f"  Rows kept                  : {len(df):,}")
    return df


# ═════════════════════════════════════════════════════════════
# STEP 4b - Select 100 best samples
# ═════════════════════════════════════════════════════════════
def select_100_samples(df, log):
    log.info("=" * 60)
    log.info("STEP 4b - Selecting 100 best samples")
    log.info("=" * 60)

    TARGET_PER_LABEL = 7
    selected_indices = set()
    disease_cols = [l.lower().replace(" ", "_").replace("-", "_") for l in ALL_LABELS]

    for label in ALL_LABELS:
        col = label.lower().replace(" ", "_").replace("-", "_")

        single = df[
            (df[col] == 1) &
            (df[disease_cols].sum(axis=1) == 1)
        ]

        pa   = single[single["view_position"] == "PA"]
        pool = pa if len(pa) >= TARGET_PER_LABEL else single

        picked = pool.sample(n=min(TARGET_PER_LABEL, len(pool)), random_state=42)
        selected_indices.update(picked.index.tolist())
        log.info(f"  {label:<22} -> {len(picked):>2} samples")

    if len(selected_indices) < 100:
        remaining = df[~df.index.isin(selected_indices)]
        extra = remaining.sample(
            n=min(100 - len(selected_indices), len(remaining)),
            random_state=42
        )
        selected_indices.update(extra.index.tolist())
        log.info(f"  Filled {len(extra)} remaining slots")

    df_100 = df.loc[list(selected_indices)[:100]].copy()
    log.info(f"\n  Final selection : {len(df_100)} images")
    return df_100


# ═════════════════════════════════════════════════════════════
# IMAGE PROCESSING FUNCTIONS
# ═════════════════════════════════════════════════════════════
def process_single_image(src):
    img = Image.open(src)

    # 1. Convert to Grayscale
    img = img.convert("L")

    # 2. Noise Removal - Gaussian Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_BLUR_RADIUS))

    # 3. Noise Removal - Median Filter
    img = img.filter(ImageFilter.MedianFilter(size=MEDIAN_FILTER_SIZE))

    # 4. Noise Removal - Bilateral Filter (OpenCV) or fallback
    if CV2_AVAILABLE:
        arr = np.array(img, dtype=np.uint8)
        arr = cv2.bilateralFilter(arr, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        img = Image.fromarray(arr)
    else:
        img = img.filter(ImageFilter.SMOOTH_MORE)

    # 5. Contrast Enhancement - CLAHE (OpenCV) or fallback
    if CV2_AVAILABLE:
        arr   = np.array(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        arr   = clahe.apply(arr)
        img   = Image.fromarray(arr)
    else:
        enhancer = ImageEnhance.Contrast(img)
        img      = enhancer.enhance(1.5)

    # 6. Sharpening - Unsharp Mask
    img = img.filter(ImageFilter.UnsharpMask(
        radius=SHARPEN_RADIUS,
        percent=SHARPEN_PERCENT,
        threshold=SHARPEN_THRESHOLD,
    ))

    # 7. Resize to 224x224
    img = img.resize(TARGET_SIZE, Image.LANCZOS)

    # 8. Convert back to RGB for VLM input
    img = img.convert("RGB")

    return img


# ═════════════════════════════════════════════════════════════
# STEP 5 - Process all images
# ═════════════════════════════════════════════════════════════
def preprocess_images(df, log):
    log.info("=" * 60)
    log.info("STEP 5 - Image Processing Pipeline")
    log.info("=" * 60)
    log.info(f"  Output folder    : {OUTPUT_IMAGES}")
    log.info(f"  Target size      : {TARGET_SIZE}")
    log.info(f"  OpenCV available : {CV2_AVAILABLE}")
    log.info("")
    log.info("  Pipeline: Grayscale -> Gaussian -> Median -> Bilateral")
    log.info("         -> CLAHE -> Sharpen -> Resize 224x224 -> RGB")
    log.info("")

    processed_paths = []
    skipped         = []
    total           = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        src = row["image_path"]
        dst = OUTPUT_IMAGES / row["image_index"]

        log.info(f"  [{i:>3}/{total}]  {row['image_index']}  ->  {row['finding_labels']}")

        try:
            img = process_single_image(src)
            img.save(str(dst), format="PNG")
            processed_paths.append(str(dst))

        except (UnidentifiedImageError, OSError, Exception) as e:
            log.warning(f"  SKIP {row['image_index']} - {e}")
            skipped.append(row["image_index"])
            processed_paths.append(None)

    df = df.copy()
    df["processed_path"] = processed_paths
    df = df[df["processed_path"].notnull()].copy()

    log.info(f"\n  Successfully processed : {len(df)}")
    log.info(f"  Skipped               : {len(skipped)}")
    return df, skipped


# ═════════════════════════════════════════════════════════════
# STEP 6 - Merge bounding boxes
# ═════════════════════════════════════════════════════════════
def merge_bbox(df, bb, log):
    log.info("=" * 60)
    log.info("STEP 6 - Merging bounding box annotations")
    log.info("=" * 60)

    bb_grouped = (
        bb.groupby("image_index")
        .apply(lambda g: g[["finding_label", "bbox_x", "bbox_y",
                             "bbox_w", "bbox_h"]].to_dict(orient="records"))
        .reset_index()
        .rename(columns={0: "bboxes"})
    )

    df = df.merge(bb_grouped, on="image_index", how="left")
    df["bboxes"] = df["bboxes"].apply(lambda x: x if isinstance(x, list) else [])

    n_bbox = (df["bboxes"].apply(len) > 0).sum()
    log.info(f"  Images with bbox : {n_bbox}")
    return df


# ═════════════════════════════════════════════════════════════
# STEP 7 - Save outputs
# ═════════════════════════════════════════════════════════════
def save_outputs(df, cleaning_report, skipped, log):
    log.info("=" * 60)
    log.info("STEP 7 - Saving outputs")
    log.info("=" * 60)

    csv_cols = [c for c in df.columns if c not in ("labels_list", "bboxes", "image_path")]
    csv_path = PROCESSED_DIR / "metadata_clean.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    log.info(f"  Saved metadata_clean.csv ({len(df)} rows)")

    label_dist = {}
    for label in ALL_LABELS:
        col = label.lower().replace(" ", "_").replace("-", "_")
        label_dist[label] = int(df[col].sum()) if col in df.columns else 0

    sample_records = []
    for _, row in df.head(5).iterrows():
        sample_records.append({
            "image_index"   : row["image_index"],
            "finding_labels": row["finding_labels"],
            "patient_age"   : row["patient_age"] if not pd.isna(row["patient_age"]) else None,
            "patient_gender": row["patient_gender"],
            "view_position" : row["view_position"],
            "processed_path": row["processed_path"],
        })

    age_stats = df["patient_age"].describe()

    info = {
        "generated_at"        : datetime.now().isoformat(),
        "dataset_name"        : "NIH ChestX-ray14",
        "demo_mode"           : True,
        "total_images"        : len(df),
        "target_size"         : list(TARGET_SIZE),
        "image_mode"          : "RGB",
        "processing_pipeline" : [
            "1. Grayscale conversion",
            "2. Gaussian blur (radius=1.0)",
            "3. Median filter (size=3)",
            "4. Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)",
            "5. CLAHE contrast enhancement (clipLimit=2.0, grid=8x8)",
            "6. Unsharp mask sharpening (radius=2, percent=150, threshold=3)",
            "7. Resize to 224x224 (LANCZOS)",
            "8. Convert to RGB",
        ],
        "normalization"       : {
            "note": "Apply at inference time - NOT baked into PNGs",
            "mean": IMAGENET_MEAN,
            "std" : IMAGENET_STD,
        },
        "label_distribution"  : label_dist,
        "view_positions"      : df["view_position"].value_counts().to_dict(),
        "gender_distribution" : df["patient_gender"].value_counts().to_dict(),
        "age_stats"           : {
            "mean"  : round(float(age_stats["mean"]), 2),
            "std"   : round(float(age_stats["std"]),  2),
            "min"   : round(float(age_stats["min"]),  2),
            "max"   : round(float(age_stats["max"]),  2),
            "median": round(float(age_stats["50%"]),  2),
        },
        "images_with_bbox"    : int((df["bboxes"].apply(len) > 0).sum()),
        "skipped_images"      : skipped,
        "cleaning_summary"    : cleaning_report,
        "sample_records"      : sample_records,
    }

    info_path = PROCESSED_DIR / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    log.info("  Saved dataset_info.json")

    return info


# ═════════════════════════════════════════════════════════════
# STEP 8 - Convert to DICOM
# ═════════════════════════════════════════════════════════════
def convert_to_dicom(df, log, n_samples=10):
    log.info("=" * 60)
    log.info("STEP 8 - Converting sample images to DICOM")
    log.info("=" * 60)

    try:
        import pydicom
        from pydicom.dataset import Dataset
        from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    except ImportError:
        log.warning("  pydicom not installed - skipping. Run: pip install pydicom")
        return

    disease_cols = [l.lower().replace(" ", "_").replace("-", "_") for l in ALL_LABELS]
    samples = df.drop_duplicates(subset=disease_cols).head(n_samples)

    for _, row in samples.iterrows():
        src = row["processed_path"]
        out = DICOM_DIR / (Path(row["image_index"]).stem + ".dcm")

        img     = Image.open(src).convert("L")
        img_arr = np.array(img, dtype=np.uint8)

        ds = Dataset()
        ds.file_meta                            = Dataset()
        ds.file_meta.MediaStorageSOPClassUID    = pydicom.uid.SecondaryCaptureImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
        ds.is_implicit_VR                       = False
        ds.is_little_endian                     = True
        ds.PatientName                          = f"PATIENT_{row['patient_id']}"
        ds.PatientID                            = str(row["patient_id"])
        ds.PatientAge                           = f"{int(row['patient_age']):03d}Y" if pd.notnull(row["patient_age"]) else "000Y"
        ds.PatientSex                           = row["patient_gender"] if pd.notnull(row["patient_gender"]) else "O"
        ds.StudyDate                            = datetime.now().strftime("%Y%m%d")
        ds.StudyTime                            = datetime.now().strftime("%H%M%S")
        ds.StudyInstanceUID                     = generate_uid()
        ds.SeriesInstanceUID                    = generate_uid()
        ds.SOPInstanceUID                       = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID                          = pydicom.uid.SecondaryCaptureImageStorage
        ds.Modality                             = "CR"
        ds.ViewPosition                         = row["view_position"]
        ds.ImageComments                        = row["finding_labels"]
        ds.SamplesPerPixel                      = 1
        ds.PhotometricInterpretation            = "MONOCHROME2"
        ds.Rows                                 = img_arr.shape[0]
        ds.Columns                              = img_arr.shape[1]
        ds.BitsAllocated                        = 8
        ds.BitsStored                           = 8
        ds.HighBit                              = 7
        ds.PixelRepresentation                  = 0
        ds.PixelData                            = img_arr.tobytes()

        pydicom.dcmwrite(str(out), ds, write_like_original=False)
        log.info(f"  Saved : {out.name}  ({row['finding_labels']})")

    log.info(f"  DICOM files saved to : {DICOM_DIR}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Clean all previous outputs first
    clean_previous_outputs()

    # Set up fresh logging
    log = setup_logging()

    log.info("=" * 60)
    log.info("  NIH ChestX-ray14 - Phase 1 Preprocessing Pipeline")
    log.info("  Mode: 100-sample demonstration dataset")
    log.info("=" * 60)
    log.info(f"  Raw dir    : {RAW_DIR}")
    log.info(f"  Output dir : {PROCESSED_DIR}")
    log.info(f"  OpenCV     : {'available' if CV2_AVAILABLE else 'NOT installed - using PIL fallback'}")
    log.info("")

    df, bb        = load_csvs(log)
    df, cl_report = clean_metadata(df, log)
    image_index   = build_image_index(log)
    df            = match_images(df, image_index, log)
    df            = select_100_samples(df, log)
    df, skipped   = preprocess_images(df, log)
    df            = merge_bbox(df, bb, log)
    info          = save_outputs(df, cl_report, skipped, log)
    convert_to_dicom(df, log, n_samples=10)

    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE - 100 sample demonstration dataset")
    print("=" * 60)
    print(f"  Total images processed : {info['total_images']}")
    print(f"  Disease labels covered : {len(ALL_LABELS)}")
    print(f"  Skipped (corrupt)      : {len(skipped)}")
    print(f"\n  Files created:")
    print(f"    data/processed/images/           <- 100 processed PNGs")
    print(f"    data/processed/metadata_clean.csv")
    print(f"    data/processed/dataset_info.json")
    print(f"    data/processed/preprocessing_log.txt")
    print(f"    data/dicom_samples/              <- 10 sample .dcm files")
    print("=" * 60)
    print("\n  TIP: To process ALL 112,120 images later,")
    print("  remove the select_100_samples line in main.")
    print("=" * 60)