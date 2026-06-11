"""
models/vlm_inference.py
=======================
VLM Inference using PubMedCLIP
Model: flaviagiammarino/pubmed-clip-vit-base-patch32

PubMedCLIP is trained on PubMed medical images and understands
medical terminology. It works by matching the CXR image against
disease text descriptions and scoring similarity.

Flow:
    CXR Image + Disease text descriptions
        -> PubMedCLIP scores similarity for each disease
        -> Returns ranked diseases with confidence scores
        -> Passed to LLaMA for structured report

Install:
    pip install torch torchvision transformers pillow requests
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_NAME     = "flaviagiammarino/pubmed-clip-vit-base-patch32"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
TOP_N_DISEASES = 3

# Disease text descriptions for PubMedCLIP matching
# Phrased as medical observations — matches how PubMed papers describe findings
DISEASE_TEMPLATES = {
    "Atelectasis"       : "Chest X-Ray showing atelectasis with partial lung collapse",
    "Cardiomegaly"      : "Chest X-Ray showing cardiomegaly with enlarged cardiac silhouette",
    "Consolidation"     : "Chest X-Ray showing consolidation with dense homogeneous opacity",
    "Edema"             : "Chest X-Ray showing pulmonary edema with bilateral haziness",
    "Effusion"          : "Chest X-Ray showing pleural effusion with blunting of costophrenic angle",
    "Emphysema"         : "Chest X-Ray showing emphysema with hyperinflated lungs",
    "Fibrosis"          : "Chest X-Ray showing pulmonary fibrosis with reticular opacity",
    "Hernia"            : "Chest X-Ray showing hiatal hernia with bowel above diaphragm",
    "Infiltration"      : "Chest X-Ray showing lung infiltration with patchy haziness",
    "Mass"              : "Chest X-Ray showing a large lung mass with irregular border",
    "Nodule"            : "Chest X-Ray showing a small pulmonary nodule",
    "Pleural_Thickening": "Chest X-Ray showing pleural thickening along lateral chest wall",
    "Pneumonia"         : "Chest X-Ray showing pneumonia with lobar opacity and consolidation",
    "Pneumothorax"      : "Chest X-Ray showing pneumothorax with visible pleural line",
    "No Finding"        : "Normal Chest X-Ray with clear lung fields and no abnormalities",
}

# Singleton
_model     = None
_processor = None


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    global _model, _processor

    if _model is not None:
        return True

    try:
        print(f"  [PubMedCLIP] Loading : {MODEL_NAME}")
        print(f"  [PubMedCLIP] Device  : {DEVICE}")
        print(f"  [PubMedCLIP] Trained on PubMed medical images")
        print(f"  [PubMedCLIP] Downloading — please wait...")

        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model     = CLIPModel.from_pretrained(MODEL_NAME)
        _model.to(DEVICE)
        _model.eval()

        print(f"  [PubMedCLIP] Loaded successfully")
        return True

    except Exception as e:
        print(f"  [PubMedCLIP] Failed to load: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# DISEASE PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_diseases(image_path: str) -> dict:
    """
    Predict diseases by matching CXR image against disease descriptions.

    Uses same approach as the official PubMedCLIP usage example:
        processor(text=texts, images=image, return_tensors="pt")
        model(**inputs).logits_per_image.softmax(dim=1)

    Args:
        image_path : path to CXR image

    Returns:
        {
            "top_diseases"  : list of (disease, score) tuples
            "all_scores"    : dict of all disease scores
            "primary_label" : str top predicted disease
            "description"   : str text description of top disease
            "success"       : bool
        }
    """
    if not load_model():
        return {
            "top_diseases" : [("Unknown", 0.0)],
            "all_scores"   : {},
            "primary_label": "Unknown",
            "description"  : "Model unavailable",
            "success"      : False,
        }

    try:
        # Load CXR image
        image = Image.open(str(image_path)).convert("RGB")

        # Get disease names and descriptions
        disease_names = list(DISEASE_TEMPLATES.keys())
        disease_texts = list(DISEASE_TEMPLATES.values())

        # Process image + all disease texts (official usage pattern)
        inputs = _processor(
            text=disease_texts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get similarity scores
        with torch.no_grad():
            outputs = _model(**inputs)
            probs   = outputs.logits_per_image.softmax(dim=1).squeeze()

        # Map disease names to scores
        all_scores = {
            disease: float(round(float(prob), 4))
            for disease, prob in zip(disease_names, probs)
        }

        # Sort by score descending
        sorted_diseases = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_diseases  = sorted_diseases[:TOP_N_DISEASES]
        primary_label = top_diseases[0][0]
        description   = DISEASE_TEMPLATES[primary_label]

        return {
            "top_diseases" : top_diseases,
            "all_scores"   : all_scores,
            "primary_label": primary_label,
            "description"  : description,
            "success"      : True,
        }

    except Exception as e:
        print(f"  [PubMedCLIP] Prediction error: {e}")
        return {
            "top_diseases" : [("Unknown", 0.0)],
            "all_scores"   : {},
            "primary_label": "Unknown",
            "description"  : "Prediction failed",
            "success"      : False,
        }


# ─────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def infer_vlm(image_path: str) -> str:
    """Returns disease description string."""
    result = infer_vlm_with_label(image_path)
    return result["caption"]


def infer_vlm_with_label(image_path: str) -> dict:
    """
    Full PubMedCLIP inference on a CXR image.

    Args:
        image_path : full path to CXR image (PNG)

    Returns:
        {
            "caption"       : str   description of predicted disease
            "disease_label" : str   primary predicted disease
            "top_diseases"  : list  top N diseases with scores
            "all_scores"    : dict  all disease scores
            "image_name"    : str
            "response_time" : float
            "success"       : bool
        }
    """
    start      = time.time()
    image_name = Path(image_path).name

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"  [PubMedCLIP] Predicting diseases...")
    result = predict_diseases(image_path)

    print(f"  [PubMedCLIP] Primary disease : {result['primary_label']}")
    print(f"  [PubMedCLIP] Top predictions :")
    for disease, score in result["top_diseases"]:
        bar = "█" * int(score * 40)
        print(f"               {disease:<25} : {bar:<40} {score*100:.1f}%")

    elapsed = round(time.time() - start, 3)

    return {
        "caption"       : result["description"],
        "disease_label" : result["primary_label"],
        "top_diseases"  : result["top_diseases"],
        "all_scores"    : result["all_scores"],
        "image_name"    : image_name,
        "response_time" : elapsed,
        "success"       : result["success"],
    }


def infer_vlm_timed(image_path: str) -> dict:
    return infer_vlm_with_label(image_path)


# ─────────────────────────────────────────────────────────────
# Run directly to test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PubMedCLIP VLM Inference")
    print(f"  {MODEL_NAME}")
    print("=" * 60)

    images_dir = ROOT / "data" / "processed" / "images"
    images     = sorted(images_dir.glob("*.png"))

    if not images:
        print(f"  No images found in {images_dir}")
        print(f"  Run scripts/preprocess.py first.")
        sys.exit(1)

    # Test on first 3 images to see variety
    test_images = images[:3]

    for test_image in test_images:
        print(f"\n  Image : {test_image.name}")
        print("-" * 50)

        result = infer_vlm_with_label(str(test_image))

        print(f"\n  Primary Disease : {result['disease_label']}")
        print(f"  Description     : {result['caption']}")
        print(f"  Top Predictions :")
        for disease, score in result["top_diseases"]:
            bar = "█" * int(score * 40)
            print(f"    {disease:<25} : {bar:<40} {score*100:.1f}%")
        print(f"  Response time   : {result['response_time']}s")

    print("\n" + "=" * 60)