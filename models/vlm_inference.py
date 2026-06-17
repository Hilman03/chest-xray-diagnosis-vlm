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
            disease: round(float(prob), 4)
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


# ─────────────────────────────────────────────────────────────
# GRAD-CAM EXPLAINABILITY (heatmap of where PubMedCLIP "looked")
# ─────────────────────────────────────────────────────────────
def _jet_colormap(cam):
    """Map a HxW array in [0,1] to a jet-like RGB uint8 image (numpy only)."""
    import numpy as np
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype("uint8")


def compute_gradcam(image, disease_label: str) -> bytes:
    """
    Grad-CAM heatmap for PubMedCLIP — highlights the image regions that most
    drove the similarity to the predicted disease's text description. This is
    a visualization of the EXISTING model (gradients of its own score); it adds
    no new model and does not change the architecture.

    Args:
        image         : a PIL.Image (RGB) of the chest X-ray
        disease_label : the predicted disease to explain

    Returns:
        PNG bytes of the X-ray with the heatmap overlaid.
    """
    import io
    import numpy as np

    if not load_model():
        raise RuntimeError("PubMedCLIP unavailable — cannot compute Grad-CAM.")

    image = image.convert("RGB")
    text  = DISEASE_TEMPLATES.get(
        disease_label, f"Chest X-Ray showing {disease_label}")

    inputs = _processor(text=[text], images=image,
                        return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Capture activations + gradients of the last vision transformer block.
    layer = _model.vision_model.encoder.layers[-1]
    store = {}

    def _fwd_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        store["act"] = h

    handle = layer.register_forward_hook(_fwd_hook)
    try:
        _model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out   = _model(**inputs)
            score = out.logits_per_image[0, 0]   # image↔disease-text similarity
            score.backward()
    finally:
        handle.remove()

    act  = store["act"].detach().float()        # [1, tokens, dim]
    grad = act.grad if act.grad is not None else store["act"].grad
    grad = grad.detach().float()                 # [1, tokens, dim]

    # Canonical Grad-CAM: channel weights = global-average-pooled gradients,
    # CAM = ReLU(sum_c w_c * activation_c) per token.
    weights = grad.mean(dim=1, keepdim=True)     # [1, 1, dim]
    cam     = (weights * act).sum(dim=-1)[0]     # [tokens]
    cam     = cam[1:]                            # drop CLS token
    cam     = torch.relu(cam)

    side = int(cam.shape[0] ** 0.5)              # 49 -> 7 for patch32 @224
    cam  = cam[: side * side].reshape(side, side)
    cam  = cam.cpu().numpy().astype("float32")

    # Suppress the broad baseline so only the strongest regions light up
    # (map the median->99th-percentile range into 0..1). Keeps the heatmap
    # focused instead of flooding the whole image.
    lo = np.percentile(cam, 60)
    hi = np.percentile(cam, 99)
    cam = np.clip((cam - lo) / (hi - lo + 1e-8), 0, 1)

    # Upsample the small CAM grid to image size with PIL (no cv2/scipy needed).
    base = np.array(image.resize((224, 224))).astype("float32")
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(
        (224, 224), Image.BILINEAR)
    cam_up  = np.array(cam_img).astype("float32") / 255.0

    color   = _jet_colormap(cam_up).astype("float32")
    alpha   = (cam_up * 0.6)[..., None]          # hot areas colored, anatomy still visible
    overlay = (base * (1 - alpha) + color * alpha).astype("uint8")

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()


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