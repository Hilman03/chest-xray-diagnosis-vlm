"""
models/vlm_inference.py
=======================
VLM Inference using BiomedCLIP (zero-shot vision-language model).

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

BiomedCLIP is trained on PMC-15M (15M biomedical image-text pairs) — far
larger and broader than PubMedCLIP — so it is a stronger zero-shot chest-X-ray
matcher while remaining a true vision-language model (image encoder + text
encoder). It scores the CXR image against disease text descriptions and ranks
them by similarity.

Flow:
    CXR Image + Disease text descriptions (prompt-ensembled)
        -> BiomedCLIP scores similarity for each disease
        -> Returns ranked diseases with confidence scores
        -> Passed to LLaMA for structured report

Loaded via open_clip (not HuggingFace CLIPModel):
    pip install open_clip_torch
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time
import torch
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_NAME     = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MODEL_TAG      = f"hf-hub:{MODEL_NAME}"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
TOP_N_DISEASES = 3
CONTEXT_LEN    = 256

# Prompt-ensembling: several phrasings per disease are scored and averaged,
# which is more robust for zero-shot CLIP than a single template.
DISEASE_PROMPTS = {
    "Atelectasis"       : ["chest x-ray showing atelectasis",
                           "lung collapse on chest radiograph",
                           "atelectasis with volume loss"],
    "Cardiomegaly"      : ["chest x-ray showing cardiomegaly",
                           "enlarged cardiac silhouette",
                           "an enlarged heart on chest radiograph"],
    "Consolidation"     : ["chest x-ray showing consolidation",
                           "dense airspace consolidation in the lung",
                           "lobar consolidation on chest radiograph"],
    "Edema"             : ["chest x-ray showing pulmonary edema",
                           "bilateral pulmonary edema with haziness",
                           "fluid overload pulmonary edema"],
    "Effusion"          : ["chest x-ray showing pleural effusion",
                           "blunting of the costophrenic angle from effusion",
                           "pleural fluid effusion on chest radiograph"],
    "Emphysema"         : ["chest x-ray showing emphysema",
                           "hyperinflated lungs with emphysema",
                           "emphysematous lungs on chest radiograph"],
    "Fibrosis"          : ["chest x-ray showing pulmonary fibrosis",
                           "reticular fibrotic opacities in the lung",
                           "interstitial pulmonary fibrosis"],
    "Hernia"            : ["chest x-ray showing a hiatal hernia",
                           "diaphragmatic hernia on chest radiograph",
                           "bowel above the diaphragm hernia"],
    "Infiltration"      : ["chest x-ray showing lung infiltration",
                           "patchy pulmonary infiltrate",
                           "ill-defined infiltration in the lung"],
    "Mass"              : ["chest x-ray showing a lung mass",
                           "a large pulmonary mass with irregular border",
                           "soft tissue mass in the lung"],
    "Nodule"            : ["chest x-ray showing a pulmonary nodule",
                           "a small solitary lung nodule",
                           "rounded nodular opacity in the lung"],
    "Pleural_Thickening": ["chest x-ray showing pleural thickening",
                           "thickened pleura along the chest wall",
                           "pleural thickening on chest radiograph"],
    "Pneumonia"         : ["chest x-ray showing pneumonia",
                           "lobar pneumonia with consolidation",
                           "infectious pneumonia in the lung"],
    "Pneumothorax"      : ["chest x-ray showing pneumothorax",
                           "collapsed lung with a visible pleural line",
                           "air in the pleural space pneumothorax"],
    "No Finding"        : ["a normal chest x-ray with clear lungs",
                           "no acute cardiopulmonary abnormality",
                           "healthy chest radiograph with no findings"],
}

# One human-readable description per disease for the report/LLM/Grad-CAM.
DISEASE_TEMPLATES = {d: p[0] for d, p in DISEASE_PROMPTS.items()}

# Singletons
_model      = None
_preprocess = None
_tokenizer  = None
_text_feat  = None                     # cached, normalised, ensembled text features
_disease_names = list(DISEASE_PROMPTS.keys())


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    global _model, _preprocess, _tokenizer, _text_feat

    if _model is not None:
        return True

    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        print(f"  [BiomedCLIP] Loading : {MODEL_NAME}")
        print(f"  [BiomedCLIP] Device  : {DEVICE}")
        print(f"  [BiomedCLIP] Trained on PMC-15M biomedical image-text pairs")

        _model, _preprocess = create_model_from_pretrained(MODEL_TAG)
        _tokenizer = get_tokenizer(MODEL_TAG)
        _model.to(DEVICE).eval()

        # Pre-compute averaged text features per disease (prompt ensemble).
        feats = []
        with torch.no_grad():
            for name in _disease_names:
                toks = _tokenizer(DISEASE_PROMPTS[name],
                                  context_length=CONTEXT_LEN).to(DEVICE)
                tf = _model.encode_text(toks)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                feats.append(tf.mean(dim=0, keepdim=True))
        _text_feat = torch.cat(feats, dim=0)
        _text_feat = _text_feat / _text_feat.norm(dim=-1, keepdim=True)

        print(f"  [BiomedCLIP] Loaded successfully ({len(_disease_names)} classes)")
        return True

    except Exception as e:
        print(f"  [BiomedCLIP] Failed to load: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# DISEASE PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_diseases(image_path: str) -> dict:
    """
    Predict diseases by matching the CXR image against disease descriptions.

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
            "error"        : "Model failed to load (see backend logs)",
            "success"      : False,
        }

    try:
        image = _preprocess(
            Image.open(str(image_path)).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_f = _model.encode_image(image)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            logit_scale = _model.logit_scale.exp()
            probs = (logit_scale * img_f @ _text_feat.t()).softmax(dim=-1).squeeze(0)

        all_scores = {
            disease: round(float(prob), 4)
            for disease, prob in zip(_disease_names, probs)
        }
        sorted_diseases = sorted(
            all_scores.items(), key=lambda x: x[1], reverse=True)

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
        import traceback
        traceback.print_exc()
        detail = f"{type(e).__name__}: {e}".strip().rstrip(":").strip()
        print(f"  [BiomedCLIP] Prediction error: {detail}")
        return {
            "top_diseases" : [("Unknown", 0.0)],
            "all_scores"   : {},
            "primary_label": "Unknown",
            "description"  : "Prediction failed",
            "error"        : detail or "Prediction failed",
            "success"      : False,
        }


# ─────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def infer_vlm_with_label(image_path: str) -> dict:
    """
    Full BiomedCLIP inference on a CXR image.

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

    print(f"  [BiomedCLIP] Predicting diseases...")
    result = predict_diseases(image_path)

    print(f"  [BiomedCLIP] Primary disease : {result['primary_label']}")
    print(f"  [BiomedCLIP] Top predictions :")
    for disease, score in result["top_diseases"]:
        bar = "#" * int(score * 40)
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
        "error"         : result.get("error", ""),
    }


# ─────────────────────────────────────────────────────────────
# GRAD-CAM EXPLAINABILITY (heatmap of where BiomedCLIP "looked")
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
    Grad-CAM heatmap for BiomedCLIP — highlights the image regions that most
    drove the similarity to the predicted disease's text description. This is a
    visualization of the EXISTING model (gradients of its own score); it adds no
    new model and does not change the architecture.

    Args:
        image         : a PIL.Image (RGB) of the chest X-ray
        disease_label : the predicted disease to explain

    Returns:
        PNG bytes of the X-ray with the heatmap overlaid.
    """
    import io
    import numpy as np

    if not load_model():
        raise RuntimeError("BiomedCLIP unavailable — cannot compute Grad-CAM.")

    image = image.convert("RGB")
    img_t = _preprocess(image).unsqueeze(0).to(DEVICE)

    # Ensembled, normalised text feature for the target disease.
    prompts = DISEASE_PROMPTS.get(
        disease_label, [f"chest x-ray showing {disease_label}"])
    with torch.no_grad():
        toks = _tokenizer(prompts, context_length=CONTEXT_LEN).to(DEVICE)
        tf = _model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        tf = tf.mean(dim=0, keepdim=True)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    # Hook the last vision-transformer block of the timm trunk.
    layer = _model.visual.trunk.blocks[-1]
    store = {}

    def _fwd_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        store["act"] = h

    handle = layer.register_forward_hook(_fwd_hook)
    try:
        _model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            img_f = _model.encode_image(img_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            score = (_model.logit_scale.exp() * img_f @ tf.t())[0, 0]
            score.backward()
    finally:
        handle.remove()

    act  = store["act"].detach().float()         # [1, tokens, dim]
    grad = store["act"].grad.detach().float()     # [1, tokens, dim]

    weights = grad.mean(dim=1, keepdim=True)      # [1, 1, dim]
    cam     = (weights * act).sum(dim=-1)[0]      # [tokens]
    cam     = cam[1:]                             # drop CLS token
    cam     = torch.relu(cam)

    side = int(cam.shape[0] ** 0.5)               # 196 -> 14 for patch16 @224
    cam  = cam[: side * side].reshape(side, side)
    cam  = cam.cpu().numpy().astype("float32")

    lo = np.percentile(cam, 60)
    hi = np.percentile(cam, 99)
    cam = np.clip((cam - lo) / (hi - lo + 1e-8), 0, 1)

    base = np.array(image.resize((224, 224))).astype("float32")
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(
        (224, 224), Image.BILINEAR)
    cam_up  = np.array(cam_img).astype("float32") / 255.0

    color   = _jet_colormap(cam_up).astype("float32")
    alpha   = (cam_up * 0.6)[..., None]
    overlay = (base * (1 - alpha) + color * alpha).astype("uint8")

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Run directly to test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  BiomedCLIP VLM Inference")
    print(f"  {MODEL_NAME}")
    print("=" * 60)

    images_dir = ROOT / "data" / "processed" / "images"
    images     = sorted(images_dir.glob("*.png"))

    if not images:
        print(f"  No images found in {images_dir}")
        print(f"  Run scripts/preprocess.py first.")
        sys.exit(1)

    for test_image in images[:3]:
        print(f"\n  Image : {test_image.name}")
        print("-" * 50)
        result = infer_vlm_with_label(str(test_image))
        print(f"\n  Primary Disease : {result['disease_label']}")
        print(f"  Description     : {result['caption']}")
        print(f"  Response time   : {result['response_time']}s")

    print("\n" + "=" * 60)
