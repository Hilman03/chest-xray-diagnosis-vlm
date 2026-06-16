"""
models/llm_refine.py
====================
LLM Report Generation — Ollama only (single backend).

Backend : Ollama HTTP API  (http://localhost:11434)
          Vision model reads the X-ray image directly; text model is used
          only when no image is supplied or vision is disabled.

The model receives the chest X-ray image plus the PubMedCLIP findings
(confidence scores) as a second opinion. The prompt is constrained so the
report stays grounded and on-topic — no invented findings.

Run Ollama:
    ollama serve
    ollama pull llama3.2-vision     # vision model (reads the image)
    ollama pull llama3.2            # text model (no-image path)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time
import base64
import requests

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Multimodal path — a vision LLM that actually SEES the X-ray image.
# When enabled and an image is given, the report is grounded in the pixels
# plus the PubMedCLIP findings (second opinion), instead of the label alone.
ENABLE_VISION       = os.getenv("OLLAMA_USE_VISION", "1") not in ("0", "false", "False")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

MAX_NEW_TOKENS = 350

# Deterministic-leaning decoding — low temperature + nucleus + repeat penalty.
GEN_TEMPERATURE = 0.2
GEN_TOP_P       = 0.85
GEN_REPEAT_PEN  = 1.3


# ─────────────────────────────────────────────────────────────
# PROMPT — constrained, fact-grounded, no invention
# ─────────────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are a radiologist drafting a factual chest X-ray observation report "
    "from an AI image-analysis result. You did NOT see the image yourself — "
    "you only have the PubMedCLIP findings below. "
    "Report ONLY those findings and their confidence. "
    "Do NOT invent measurements, locations, patient history, comparisons, or "
    "treatment advice. Be concise, neutral, and stay strictly on-topic."
)

# Used only on the multimodal path, where the model can actually see the image.
SYSTEM_MSG_VISION = (
    "You are a radiologist reading a chest X-ray image that is provided to you. "
    "Describe ONLY what is visible in this chest radiograph. "
    "A separate AI tool (PubMedCLIP) has given candidate findings with "
    "confidence scores — use them as a second opinion, and note where the "
    "image agrees or disagrees with them. "
    "Do NOT invent patient history, prior studies, exact measurements, or "
    "treatment advice. Be concise, neutral, and stay strictly on-topic to "
    "this chest X-ray."
)


def _top_score(top_diseases: list) -> str:
    """Format the top finding's confidence, e.g. '87% confidence'."""
    if top_diseases:
        return f"{top_diseases[0][1]*100:.0f}% confidence"
    return "unknown confidence"


def _build_prompt(caption: str, disease_label: str,
                  top_diseases: list = None, vision: bool = False) -> str:
    """User message: hand the model the exact facts and a fixed structure."""
    if top_diseases:
        findings_str = "\n".join(
            f"  - {d}: {s*100:.1f}% confidence" for d, s in top_diseases[:3]
        )
    else:
        findings_str = f"  - {disease_label}"

    if vision:
        findings_clause = (
            f"2. Findings: describe what you actually see in this chest X-ray, "
            f"focusing on the region relevant to {disease_label}. State whether "
            f"the image supports the PubMedCLIP prediction of {disease_label} "
            f"({_top_score(top_diseases)}), and mention the other listed "
            f"findings only as lower-confidence considerations.\n"
        )
    else:
        findings_clause = (
            f"2. Findings: describe the radiographic appearance associated with "
            f"{disease_label}, and note the supporting confidence score. "
            f"Mention the other listed findings only as lower-confidence "
            f"considerations.\n"
        )

    return (
        f"PubMedCLIP image analysis findings:\n"
        f"{findings_str}\n"
        f"Primary finding: {disease_label}\n"
        f"Reference description: \"{caption}\"\n\n"
        f"Write a short observational report using EXACTLY this structure:\n"
        f"1. Technique: state it is a frontal chest radiograph of adequate "
        f"diagnostic quality.\n"
        f"{findings_clause}"
        f"3. Impression: one sentence summarising {disease_label} as the "
        f"AI-predicted primary finding, noting this is an AI observation "
        f"requiring radiologist confirmation.\n\n"
        f"Output ONLY the report itself about this chest X-ray. "
        f"Do NOT add any preamble, introduction, sign-off, or sentences "
        f"that are not about the radiograph (no \"Here is a report\", "
        f"no \"based on the findings\", no notes about the AI model)."
    )


# ─────────────────────────────────────────────────────────────
# POST-PROCESS — keep only X-ray-relevant report text
# ─────────────────────────────────────────────────────────────
# Lines that are conversational filler / meta-talk, not radiograph content.
_FILLER_PREFIXES = (
    "here is", "here's", "here are", "sure", "certainly", "of course",
    "based on", "the following", "below is", "i have", "i've", "as requested",
    "this report", "in summary", "note:", "disclaimer", "please note",
)


def _clean_report(text: str) -> str:
    """Strip preamble/closing filler so only X-ray observation lines remain."""
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if any(low.startswith(p) for p in _FILLER_PREFIXES):
            continue
        kept.append(ln)
    cleaned = "\n".join(kept).strip()
    return cleaned if len(cleaned) > 20 else text.strip()


# ─────────────────────────────────────────────────────────────
# OLLAMA BACKEND
# ─────────────────────────────────────────────────────────────
def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False


def _encode_image(image_path: str) -> str:
    """Base64-encode an image for Ollama's multimodal `images` field."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _generate_ollama(caption: str, disease_label: str,
                     top_diseases: list = None,
                     image_path: str = None) -> str:
    """
    Generate via Ollama. If image_path is given and vision is enabled, the
    image is sent to a vision model so the report is grounded in the pixels.
    """
    use_vision = bool(image_path) and ENABLE_VISION
    model      = OLLAMA_VISION_MODEL if use_vision else OLLAMA_MODEL
    system_msg = SYSTEM_MSG_VISION if use_vision else SYSTEM_MSG
    prompt     = _build_prompt(caption, disease_label, top_diseases, vision=use_vision)

    user_msg = {"role": "user", "content": prompt}
    if use_vision:
        user_msg["images"] = [_encode_image(image_path)]

    payload = {
        "model"   : model,
        "stream"  : False,
        "messages": [
            {"role": "system", "content": system_msg},
            user_msg,
        ],
        "options": {
            "temperature"   : GEN_TEMPERATURE,
            "top_p"         : GEN_TOP_P,
            "repeat_penalty": GEN_REPEAT_PEN,
            "num_predict"   : MAX_NEW_TOKENS,
        },
    }

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    text = r.json().get("message", {}).get("content", "").strip()
    text = _clean_report(text)

    if text and len(text) > 20:
        return text
    raise ValueError(f"Ollama returned unusable output: '{text}'")


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def refine_llm(caption: str, disease_label: str,
               top_diseases: list = None,
               image_path: str = None) -> dict:
    """
    Generate a structured observational report using Ollama.

    If image_path is given and vision is enabled, a vision LLM reads the
    actual X-ray (grounded report). Otherwise a text LLM works from the
    PubMedCLIP findings.

    Returns:
        {
            "report"        : str,
            "backend"       : "ollama-vision" | "ollama",
            "response_time" : float,
        }
    """
    start = time.time()

    if not _ollama_available():
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_URL}. "
            "Start it with `ollama serve` and pull the model "
            f"(`ollama pull {OLLAMA_VISION_MODEL}`)."
        )

    use_vision = bool(image_path) and ENABLE_VISION
    model_name = OLLAMA_VISION_MODEL if use_vision else OLLAMA_MODEL
    backend    = "ollama-vision" if use_vision else "ollama"

    print(f"  [LLM] Generating report with Ollama ({model_name})"
          f"{' + image' if use_vision else ''}...")
    report  = _generate_ollama(caption, disease_label, top_diseases, image_path)
    elapsed = round(time.time() - start, 3)
    print(f"  [LLM] Report generated in {elapsed}s")

    return {
        "report"        : report,
        "backend"       : backend,
        "response_time" : elapsed,
    }


# ─────────────────────────────────────────────────────────────
# Run directly to test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Report Generation — Ollama only")
    print(f"  Ollama URL    : {OLLAMA_URL}")
    print(f"  Text model    : {OLLAMA_MODEL}")
    print(f"  Vision model  : {OLLAMA_VISION_MODEL}")
    print(f"  Vision on?    : {ENABLE_VISION}")
    print(f"  Available     : {_ollama_available()}")
    print("=" * 60)

    test_cases = [
        ("Chest X-Ray showing pneumonia with lobar opacity", "Pneumonia",
         [("Pneumonia", 0.87), ("Consolidation", 0.08)]),
        ("Chest X-Ray showing pneumothorax with pleural line", "Pneumothorax",
         [("Pneumothorax", 0.92), ("Effusion", 0.05)]),
    ]

    for caption, label, top in test_cases:
        print(f"\n  Disease : {label}\n")
        result = refine_llm(caption, label, top)
        print(f"  Backend : {result['backend']}")
        print(f"  Time    : {result['response_time']}s\n")
        print(f"  Report:\n  {result['report']}")
        print("-" * 60)
