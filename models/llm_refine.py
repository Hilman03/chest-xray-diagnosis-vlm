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
    "You are a board-certified radiologist drafting a factual chest X-ray "
    "observation report from an AI image-analysis result. You did NOT see the "
    "image yourself — your ONLY source is the PubMedCLIP findings below. "
    "Ground every statement in those findings and calibrate your wording to "
    "their confidence: high confidence reads as a clear finding, moderate as "
    "'suggested'/'possible', low as 'cannot be excluded'. "
    "Do NOT invent measurements, laterality, locations, patient history, "
    "comparisons with priors, or treatment advice. Use professional "
    "radiology register, be concise and neutral, and stay strictly on-topic."
)

# Used only on the multimodal path, where the model can actually see the image.
SYSTEM_MSG_VISION = (
    "You are a board-certified radiologist reading the chest X-ray image "
    "provided to you. Describe ONLY what is genuinely visible in this "
    "radiograph. A separate AI tool (PubMedCLIP) has supplied candidate "
    "findings with confidence scores — treat these as a second opinion: state "
    "explicitly where the image agrees or disagrees with them rather than "
    "repeating them uncritically. Calibrate wording to the visible evidence. "
    "Do NOT invent patient history, prior studies, exact measurements, or "
    "treatment advice, and do NOT claim to see findings that are not there. "
    "Use professional radiology register, be concise and neutral, and stay "
    "strictly on-topic to this chest X-ray."
)


def _confidence_tier(score: float) -> str:
    """Map a probability to calibrated radiology language."""
    if score >= 0.60:
        return "high"
    if score >= 0.30:
        return "moderate"
    return "low"


def _top_score(top_diseases: list) -> str:
    """Format the top finding's confidence, e.g. '87% confidence (high)'."""
    if top_diseases:
        score = top_diseases[0][1]
        return f"{score*100:.0f}% confidence ({_confidence_tier(score)})"
    return "unknown confidence"


def _build_prompt(caption: str, disease_label: str,
                  top_diseases: list = None, vision: bool = False) -> str:
    """User message: hand the model the exact facts and a fixed structure."""
    if top_diseases:
        findings_str = "\n".join(
            f"  - {d}: {s*100:.1f}% confidence ({_confidence_tier(s)})"
            for d, s in top_diseases[:3]
        )
    else:
        findings_str = f"  - {disease_label}"

    if vision:
        findings_clause = (
            f"2. Findings: describe what you actually see in this chest X-ray, "
            f"focusing on the region relevant to {disease_label}. State whether "
            f"the image supports the PubMedCLIP prediction of {disease_label} "
            f"({_top_score(top_diseases)}), and mention the other listed "
            f"findings only as lower-confidence considerations. Note relevant "
            f"normal/clear areas where appropriate.\n"
        )
    else:
        findings_clause = (
            f"2. Findings: describe the radiographic appearance associated with "
            f"{disease_label}, using wording calibrated to its confidence "
            f"({_top_score(top_diseases)}). Mention the other listed findings "
            f"only as lower-confidence considerations.\n"
        )

    return (
        f"PubMedCLIP image analysis findings:\n"
        f"{findings_str}\n"
        f"Primary finding: {disease_label}\n"
        f"Reference description: \"{caption}\"\n\n"
        f"Write a short observational report (3 short paragraphs, ~120 words "
        f"total, prose not bullet points) using EXACTLY this structure and "
        f"these three numbered headings:\n"
        f"1. Technique: state it is a frontal chest radiograph of adequate "
        f"diagnostic quality.\n"
        f"{findings_clause}"
        f"3. Impression: one sentence summarising {disease_label} as the "
        f"AI-predicted primary finding, noting this is an AI observation "
        f"requiring radiologist confirmation.\n\n"
        f"Rules:\n"
        f"- Output ONLY the report itself about this chest X-ray.\n"
        f"- Do NOT add any preamble, introduction, sign-off, or markdown "
        f"formatting (no \"Here is a report\", no \"based on the findings\", "
        f"no notes about the AI model, no asterisks or headers).\n"
        f"- Do NOT state any finding not supported by the data above."
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


def _strip_markdown(line: str) -> str:
    """Remove leaked markdown markers (bold/italic, headers, bullets)."""
    line = line.replace("**", "").replace("__", "")
    # Leading header (#) or bullet (*, -, •) markers, kept defensively because
    # the prompt forbids markdown but small models occasionally emit it.
    return line.lstrip("#*-• \t")


def _clean_report(text: str) -> str:
    """Strip preamble/closing filler so only X-ray observation lines remain."""
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        ln = _strip_markdown(ln)
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
    # Surface Ollama's actual error body (e.g. an OOM "model runner has
    # stopped" message) instead of a bare HTTP status, so failures are
    # diagnosable from the backend logs.
    if r.status_code >= 400:
        raise RuntimeError(
            f"Ollama /api/chat HTTP {r.status_code} for model '{model}': "
            f"{r.text[:400]}"
        )
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
    try:
        report = _generate_ollama(caption, disease_label, top_diseases, image_path)
    except Exception as e:
        # The vision model can fail on constrained GPUs (e.g. a T4 OOM returns
        # HTTP 500 from /api/chat). Rather than fail the whole analysis, degrade
        # gracefully to a text-only report grounded in the PubMedCLIP findings.
        if use_vision:
            print(f"  [LLM] Vision generation failed ({e}); "
                  f"falling back to text-only report...")
            report  = _generate_ollama(caption, disease_label,
                                       top_diseases, image_path=None)
            backend = "ollama-text-fallback"
        else:
            raise

    elapsed = round(time.time() - start, 3)
    print(f"  [LLM] Report generated in {elapsed}s (backend: {backend})")

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
