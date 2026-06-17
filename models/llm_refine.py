"""
models/llm_refine.py
====================
LLM Report Generation — in-process Hugging Face transformers (single backend).

No external server (no Ollama). The model is a small instruct LLM that runs
directly in this Python process and is loaded ONCE as a singleton, so there is
nothing to install or keep alive separately and nothing that can crash with a
500 / OOM the way an external vision server does on a small GPU.

The model receives the PubMedCLIP findings (disease + confidence scores) and
writes a structured observational report. The prompt is constrained so the
report stays grounded and on-topic — no invented findings.

Model is configurable via the LLM_MODEL env var, e.g.:
    TinyLlama/TinyLlama-1.1B-Chat-v1.0   (default — small, reliable)
    Qwen/Qwen2.5-1.5B-Instruct           (better quality, still T4-friendly)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
# Default to Qwen2.5-1.5B-Instruct: still small enough for a T4 but follows the
# report structure far better than TinyLlama (which hallucinated form fields).
# Override with the LLM_MODEL env var, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 350

# Greedy decoding (no sampling) for reliability and determinism. Sampling in
# fp16 on GPU can yield inf/nan probabilities and crash generate() with an
# unhelpful/blank error; greedy avoids that entirely. A repetition penalty +
# no-repeat-ngram window keep a small model from looping.
GEN_REPEAT_PEN      = 1.3
GEN_NO_REPEAT_NGRAM = 3

# Singleton — loaded once, kept in memory for the whole process lifetime.
_model     = None
_tokenizer = None


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


# Clinical knowledge for each NIH ChestX-ray14 class. This is the "overall
# information" handed from the VLM findings to the LLM so the report can
# explain the finding in detail (definition + radiographic appearance +
# clinical significance) instead of inventing content.
DISEASE_KNOWLEDGE = {
    "Atelectasis": "collapse or incomplete expansion of lung tissue; appears as "
        "increased opacity with volume loss, displacement of fissures and "
        "elevation of the hemidiaphragm; often from airway obstruction or "
        "hypoventilation.",
    "Cardiomegaly": "enlargement of the cardiac silhouette (cardiothoracic ratio "
        ">0.5 on a PA film); suggests underlying cardiac disease such as heart "
        "failure, valvular disease or cardiomyopathy.",
    "Consolidation": "alveolar air replaced by fluid, pus or cells, producing a "
        "dense homogeneous opacity, often with air bronchograms; typical of "
        "pneumonia, aspiration or pulmonary haemorrhage.",
    "Edema": "fluid accumulation in the pulmonary interstitium and alveoli; shows "
        "bilateral perihilar haziness, Kerley B lines and vascular redistribution; "
        "commonly cardiogenic (left heart failure) or from fluid overload.",
    "Effusion": "fluid in the pleural space; appears as blunting of the "
        "costophrenic angle and a meniscus, with larger collections opacifying "
        "the hemithorax; causes include heart failure, infection and malignancy.",
    "Emphysema": "permanent enlargement and destruction of distal air spaces; "
        "hyperinflated lucent lungs, flattened diaphragms and a narrow cardiac "
        "silhouette; strongly associated with smoking and COPD.",
    "Fibrosis": "scarring and thickening of lung interstitium; reticular opacities, "
        "volume loss and architectural distortion, often basal and peripheral; "
        "seen in interstitial lung disease.",
    "Hernia": "protrusion of abdominal contents (commonly stomach) through the "
        "diaphragm into the thorax; may show a retrocardiac air-fluid level.",
    "Infiltration": "ill-defined patchy or hazy opacity from cells or fluid in the "
        "lung; a nonspecific sign that can reflect infection, inflammation or "
        "oedema.",
    "Mass": "a focal opacity larger than 3 cm with defined margins; raises concern "
        "for neoplasm and warrants further characterisation.",
    "Nodule": "a rounded focal opacity 3 cm or smaller; may be benign (granuloma) "
        "or malignant and usually needs follow-up or comparison with priors.",
    "Pleural_Thickening": "fibrotic thickening of the pleural surface, sometimes "
        "with calcification; from prior infection, asbestos exposure or "
        "haemorrhage.",
    "Pneumonia": "infection of lung parenchyma producing consolidation, air "
        "bronchograms and patchy or lobar opacity; correlate with fever, cough "
        "and raised inflammatory markers.",
    "Pneumothorax": "air in the pleural space causing lung collapse; a visible "
        "visceral pleural line with absent lung markings peripherally; a tension "
        "pneumothorax is a medical emergency.",
    "No Finding": "no significant radiographic abnormality identified; clear lung "
        "fields, normal cardiomediastinal contour and no effusion.",
}


def _disease_info(name: str) -> str:
    """Clinical description for a disease label (empty if unknown)."""
    return DISEASE_KNOWLEDGE.get(name, "")


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
                  top_diseases: list = None) -> str:
    """User message: hand the model the exact facts and a fixed structure."""
    if top_diseases:
        # Pass the full set of findings WITH their clinical descriptions, so the
        # LLM has real medical content to explain rather than inventing it.
        findings_str = "\n".join(
            f"  - {d}: {s*100:.1f}% confidence ({_confidence_tier(s)}) — "
            f"{_disease_info(d)}"
            for d, s in top_diseases[:3]
        )
    else:
        findings_str = f"  - {disease_label} — {_disease_info(disease_label)}"

    primary_info = _disease_info(disease_label) or "the predicted condition"

    return (
        f"PubMedCLIP image analysis findings (with clinical background):\n"
        f"{findings_str}\n"
        f"Primary finding: {disease_label} ({_top_score(top_diseases)})\n"
        f"Reference description: \"{caption}\"\n\n"
        f"Clinical background for the primary finding ({disease_label}):\n"
        f"  {primary_info}\n\n"
        f"Write a detailed observational report (3 paragraphs) using EXACTLY "
        f"this structure and these three numbered headings:\n"
        f"1. Technique: state it is a frontal chest radiograph of adequate "
        f"diagnostic quality.\n"
        f"2. Findings: explain {disease_label} in detail — what the condition "
        f"is, its typical radiographic appearance on a chest X-ray, and its "
        f"clinical significance — using the clinical background above and "
        f"wording calibrated to the {_confidence_tier(top_diseases[0][1]) if top_diseases else 'stated'} "
        f"confidence. Then briefly note the other listed findings as "
        f"lower-confidence alternative considerations.\n"
        f"3. Impression: two or three sentences summarising {disease_label} as "
        f"the AI-predicted primary finding, its likely significance, and a note "
        f"that this is an AI observation requiring radiologist confirmation.\n\n"
        f"Rules:\n"
        f"- This is a chest X-ray report ONLY. Do NOT output any form, header "
        f"block, patient demographics, names, dates, signatures, certification "
        f"or contact fields.\n"
        f"- Write in clinical prose. No markdown, asterisks, bullet symbols, or "
        f"preamble like \"Here is a report\".\n"
        f"- Do NOT invent measurements, laterality or findings not supported "
        f"by the data above."
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

# Hallucinated form / demographic / header content that small models sometimes
# emit (e.g. a fake certification form). Any line containing one of these is
# dropped — none belong in a radiograph observation report.
_FORM_JUNK = (
    "patient name", "name:", "date of birth", "dob:", "gender:", "sex:",
    "height:", "weight:", "eye color", "hair color", "skin tone",
    "marital status", "race:", "ethnicity:", "occupation", "industry:",
    "zip code", "city:", "address:", "phone", "email", "signature",
    "board certification", "certification:", "date and location",
    "date:", "[insert", "_____",
)


def _is_form_junk(line: str) -> bool:
    low = line.lower()
    return any(tok in low for tok in _FORM_JUNK)


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
        if _is_form_junk(ln):          # drop hallucinated form/demographic lines
            continue
        kept.append(ln)
    cleaned = "\n".join(kept).strip()
    return cleaned if len(cleaned) > 20 else text.strip()


# ─────────────────────────────────────────────────────────────
# MODEL — load once, keep in memory
# ─────────────────────────────────────────────────────────────
def load_llm() -> bool:
    """Load the LLM + tokenizer once. Safe to call repeatedly (no-op after)."""
    global _model, _tokenizer
    if _model is not None:
        return True
    try:
        print(f"  [LLM] Loading : {LLM_MODEL}")
        print(f"  [LLM] Device  : {DEVICE}")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        _model.to(DEVICE)
        _model.eval()
        print(f"  [LLM] Loaded successfully")
        return True
    except Exception as e:
        print(f"  [LLM] Failed to load: {e}")
        return False


def _generate(caption: str, disease_label: str,
              top_diseases: list = None) -> str:
    """Run the instruct model on the structured prompt and clean the output."""
    prompt   = _build_prompt(caption, disease_label, top_diseases)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": prompt},
    ]

    # Use the model's chat template when available; fall back to a plain
    # concatenation for tokenizers that don't define one. apply_chat_template
    # may return a tensor, a dict, or a BatchEncoding depending on the
    # transformers version — normalise all three to a plain input_ids tensor.
    try:
        enc = _tokenizer.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        )
    except Exception:
        flat = f"{SYSTEM_MSG}\n\n{prompt}\n\nReport:\n"
        enc = _tokenizer(flat, return_tensors="pt")

    if hasattr(enc, "input_ids"):          # BatchEncoding
        input_ids = enc.input_ids
    elif isinstance(enc, dict):            # plain dict
        input_ids = enc["input_ids"]
    else:                                  # already a tensor
        input_ids = enc
    input_ids      = input_ids.to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    pad_id         = _tokenizer.pad_token_id or _tokenizer.eos_token_id

    try:
        with torch.no_grad():
            output = _model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                       # greedy — no fp16 nan crash
                repetition_penalty=GEN_REPEAT_PEN,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
                pad_token_id=pad_id,
            )
    except Exception as e:
        # Never let a blank-message exception bubble up unlabelled.
        raise RuntimeError(
            f"LLM generate() failed [{type(e).__name__}]: {e}"
        ) from e

    # Decode only the newly generated tokens (skip the prompt).
    gen  = output[0][input_ids.shape[-1]:]
    text = _tokenizer.decode(gen, skip_special_tokens=True).strip()
    text = _clean_report(text)

    if text and len(text) > 20:
        return text
    raise ValueError(f"LLM returned unusable output (len={len(text)}): {text[:80]!r}")


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def refine_llm(caption: str, disease_label: str,
               top_diseases: list = None,
               image_path: str = None) -> dict:
    """
    Generate a structured observational report from the PubMedCLIP findings
    using the in-process transformers LLM.

    `image_path` is accepted for backward compatibility with the pipeline but
    is not used — the LLM is text-only and is grounded in the VLM findings.

    Returns:
        {
            "report"        : str,
            "backend"       : "transformers:<model>",
            "response_time" : float,
        }
    """
    start = time.time()

    if not load_llm():
        raise RuntimeError(
            f"LLM model '{LLM_MODEL}' failed to load. Check that transformers "
            "and torch are installed and the model name is correct."
        )

    backend = f"transformers:{LLM_MODEL.split('/')[-1]}"
    print(f"  [LLM] Generating report with {backend}...")
    report  = _generate(caption, disease_label, top_diseases)
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
    print("  LLM Report Generation — in-process transformers")
    print(f"  Model   : {LLM_MODEL}")
    print(f"  Device  : {DEVICE}")
    print(f"  Loaded  : {load_llm()}")
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
