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
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 350

# Deterministic-leaning decoding — low temperature + nucleus + repeat penalty.
GEN_TEMPERATURE = 0.2
GEN_TOP_P       = 0.85
GEN_REPEAT_PEN  = 1.3

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
        findings_str = "\n".join(
            f"  - {d}: {s*100:.1f}% confidence ({_confidence_tier(s)})"
            for d, s in top_diseases[:3]
        )
    else:
        findings_str = f"  - {disease_label}"

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
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REPEAT_PEN,
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
