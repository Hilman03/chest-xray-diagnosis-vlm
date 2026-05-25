"""
models/llm_refine.py
====================
LLM Report Generation using TinyLlama 1.1B Chat

Model : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Link  : https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
Size  : ~600MB
RAM   : ~1GB on CPU

Uses AutoModelForCausalLM + AutoTokenizer + GenerationConfig
as per official HuggingFace transformers documentation.

Backends (tried in order):
    1. TinyLlama  — true LLM, proper fluent sentences
    2. Template   — always works, no model needed

Install:
    pip install torch transformers accelerate
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
LLM_MODEL      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200

# Singletons
_model     = None
_tokenizer = None


# ─────────────────────────────────────────────────────────────
# BACKEND 1 — TinyLlama
# ─────────────────────────────────────────────────────────────
def load_tinyllama() -> bool:
    """Load TinyLlama model and tokenizer. Returns True if successful."""
    global _model, _tokenizer

    if _model is not None:
        return True

    try:
        print(f"  [LLM] Loading TinyLlama 1.1B Chat")
        print(f"  [LLM] Model  : {LLM_MODEL}")
        print(f"  [LLM] Link   : https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print(f"  [LLM] Device : {DEVICE}")
        print(f"  [LLM] First run downloads ~600MB — please wait...")

        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model     = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float32,   # float32 for CPU stability
            low_cpu_mem_usage=True,
        )
        _model.to(DEVICE)
        _model.eval()

        print(f"  [LLM] TinyLlama loaded successfully")
        return True

    except Exception as e:
        print(f"  [LLM] Failed to load TinyLlama: {e}")
        return False


def _build_prompt(caption: str, disease_label: str,
                  top_diseases: list = None) -> str:
    """
    Build TinyLlama chat prompt — focused strictly on disease description.
    Avoids mentioning PubMedCLIP or any system names.
    """
    # Get top confidence score for context
    if top_diseases and len(top_diseases) > 0:
        top_score = f"{top_diseases[0][1]*100:.1f}%"
    else:
        top_score = "high"

    system_msg = (
        "You are a radiologist assistant. Your only job is to write "
        "short, factual, 3-sentence chest X-ray observation reports. "
        "You describe what is visually observed. "
        "You never diagnose. You never mention AI systems or tools. "
        "You write in plain clinical language."
    )

    user_msg = (
        f"Write a 3-sentence chest X-ray observation report for a patient "
        f"whose X-ray shows {disease_label}.\n\n"
        f"Use this information:\n"
        f"- Primary finding: {disease_label}\n"
        f"- Detection confidence: {top_score}\n"
        f"- Visual observation: {caption}\n\n"
        f"Format:\n"
        f"Sentence 1: Describe overall image quality and patient positioning.\n"
        f"Sentence 2: Describe the specific visual findings related to "
        f"{disease_label} seen in the chest X-ray.\n"
        f"Sentence 3: Note any additional observations or confirm no other "
        f"abnormalities are present.\n\n"
        f"Rules: 3 sentences only. No diagnosis. No treatment advice. "
        f"Do not mention any AI or software systems."
    )

    # TinyLlama chat template
    prompt = (
        f"<|system|>\n{system_msg}</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n"
    )

    return prompt


def _refine_with_tinyllama(caption: str, disease_label: str,
                            top_diseases: list = None) -> str:
    """
    Generate report using TinyLlama with GenerationConfig.
    Uses multinomial sampling (num_beams=1, do_sample=True)
    for natural, varied text output.
    """
    prompt = _build_prompt(caption, disease_label, top_diseases)

    # Tokenize prompt
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    # GenerationConfig — tight settings to keep output short and focused
    gen_config = GenerationConfig(
        max_new_tokens=150,      # strict limit — 3 sentences only
        do_sample=True,
        num_beams=1,
        temperature=0.4,         # lower = more focused, less creative
        top_p=0.85,
        repetition_penalty=1.3,  # stronger penalty against repetition
        pad_token_id=_tokenizer.eos_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )

    # Generate
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            generation_config=gen_config,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][input_length:]
    text       = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up leftover chat tags
    for tag in ["<|system|>", "<|user|>", "<|assistant|>", "</s>"]:
        text = text.replace(tag, "").strip()

    # Extract exactly 3 sentences — stop after the third period
    sentences = []
    for part in text.split("."):
        part = part.strip()
        if len(part) > 10:
            sentences.append(part)
        if len(sentences) == 3:
            break

    if sentences:
        text = ". ".join(sentences) + "."

    if not text or len(text) < 20:
        raise ValueError("TinyLlama returned empty or too short response")

    return text


# ─────────────────────────────────────────────────────────────
# BACKEND 2 — Template (always works, no model needed)
# ─────────────────────────────────────────────────────────────
def _refine_with_template(caption: str, disease_label: str,
                           top_diseases: list = None) -> str:
    """
    Template-based report — always works even without any LLM.
    Used as final fallback.
    """
    if top_diseases and len(top_diseases) > 0:
        primary_score = top_diseases[0][1]
        score_str     = f"{primary_score*100:.1f}%"
        others        = [d for d, _ in top_diseases[1:] if d != "No Finding"]
        other_str     = (
            f"Additional observations include possible "
            f"{' and '.join(others)} findings."
            if others else
            "No additional significant findings are observed."
        )
    else:
        score_str = "N/A"
        other_str = "No additional significant findings are observed."

    observations = {
        "Pneumonia"         : "increased opacity and air space consolidation consistent with pneumonia pattern",
        "Effusion"          : "blunting of the costophrenic angle suggesting pleural fluid accumulation",
        "Atelectasis"       : "reduced lung volume with plate-like opacity suggesting partial collapse",
        "Cardiomegaly"      : "enlargement of the cardiac silhouette beyond normal limits",
        "Consolidation"     : "dense homogeneous opacity consistent with consolidation",
        "Edema"             : "bilateral perihilar haziness and vascular prominence",
        "Emphysema"         : "hyperinflated lung fields with flattened diaphragm",
        "Fibrosis"          : "reticular opacity pattern with reduced lung volume",
        "Hernia"            : "bowel gas shadow visible above the diaphragm",
        "Infiltration"      : "patchy haziness in the lung fields",
        "Mass"              : "a focal opacity with irregular borders noted",
        "Nodule"            : "a small focal opacity consistent with a pulmonary nodule",
        "Pleural_Thickening": "thickening along the lateral chest wall pleural surface",
        "Pneumothorax"      : "a visible pleural line with absence of lung markings peripherally",
        "No Finding"        : "clear lung fields with no significant abnormalities identified",
    }

    observation = observations.get(
        disease_label,
        f"findings consistent with {disease_label}"
    )

    return (
        f"The chest X-ray image demonstrates adequate image quality with "
        f"the patient in standard positioning for radiographic evaluation. "
        f"PubMedCLIP analysis with {score_str} similarity confidence identifies "
        f"{observation} in this examination. "
        f"{other_str}"
    )


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def refine_llm(caption: str, disease_label: str,
               top_diseases: list = None) -> dict:
    """
    Generate structured observational report.

    Tries in order:
        1. TinyLlama 1.1B Chat  — true LLM, fluent sentences
        2. Template             — always works, no model needed

    Args:
        caption       : PubMedCLIP disease description
        disease_label : primary predicted disease
        top_diseases  : list of (disease, score) tuples

    Returns:
        {
            "report"        : str,
            "backend"       : str,   "tinyllama" or "template"
            "response_time" : float  seconds
        }
    """
    start = time.time()

    # Try TinyLlama first
    print(f"  [LLM] Using TinyLlama 1.1B Chat")
    if load_tinyllama():
        try:
            report  = _refine_with_tinyllama(caption, disease_label, top_diseases)
            backend = "tinyllama"
            print(f"  [LLM] TinyLlama report generated successfully")
            return {
                "report"        : report,
                "backend"       : backend,
                "response_time" : round(time.time() - start, 3),
            }
        except Exception as e:
            print(f"  [LLM] TinyLlama generation failed: {e}")
            print(f"  [LLM] Switching to template fallback...")

    # Template fallback — always works
    print(f"  [LLM] Using template fallback")
    report  = _refine_with_template(caption, disease_label, top_diseases)
    backend = "template"
    print(f"  [LLM] Template report generated")

    return {
        "report"        : report,
        "backend"       : backend,
        "response_time" : round(time.time() - start, 3),
    }


# ─────────────────────────────────────────────────────────────
# Run directly to test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Report Generation")
    print("  Model : TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("  Link  : https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("=" * 60)

    test_caption = "Chest X-Ray showing pneumonia with lobar opacity and consolidation"
    test_label   = "Pneumonia"
    test_top     = [
        ("Pneumonia",   0.72),
        ("Effusion",    0.18),
        ("No Finding",  0.10),
    ]

    print(f"\n  Disease  : {test_label}")
    print(f"  Caption  : {test_caption}")
    print(f"  Top      : {test_top}\n")

    result = refine_llm(test_caption, test_label, test_top)

    print(f"\n  Backend       : {result['backend']}")
    print(f"  Response time : {result['response_time']}s")
    print(f"\n  Generated Report:")
    print(f"  {result['report']}")
    print("=" * 60)