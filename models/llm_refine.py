"""
models/llm_refine.py
====================
LLM Report Generation using TinyLlama 1.1B Chat

Model : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Link  : https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
Size  : ~600MB
RAM   : ~1.5GB on CPU

Pure LLM generation — no template fallback.
TinyLlama is a true causal language model trained on 3 trillion tokens.
It generates natural clinical observations from PubMedCLIP predictions.

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
MAX_NEW_TOKENS = 220

_model     = None
_tokenizer = None


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_tinyllama() -> bool:
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
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        _model.to(DEVICE)
        _model.eval()

        print(f"  [LLM] TinyLlama loaded successfully")
        return True

    except Exception as e:
        print(f"  [LLM] Failed to load TinyLlama: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# BUILD PROMPT
# ─────────────────────────────────────────────────────────────
def _build_prompt(caption: str, disease_label: str,
                  top_diseases: list = None) -> str:
    """
    Build TinyLlama chat prompt.

    TinyLlama chat template:
        <|system|> ... </s>
        <|user|>   ... </s>
        <|assistant|>

    The prompt is designed to:
    - Focus on clinical observation language
    - Avoid mentioning AI or software systems
    - Produce exactly 3 structured sentences
    - Stay neutral and non-diagnostic
    """
    # Format confidence scores
    if top_diseases and len(top_diseases) > 0:
        top_score = f"{top_diseases[0][1]*100:.0f}%"
        other_findings = ", ".join(
            d for d, _ in top_diseases[1:]
            if d != "No Finding"
        )
    else:
        top_score      = "high"
        other_findings = ""

    system_msg = (
        "You are a radiologist assistant. "
        "You write short, factual, 3-sentence chest X-ray "
        "observational reports in plain clinical language. "
        "You describe what is visually observed. "
        "You never diagnose. You never mention software or AI. "
        "You write in neutral, professional radiology language."
    )

    user_msg = (
        f"Write a 3-sentence chest X-ray observational report "
        f"for a patient whose X-ray shows {disease_label} "
        f"with {top_score} detection confidence.\n\n"
        f"Visual description: {caption}\n"
        + (f"Additional findings considered: {other_findings}\n"
           if other_findings else "") +
        f"\nFormat your response as exactly 3 sentences:\n"
        f"Sentence 1: Describe the overall image quality and "
        f"patient positioning.\n"
        f"Sentence 2: Describe the specific radiographic findings "
        f"related to {disease_label}.\n"
        f"Sentence 3: Note any additional observations or confirm "
        f"no other significant abnormalities.\n\n"
        f"Rules: exactly 3 sentences, neutral clinical language, "
        f"no diagnosis, no treatment advice, no mention of AI or software."
    )

    return (
        f"<|system|>\n{system_msg}</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n"
    )


# ─────────────────────────────────────────────────────────────
# GENERATE REPORT
# ─────────────────────────────────────────────────────────────
def _generate(caption: str, disease_label: str,
              top_diseases: list = None) -> str:
    """
    Generate clinical report using TinyLlama.

    Uses GenerationConfig with multinomial sampling
    (num_beams=1, do_sample=True) as per HuggingFace docs.
    """
    prompt = _build_prompt(caption, disease_label, top_diseases)

    # Tokenize
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    # Generation config — multinomial sampling
    gen_config = GenerationConfig(
        max_new_tokens    = MAX_NEW_TOKENS,
        do_sample         = True,
        num_beams         = 1,
        temperature       = 0.4,     # lower = more focused
        top_p             = 0.85,    # nucleus sampling
        repetition_penalty= 1.3,     # avoid repeating phrases
        pad_token_id      = _tokenizer.eos_token_id,
        eos_token_id      = _tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            generation_config=gen_config,
        )

    # Decode only new tokens (exclude prompt)
    new_tokens = outputs[0][input_length:]
    text       = _tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    # Clean leftover chat tags
    for tag in ["<|system|>", "<|user|>", "<|assistant|>", "</s>"]:
        text = text.replace(tag, "").strip()

    # Extract first 3 complete sentences
    sentences = []
    for part in text.replace("\n", " ").split("."):
        part = part.strip()
        if len(part) > 15:
            sentences.append(part)
        if len(sentences) == 3:
            break

    if sentences:
        return ". ".join(sentences) + "."

    # If less than 3 sentences came back, return what we have
    if text and len(text) > 20:
        return text

    raise ValueError(f"TinyLlama returned unusable output: '{text}'")


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def refine_llm(caption: str, disease_label: str,
               top_diseases: list = None) -> dict:
    """
    Generate structured clinical report using TinyLlama LLM.

    Args:
        caption       : PubMedCLIP disease description
        disease_label : primary predicted disease
        top_diseases  : list of (disease, score) tuples

    Returns:
        {
            "report"        : str   generated clinical report
            "backend"       : str   "tinyllama"
            "response_time" : float seconds
        }
    """
    start = time.time()

    print(f"  [LLM] Generating report with TinyLlama...")

    if not load_tinyllama():
        raise RuntimeError(
            "TinyLlama failed to load. "
            "Check that torch and transformers are installed."
        )

    report  = _generate(caption, disease_label, top_diseases)
    elapsed = round(time.time() - start, 3)

    print(f"  [LLM] Report generated in {elapsed}s")

    return {
        "report"        : report,
        "backend"       : "tinyllama",
        "response_time" : elapsed,
    }


# ─────────────────────────────────────────────────────────────
# Run directly to test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TinyLlama LLM Report Generation")
    print(f"  Model : {LLM_MODEL}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Test cases covering different diseases
    test_cases = [
        (
            "Chest X-Ray showing pneumonia with lobar opacity",
            "Pneumonia",
            [("Pneumonia", 0.87), ("Consolidation", 0.08)]
        ),
        (
            "Chest X-Ray showing pneumothorax with pleural line",
            "Pneumothorax",
            [("Pneumothorax", 0.92), ("Effusion", 0.05)]
        ),
        (
            "Chest X-Ray showing pleural effusion",
            "Effusion",
            [("Effusion", 0.78), ("Atelectasis", 0.12)]
        ),
    ]

    for caption, label, top in test_cases:
        print(f"\n  Disease : {label}")
        print(f"  Caption : {caption}\n")

        result = refine_llm(caption, label, top)

        print(f"  Backend : {result['backend']}")
        print(f"  Time    : {result['response_time']}s")
        print(f"\n  Report:")
        print(f"  {result['report']}")
        print("-" * 60)