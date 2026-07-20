"""
models/prompts.py
=================
Prompt templates — BiomedCLIP pipeline
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = (
    "You are an AI assistant that helps describe chest X-ray observations "
    "in plain, structured language. You do NOT provide medical diagnoses or "
    "treatment recommendations. You describe visual observations based on "
    "BiomedCLIP medical image analysis predictions. "
    "Your descriptions are factual, neutral, and purely observational."
)


def get_report_prompt(caption: str, disease_label: str,
                      top_diseases: list = None) -> str:
    if top_diseases:
        disease_lines = "\n".join(
            f"  - {d}: {s*100:.1f}% similarity"
            for d, s in top_diseases
        )
        detection_section = (
            f"BiomedCLIP image-text matching results:\n"
            f"{disease_lines}\n"
            f"Primary finding: {disease_label}"
        )
    else:
        detection_section = f"BiomedCLIP detected: {disease_label}"

    return (
        f"You are an AI assistant describing chest X-ray observations. "
        f"Do not diagnose. Only describe observations.\n\n"
        f"--- BiomedCLIP Analysis ---\n"
        f"{detection_section}\n\n"
        f"Medical description: \"{caption}\"\n\n"
        f"Write a structured observational report in exactly 3 sentences:\n"
        f"Sentence 1: Describe overall chest X-ray image appearance and quality.\n"
        f"Sentence 2: Describe visual observations related to {disease_label}.\n"
        f"Sentence 3: Summarize model confidence and additional observations.\n\n"
        f"Neutral language only. Do not diagnose."
    )


def get_summary_prompt(caption: str, disease_label: str,
                       top_diseases: list = None) -> str:
    if top_diseases:
        top_str = ", ".join(f"{d} ({s*100:.0f}%)" for d, s in top_diseases)
    else:
        top_str = disease_label
    return (
        f"BiomedCLIP detected: {top_str}. "
        f"Description: \"{caption}\". "
        f"Write one neutral observational sentence. Do not diagnose."
    )


def get_structured_prompt(caption: str, disease_label: str,
                          top_diseases: list = None) -> str:
    if top_diseases:
        top_str = ", ".join(f"{d} ({s*100:.0f}%)" for d, s in top_diseases)
    else:
        top_str = disease_label
    return (
        f"BiomedCLIP predictions: \"{top_str}\"\n"
        f"Primary finding: \"{disease_label}\"\n"
        f"Description: \"{caption}\"\n\n"
        f"Respond ONLY with valid JSON:\n"
        f"{{\n"
        f"    \"image_quality\"   : \"image quality description\",\n"
        f"    \"lung_fields\"     : \"lung field observations\",\n"
        f"    \"primary_finding\" : \"observation for {disease_label}\",\n"
        f"    \"confidence\"      : \"BiomedCLIP confidence summary\",\n"
        f"    \"other_findings\"  : \"other observations or none\"\n"
        f"}}"
    )


if __name__ == "__main__":
    test_caption = "Chest X-Ray showing pneumonia with lobar opacity"
    test_label   = "Pneumonia"
    test_top     = [("Pneumonia", 0.72), ("Effusion", 0.18), ("No Finding", 0.10)]

    print("=" * 60)
    print("  Prompts Preview — BiomedCLIP")
    print("=" * 60)
    print("\n[1] REPORT PROMPT:")
    print(get_report_prompt(test_caption, test_label, test_top))
    print("\n[2] SUMMARY:")
    print(get_summary_prompt(test_caption, test_label, test_top))
    print("\n" + "=" * 60)