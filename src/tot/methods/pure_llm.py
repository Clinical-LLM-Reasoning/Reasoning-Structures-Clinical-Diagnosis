# -*- coding: utf-8 -*-
# pure_llm.py — 弱提示、弱格式的纯 LLM baseline（效果更差）

import re
from tot.models import completion

# Still compatible with your global resolver
_FINAL = re.compile(r'\bfinal\s*[:：]?\s*([01])\b', re.IGNORECASE)
def build_prompt(lab_block: str, text_summary: str | None = None) -> str:
    """
    Pure LLM baseline (uses text, but prohibits advanced inference)

    - Mimics real-world NLP text classification: only allows the model to make "overall impression" type judgments

    - Does not use role-playing (avoids loading medical knowledge into the model)

    - Does not require reasoning (avoids automatic CoT triggering)

    - Does not emphasize medical inference (avoids using background knowledge)
    """

    text_part = ""
    if text_summary:
        text_part = (
            "\nAdditional clinical notes:\n"
            f"{text_summary}\n"
        )

    return (
        "Read the following information.\n"
        "Based only on the general impression from the text and numbers,\n"
        "give a simple guess whether the patient could possibly have a thyroid-related issue.\n"
        "This is NOT a diagnostic task, and you do NOT need to apply medical knowledge.\n"
        "Just give the best rough guess you can.\n\n"
        f"{lab_block}\n"
        f"{text_part}"
        "Your answer (0 or 1):"
    )
def parse_pred(output):
    lines = output.strip().splitlines()
    last = lines[-1].strip()
    if last in ("0", "1"):
        return int(last)
    return -1

def solve(args, task, idx, to_print=False):
    """
Complete Pure LLM baseline:

- Prompt blurring → Unstable output → Significantly decreased accuracy

- Preserves result packaging format compatibility with your pipeline
    """

    # 1) input
    x = task.get_input(idx)
    lab_block = x["lab_block"]

    # 2) Text information
    text_summary = None
    if getattr(args, "use_text", False):
        if hasattr(task, "text_info_cache"):
            text_summary = task.text_info_cache.get(idx, None)
        if not text_summary and "text_summary" in x:
            text_summary = x["text_summary"]

    # 3) prompt
    prompt = build_prompt(lab_block, text_summary)

    # 4) Call LLM
    out = completion(prompt, model=args.backend)

    # 5) Analysis
    pred = parse_pred(out)
    if pred not in (0, 1):
        pred = -1

    # 6) Return to uniform format
    final_line = f"Final: {pred}"

    info = {
        "raw_output": out,
        "parsed_label": pred,
        "used_text": bool(text_summary),
    }

    if to_print:
        print("\n=== Weak Pure LLM Baseline ===")
        print("[Prompt]\n", prompt)
        print("\n[Model Output]\n", out)
        print("\n[Final]", final_line)
        print("================================\n")

    return [final_line], info
