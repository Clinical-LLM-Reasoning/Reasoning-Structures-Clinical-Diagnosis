# -*- coding: utf-8 -*-
# file: src/tot/methods/cot.py
import re
from typing import List, Tuple
from tot.models import completion

# Matches "Final: 0/1"
_FINAL = re.compile(r'\bfinal\s*[:：]?\s*([01])\b', re.IGNORECASE)


def build_cot_prompt(lab_block: str, text_summary: str | None = None) -> str:
    """
    Construct a CoT hint. Optionally, append text information.
    """
    text_part = ""
    if text_summary:  # Only include text if use_text is enabled and there is actually text present.
        text_part = (
            "\n\nAdditional patient information (do not repeat, use only for reasoning):\n"
            f"{text_summary}\n"
        )

    return (
        "You are an experienced endocrinologist.\n"
        "Task: Decide if the patient likely has a thyroid disease (1) or not (0).\n\n"
        "Think step by step privately and DO NOT reveal your reasoning.\n"
        "Output ONLY one line in the exact format:\n"
        "Final: 1  (if disease)  OR  Final: 0 (if not)\n\n"
        f"Patient lab data:\n{lab_block}\n"
        f"{text_part}"
        "Now provide ONLY the final line."
    )


def _parse_final(text: str) -> int:
    """
    Parse the final label from the model output.
    """
    if not text:
        return -1
    m = _FINAL.search(text)
    if m:
        return int(m.group(1))
    m2 = re.search(r'\b([01])\b', text)
    return int(m2.group(1)) if m2 else -1


def solve_one_cot(
    prompt: str,
    n_generate: int = 5,
    model: str = None,
) -> Tuple[int, List[str], List[int]]:
    """
    Given a complete prompt, sample n times to perform self-consistency.
    """
    raws: List[str] = []
    labels: List[int] = []
    for _ in range(n_generate):
        out = completion(prompt, model=model)  # Note: Completion does not support temperature.
        raws.append(out)
        labels.append(_parse_final(out))

    votes = [v for v in labels if v in (0, 1)]
    if not votes:
        return -1, raws, labels
    ones = sum(v == 1 for v in votes)
    zeros = len(votes) - ones
    pred = 1 if ones > zeros else 0
    return pred, raws, labels


def solve(args, task, i, verbose: bool = False):
    """
    CoT-compatible wrapper: Interface is identical to bfs.solve(args, task, i, verbose).

    Returns (res, info).
    """
    x = task.get_input(i)
    lab_block = x['lab_block'] if isinstance(x, dict) else x

    # === Key: Whether to use text ===
    text_summary = None
    if getattr(args, "use_text", False) and isinstance(x, dict):
        # ToT caches text in task.text_info_cache, and this method is also supported here.
        if hasattr(task, "text_info_cache"):
            text_summary = task.text_info_cache.get(i, None)
        # If get_input directly returns text_summary, it is also compatible.
        if not text_summary and "text_summary" in x:
            text_summary = x["text_summary"]

    # Construct a complete prompt
    prompt = build_cot_prompt(lab_block, text_summary)

    # Debug output (optional)
    if verbose:
        print("\n[Debug] CoT Prompt:\n", prompt)

    # Call sampling
    pred, outs, labels = solve_one_cot(
        prompt=prompt,
        n_generate=getattr(args, "n_generate_cot", 5),
        model=args.backend,
    )

    # If parsing fails (pred = -1), the fallback value is 0.
    final_pred = pred if pred in (0, 1) else 0
    final_output = f"Final: {final_pred}"

    info = {
        "cot_samples": [
            {"raw": r, "parsed": (int(p) if p in (0, 1) else -1)}
            for r, p in zip(outs, labels)
        ],
        "used_text": bool(text_summary)  # Did the tag actually use text_summary?
    }

    res = [final_output]
    return res, info
