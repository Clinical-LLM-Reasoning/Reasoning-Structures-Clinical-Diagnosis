"""
The Buffer of Thoughts (Buffer of Thoughts) method is specialized for thyroid lab reasoning.

Pipeline:
1) Distill (Problem Distiller): Extracts and standardizes core metrics and flags using `task.get_structured_summary(idx)`.
2) Retrieve (Meta-buffer): Pure code matching of predefined medical templates (preventing LLM from performing numerical comparisons).
3) Instantiate & Reason: Combines the reasoning path and conclusion.
4) Fallback to LLM: Only when the template is not matched (atypical).

Return Format: Aligned with CoT, `solve` returns `(["Final: 0/1"], info)`.
`info['steps']` records complete intermediate information for each stage, facilitating disk entry in the checkpoints of `main.py`.
"""
import re
from tot.models import completion

# =============== Flag Utility functions (using only program flags to avoid unit/threshold differences) ===============
def _flag_of(agg: dict, key: str) -> str | None:
    item = agg.get(key)
    if not item:
        return None
    f = item.get("flag")
    return f if isinstance(f, str) and f in ("HIGH", "LOW", "NORMAL") else None

def _any_high(agg, keys):
    return any(_flag_of(agg, k) == "HIGH" for k in keys if k in agg)

def _any_low(agg, keys):
    return any(_flag_of(agg, k) == "LOW" for k in keys if k in agg)

def _all_normal_or_missing(agg, keys):
    for k in keys:
        f = _flag_of(agg, k)
        if f is None:
            continue
        if f != "NORMAL":
            return False
    return True

def _discordant_patterns(agg) -> bool:
    """
        Detecting "Inconsistent/Illogical" Indicator Patterns:

        - TSH is normal, but any of FT4/T4/FT1/T3 is abnormal.

        - TSH is low, but FT4/T4/FT1 are also low (opposite direction).

        - TSH is high, but FT4/T4/FT1 are high (opposite direction).

        - Only T3 is abnormal, while TSH and FT4/T4/FT1 are normal (e.g., due to medication, non-thyroid syndrome).
    """
    tsh = _flag_of(agg, "tsh")
    ft4 = _flag_of(agg, "ft4")
    t4  = _flag_of(agg, "t4")
    fti = _flag_of(agg, "fti")
    t3  = _flag_of(agg, "t3")

    non_tsh = [ft4, t4, fti, t3]
    if tsh == "NORMAL" and any(f in ("HIGH", "LOW") for f in non_tsh if f is not None):
        return True
    if tsh == "LOW" and any(f == "LOW" for f in [ft4, t4, fti] if f is not None):
        return True
    if tsh == "HIGH" and any(f == "HIGH" for f in [ft4, t4, fti] if f is not None):
        return True
    # isolated T3 abnormal
    if t3 in ("HIGH", "LOW") and tsh == "NORMAL" and _all_normal_or_missing(agg, ["ft4", "t4", "fti"]):
        return True
    return False

# =============== Medication text detection (enabled only when args.use_text=True) ===============
_MED_KEYWORDS = [
    # Thyroid hormone replacement
    "levothyroxine", "l-thyroxine", "l thyroxine", "eltroxin", "euthyrox", "lt4",
    # Antithyroid drugs
    "methimazole", "carbimazole", "propylthiouracil", "ptu", "mmi",
    # Affecting the thyroid axis/transition
    "amiodarone", "lithium", "glucocorticoid", "steroid", "prednisone", "dexamethasone",
    "dopamine", "heparin", "biotin",
]

def _collect_text_summary(task, idx) -> str | None:
    """
    Called only when the outer layer checks for permission; no fallback scan is performed internally.

    It prioritizes retrieving from task.text_info_cache (it will only be populated if use_text=True during get_input).
    """
    try:
        if hasattr(task, "text_info_cache"):
            ts = task.text_info_cache.get(idx, None)
            if ts:
                return str(ts)
    except Exception:
        pass
    return None

def _detect_med_keywords(text: str | None):
    if not text:
        return False, []
    low = text.lower()
    hits = sorted({kw for kw in _MED_KEYWORDS if kw in low})
    return (len(hits) > 0), hits

# =============== Template collection (all based on flag; med templates support ctx) ===============
def _tpl_hyper_condition(agg, ctx=None):
    return _flag_of(agg, "tsh") == "LOW" and _any_high(agg, ["ft4", "t4", "fti", "t3"])

def _tpl_hypo_condition(agg, ctx=None):
    return _flag_of(agg, "tsh") == "HIGH" and _any_low(agg, ["ft4", "t4", "fti"])

def _tpl_subclinical_condition(agg, ctx=None):
    return (_flag_of(agg, "tsh") in ("HIGH", "LOW")) and _all_normal_or_missing(agg, ["ft4", "t4", "fti"])

def _tpl_normal_condition(agg, ctx=None):
    return (_flag_of(agg, "tsh") == "NORMAL") and _all_normal_or_missing(agg, ["ft4", "t4", "fti"])

def _tpl_med_or_interference_condition(agg, ctx=None):
    """
    Triggering conditions:

    - The text contains a keyword related to medication (only if use_text=True and the keyword is matched)

    - Or, a "discordant pattern" exists (_discordant_patterns).
    """
    med_hit = bool(ctx and ctx.get("med_text_hit"))
    return med_hit or _discordant_patterns(agg)

TEMPLATES = [
    # Medication/interference templates have priority
    {
        "name": "medication_or_assay_interference",
        "trigger": "Medication/Assay Interference",
        "condition": _tpl_med_or_interference_condition,
        "explain": "Laboratory indicators show discordant patterns or the text suggests medication/interference, possibly affected by drugs (e.g., LT4 replacement, antithyroid drugs, amiodarone, steroids, etc.) or assay interference (e.g., biotin).",
        "diagnosis": "Interpret thyroid function cautiously in combination with medication history and clinical context; recommend retesting or repeating after discontinuing interfering factors.",
        "suggest": "Carefully verify recent medications and evaluate potential interferences; if necessary, recheck FT4, TSH, T3 and related antibodies, as well as ultrasound.",
        "label": 1
    },
    {
        "name": "hyperthyroidism",
        "trigger": "Hyperthyroidism",
        "condition": _tpl_hyper_condition,
        "explain": "Low TSH accompanied by elevation of one of FT4/T4/FTI/T3 suggests excessive thyroid hormone secretion.",
        "diagnosis": "Highly suspicious for primary hyperthyroidism.",
        "suggest": "Recommend testing thyroid antibodies (such as TPOAb) and thyroid ultrasound to clarify etiology; if T3 is elevated while T4 is normal, consider T3-predominant hyperthyroidism.",
        "label": 1
    },
    {
        "name": "hypothyroidism",
        "trigger": "Hypothyroidism",
        "condition": _tpl_hypo_condition,
        "explain": "Elevated TSH accompanied by a decrease in one of FT4/T4/FTI suggests insufficient thyroid hormone secretion.",
        "diagnosis": "Highly suspicious for primary hypothyroidism.",
        "suggest": "Recommend testing thyroid autoantibodies (such as TPOAb) to evaluate for Hashimoto’s thyroiditis.",
        "label": 1
    },
    {
        "name": "subclinical",
        "trigger": "Subclinical Abnormality",
        "condition": _tpl_subclinical_condition,
        "explain": "TSH is abnormal but FT4/T4/FTI remain within the reference range, consistent with subclinical thyroid dysfunction.",
        "diagnosis": "Possible subclinical thyroid dysfunction.",
        "suggest": "Recommend periodic thyroid function retesting, and decide on intervention based on clinical symptoms and risk factors.",
        "label": 1
    },
    {
        "name": "normal",
        "trigger": "Normal",
        "condition": _tpl_normal_condition,
        "explain": "Core thyroid hormones are all within the normal range.",
        "diagnosis": "Current thyroid function results are normal.",
        "suggest": "If relevant clinical symptoms persist, consider disorders of other systems.",
        "label": 0
    }
]


# =============== Template calling and intermediate information recording ===============
def _truncate(text: str, limit: int = 2000) -> str:
    if not isinstance(text, str):
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text)-limit} chars]"

def _call_condition(tpl, agg, ctx):
    try:
        # It is uniformly called using (agg, ctx); if the template implementation does not use ctx, it will not have any impact.
        return bool(tpl["condition"](agg, ctx))
    except TypeError:
        # Backward compatible single-parameter implementation
        try:
            return bool(tpl["condition"](agg))
        except Exception:
            return False
    except Exception:
        return False

def retrieve_template(agg, ctx):
    for tpl in TEMPLATES:
        if _call_condition(tpl, agg, ctx):
            return tpl
    return None

def evaluate_templates(agg, ctx):
    """
    Returns the hit status of each template, facilitating the writing of intermediate information.
    """
    results = []
    for tpl in TEMPLATES:
        ok = _call_condition(tpl, agg, ctx)
        results.append({
            "name": tpl["name"],
            "trigger": tpl["trigger"],
            "matched": bool(ok)
        })
    return results

def _aggregate_snapshot(agg):
    """
    Only the core indicators (value, flag, lower, and upper) are extracted for easier placement on the trading board.
    """
    snap = {}
    for key in ("tsh", "ft4", "t3", "t4", "fti", "tpoab"):
        if key in agg:
            item = agg[key]
            snap[key] = {
                "value": item.get("value"),
                "flag": item.get("flag"),
                "lower": item.get("lower"),
                "upper": item.get("upper"),
            }
    return snap

def instantiate_reasoning(summary_obj, template, ctx):
    parts = []
    parts.append(f"[Problem Summary] {summary_obj['summary_text']}")
    parts.append(f"[Matched Template] {template['trigger']}")
    parts.append(f"[Rule Explanation] {template['explain']}")
    # Supporting evidence: Elevated TPOAb suggests an autoimmune background; medication mentions in text also serve as hints (only when use_text=True and matched)
    agg = summary_obj.get("aggregate", {})
    if _flag_of(agg, "tpoab") == "HIGH":
        parts.append("[Supporting Evidence] TPOAb is elevated, suggesting an autoimmune background.")
    if ctx and ctx.get("med_text_hit"):
        kws = ctx.get("med_keywords") or []
        if kws:
            parts.append(f"[Supporting Evidence] Text indicates possible medication/interference keyword hits: {', '.join(kws)}.")
    if template["name"] == "medication_or_assay_interference" and ctx and ctx.get("discordant"):
        parts.append("[Indicator Features] Discordant/illogical indicator combinations present, consider medication or assay interference.")
    parts.append(f"[Preliminary Diagnosis] {template['diagnosis']}")
    parts.append(f"[Further Suggestions] {template['suggest']}")
    return "\n".join(parts)

def fallback_llm(summary_obj, backend, temperature=0.2):
    prompt = (
        "You will be given a structured summary based on thyroid laboratory results. "
        "Please determine the possible thyroid functional status (hyperthyroidism/hypothyroidism/subclinical/normal), "
        "and output only 1 (thyroid disease present) or 0 (no disease) on the final line.\n"
        f"Summary: {summary_obj['summary_text']}\n"
        "Please follow the format below:\n"
        "[Reasoning] ...\n[Conclusion] ...\n[Label] 1 or 0"
    )
    out = completion(prompt, model=backend, temperature=temperature, stop=None)
    return out.strip()


def _parse_label_from_text(text: str) -> int:
    m = re.search(r'\bfinal\s*[:：]?\s*([01])\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r'\b([01])\b', text)
    return int(m2.group(1)) if m2 else -1

# =============== main entrance ===============
def solve(args, task, idx, to_print=True):
    steps = []

    # 0) Distill
    summary_obj = task.get_structured_summary(idx)
    agg = summary_obj["aggregate"]

    # Enables text-based medication clues only if args.use_text is True.
    use_text_enabled = bool(getattr(args, "use_text", False))

    if use_text_enabled:
        text_summary = _collect_text_summary(task, idx)
        med_text_hit, med_keywords = _detect_med_keywords(text_summary)
        text_excerpt = _truncate(text_summary, 800) if text_summary else None
    else:
        text_summary = None
        med_text_hit, med_keywords = False, []
        text_excerpt = None

    discordant = _discordant_patterns(agg)

    # Optional: flag_block digest (to avoid truncating excessively large JSON data)
    try:
        flag_block_text = task.get_flag_input(idx)["flag_block"]
        flag_block_excerpt = _truncate(flag_block_text, 1200)
    except Exception:
        flag_block_excerpt = None

    ctx = {
        "use_text": use_text_enabled,
        "med_text_hit": med_text_hit if use_text_enabled else False,
        "med_keywords": med_keywords if use_text_enabled else [],
        "text_excerpt": text_excerpt if use_text_enabled else None,
        "discordant": discordant,
    }

    step0 = {
        "stage": "distill",
        "subject_id": summary_obj.get("subject_id"),
        "n_sessions": len(summary_obj.get("sessions", [])),
        "summary_text": summary_obj.get("summary_text"),
        "aggregate_snapshot": _aggregate_snapshot(agg),
        "flag_block_excerpt": flag_block_excerpt,
        "use_text": use_text_enabled,
        "med_text_hit": ctx["med_text_hit"],
        "med_keywords": ctx["med_keywords"],
        "discordant": discordant,
        "text_excerpt": ctx["text_excerpt"],
    }
    steps.append(step0)

    # 1) Retrieve 模板评估（带 ctx）
    tmpl_eval = evaluate_templates(agg, ctx)
    template = retrieve_template(agg, ctx)

    step1 = {
        "stage": "retrieve",
        "templates_evaluated": tmpl_eval,
        "matched_template": template["name"] if template else None
    }
    steps.append(step1)

    # 2) Reason (template instantiation or LLM fallback)
    if template:
        reasoning_text = instantiate_reasoning(summary_obj, template, ctx)
        final_label = int(template["label"])
        reason_source = "template"
    else:
        reasoning_text = fallback_llm(summary_obj, args.backend, temperature=args.temperature)
        final_label = _parse_label_from_text(reasoning_text)
        if final_label not in (0, 1):
            final_label = 0
        reason_source = "llm_fallback"

    final_line = f"Final: {final_label}"
    step2 = {
        "stage": "reason",
        "source": reason_source,
        "reasoning_text": reasoning_text,
        "predicted_label": final_label,
        "final_line": final_line
    }
    steps.append(step2)

    # Console printing (optional)
    if to_print:
        print("\n=== Buffer of Thoughts Pipeline ===")
        print("[Distilled Summary]")
        print(summary_obj["summary_text"])
        print("\n[Template Match]" if template else "\n[Template Match] None, using LLM fallback")
        if template:
            print(f"Matched: {template['name']}")
        print("\n[Reasoning]")
        print(reasoning_text)
        print(f"\n[Final Output] {final_line}")
        print("===================================\n")

    # Main return (aligned with CoT)
    res = [final_line]
    return res, {"steps": steps}
