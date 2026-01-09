# -*- coding: utf-8 -*-
"""
Strict Template Clone (Template-only Decision Tree)
--------------------------------------------------
This implementation is a **direct clone** of the BoT template
matching logic (retrieve_template), with:

- SAME TEMPLATES
- SAME MATCH ORDER
- SAME ctx usage (use_text, med_text_hit, discordant)
- NO fallback LLM
- If no template matches → label = -1

This ensures:
    dtree performance ≤ BoT performance
and dtree coverage < BoT (because BoT has LLM fallback).
"""

import re

# =====================================================================
# ====== 1. Flag Utilities (identical to BoT) ==========================
# =====================================================================

def _flag_of(agg, key):
    item = agg.get(key)
    if not item:
        return None
    f = item.get("flag")
    return f if f in ("HIGH", "LOW", "NORMAL") else None

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

def _discordant_patterns(agg):
    """
    EXACT clone from BoT
    """
    tsh = _flag_of(agg, "tsh")
    ft4 = _flag_of(agg, "ft4")
    t4  = _flag_of(agg, "t4")
    fti = _flag_of(agg, "fti")
    t3  = _flag_of(agg, "t3")

    non_tsh = [ft4, t4, fti, t3]

    if tsh == "NORMAL" and any(f in ("HIGH", "LOW") for f in non_tsh if f):
        return True
    if tsh == "LOW" and any(f == "LOW" for f in [ft4, t4, fti] if f):
        return True
    if tsh == "HIGH" and any(f == "HIGH" for f in [ft4, t4, fti] if f):
        return True
    if t3 in ("HIGH", "LOW") and tsh == "NORMAL" and _all_normal_or_missing(agg, ["ft4", "t4", "fti"]):
        return True
    return False


# =====================================================================
# ====== 2. Medication keyword detection (identical to BoT) ============
# =====================================================================

_MED_KEYWORDS = [
    "levothyroxine", "l-thyroxine", "l thyroxine", "eltroxin", "euthyrox", "lt4",
    "methimazole", "carbimazole", "propylthiouracil", "ptu", "mmi",
    "amiodarone", "lithium", "glucocorticoid", "steroid",
    "prednisone", "dexamethasone", "dopamine", "heparin", "biotin",
]

def _detect_med_keywords(text):
    if not text:
        return False, []
    low = text.lower()
    hits = sorted({kw for kw in _MED_KEYWORDS if kw in low})
    return (len(hits) > 0), hits


# =====================================================================
# ====== 3. Templates (copied 1-to-1 from BoT) =========================
# =====================================================================

def _tpl_hyper(agg, ctx):
    return _flag_of(agg, "tsh") == "LOW" and _any_high(agg, ["ft4","t4","fti","t3"])

def _tpl_hypo(agg, ctx):
    return _flag_of(agg, "tsh") == "HIGH" and _any_low(agg, ["ft4","t4","fti"])

def _tpl_subclinical(agg, ctx):
    return _flag_of(agg, "tsh") in ("HIGH","LOW") and _all_normal_or_missing(agg, ["ft4","t4","fti"])

def _tpl_normal(agg, ctx):
    return _flag_of(agg, "tsh") == "NORMAL" and _all_normal_or_missing(agg, ["ft4","t4","fti"])

def _tpl_med_or_interference(agg, ctx):
    med_hit = bool(ctx.get("med_text_hit"))
    disc = _discordant_patterns(agg)
    return med_hit or disc


# EXACT BoT template order
TEMPLATES = [
    {"name":"medication_or_assay_interference","condition":_tpl_med_or_interference,"label":1},
    {"name":"hyperthyroidism","condition":_tpl_hyper,"label":1},
    {"name":"hypothyroidism","condition":_tpl_hypo,"label":1},
    {"name":"subclinical","condition":_tpl_subclinical,"label":1},
    {"name":"normal","condition":_tpl_normal,"label":0},
]


def _call_condition(tpl, agg, ctx):
    try:
        return bool(tpl["condition"](agg, ctx))
    except TypeError:
        return bool(tpl["condition"](agg))


def retrieve_template(agg, ctx):
    for tpl in TEMPLATES:
        if _call_condition(tpl, agg, ctx):
            return tpl
    return None


# =====================================================================
# ====== 4. Solve (dtree version — identical template use, no LLM) =====
# =====================================================================

def solve(args, task, idx, to_print=True):

    summary_obj = task.get_structured_summary(idx)
    agg = summary_obj["aggregate"]

    # build ctx (strict clone of BoT)
    use_text = bool(getattr(args, "use_text", False))
    if use_text:
        text = task.text_info_cache.get(idx, None)
        med_hit, med_keywords = _detect_med_keywords(text)
    else:
        med_hit, med_keywords = False, []

    ctx = {
        "use_text": use_text,
        "med_text_hit": med_hit,
        "med_keywords": med_keywords,
        "discordant": _discordant_patterns(agg),
    }

    tpl = retrieve_template(agg, ctx)

    # template matched
    if tpl:
        label = tpl["label"]
        final = str(label)

    else:
        label = -1
        final = "-1"

    if to_print:
        print("\n=== Template-Only DTree ===")
        print(summary_obj["summary_text"])
        if tpl:
            print(f"[Matched Template] {tpl['name']}")
        else:
            print("[Matched Template] None → Uncertain")
        print(f"[Final Output] {final}")
        print("=================================\n")

    return [final], {"steps":[]}
