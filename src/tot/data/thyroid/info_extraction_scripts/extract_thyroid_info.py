# extract_thyroid_info.py
from llm_interface import build_prompt, query_llm, remove_think
import re

def extract_thyroid_info(discharge_text, radiology_text, model="qwen3:8b"):
    prompt = build_prompt(discharge_text or "", radiology_text or "")
    raw = query_llm(prompt, model=model)
    # Clean up the <think> paragraph
    return remove_think(raw or "")
