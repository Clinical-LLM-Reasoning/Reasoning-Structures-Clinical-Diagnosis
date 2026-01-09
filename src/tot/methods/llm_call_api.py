# -*- coding: utf-8 -*-
# file: src/tot/methods/llm_call_api.py
import os
import requests
from typing import Optional
from openai import OpenAI

# ====== Three platform configurations ======

# Your own server (vLLM)
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
LOCAL_API_KEY = "not-needed"

# Qwen(V-API)
VAPI_BASE_URL = os.getenv("VAPI_BASE_URL", "https://api.gpt.ge/v1/")
VAPI_API_KEY = os.getenv("VAPI_API_KEY", "sk-XXXXXXXX")  # Your own Key

#OpenAI
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxx")  # Your own Key


def get_response(
    prompt: str,
    # model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    # model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    # model: str = "/home/zf1/qwen3_project/hf_cache/Llama3-Med42-8B",
    model: str,
    temperature: Optional[float] = 0.6,
    max_tokens: int = 1024,
    provider: str = "local",  # You must manually specify: "local", "vapi", or "openai".
    seed: Optional[int] = None,
) -> str:
    """
    Manually select the source of the call:
      provider="local"  -> Call the vLLM Qwen running on your server
      provider="vapi"   -> Call the Qwen cloud API (https://api.gpt.ge/v1/)
      provider="openai" -> Calling the OpenAI official API
    """
    if provider == "local":
        # ====== Your server vLLM ======
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            r = requests.post(
                # f"{LOCAL_BASE_URL}/chat/completions",
                f"{LOCAL_BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[ERROR] The call to the server vLLM failed.: {e}")
            return ""

    elif provider == "vapi":
        # ====== Use requests to directly call VAPI ======
        url = f"{VAPI_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VAPI_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": float(temperature),
        }

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] VAPI call failed: {e}")
            return "error"

    # ====== General OpenAI SDK calling section ======
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if seed is not None:
        kwargs["seed"] = int(seed)

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# Retain compatibility with older interfaces
def get_response_from_api_gpt(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.6,
    max_tokens: int = 1024,
) -> str:
    return get_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider="openai",  # The default version uses OpenAI.
    )
