# -*- coding: utf-8 -*-
# models.py - Supports a unified interface for local vLLM, Ollama, and remote Qwen/OpenAI.

import json
import requests

PROVIDER = "local"  # Optional: "local" | "vapi" | "openai" | "ollama"

from tot.methods.llm_call_api import get_response as _remote_get_response

# ========== Local vLLM mode ==========
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

# LOCAL_MODEL = "/home/zf1/qwen3_project/hf_cache/Llama3-Med42-8B"

def chat_completion_vllm(messages, model=None,
                         temperature=0.6, max_tokens=1024):

    model = model

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # if messages is None:
    #     return ""
    try:
        res = requests.post(VLLM_API_URL, json=payload, timeout=300)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vLLM Chat] Request failed: {e}\n")
        print(f"[models]: {model}\n[messages]: {messages}\n")
        return ""


# The ToT interface requires a completion message (single prompt).
def completion_vllm(prompt: str, model=None,
                    temperature=0.6, max_tokens=1024):

    model = model

    messages = [
        {"role": "user", "content": prompt}
    ]

    return chat_completion_vllm(messages,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens)


# ========== Remote Qwen / OpenAI mode ==========
def completion_remote(prompt: str, model=None, temperature=0.6, max_tokens=4096):
    model = model
    try:
        return _remote_get_response(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=PROVIDER
        )
    except Exception as e:
        print(f"[Remote LLM] API call failed: {e}")
        return "error"


def chat_completion_remote(messages, model=None):
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"
    return completion_remote(prompt, model=model)


# ========== Local Ollama mode ==========
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2:7b-chat"


def completion_ollama(prompt: str, model=None):
    model_name = model or OLLAMA_MODEL
    payload = {"model": model_name, "prompt": prompt}
    try:
        res = requests.post(OLLAMA_API_URL, data=json.dumps(payload), stream=True)
        if res.status_code == 200:
            output_chunks = []
            for line in res.iter_lines():
                if line:
                    line_data = json.loads(line.decode("utf-8"))
                    output_chunks.append(line_data.get("response", ""))
            return "".join(output_chunks).strip()
        else:
            print(f"[Ollama] LLM error: Status code {res.status_code}")
            return "error"
    except Exception as e:
        print(f"[Ollama] LLM request failed: {e}")
        return "error"


def chat_completion_ollama(messages, model=None):
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"
    return completion_ollama(prompt, model=model)


# ========== Unified external interface ==========
def completion(prompt: str, stop=None, model=None,
               temperature=0.6, max_tokens=1024):

    if PROVIDER == "local":
        return completion_vllm(prompt, model=model,
                               temperature=temperature,
                               max_tokens=max_tokens)

    elif PROVIDER in ["vapi", "openai"]:
        return completion_remote(prompt, model=model,
                                 temperature=temperature,
                                 max_tokens=max_tokens)

    else:
        return completion_ollama(prompt, model=model)


def chat_completion(messages, model=None):

    if PROVIDER == "local":
        return chat_completion_vllm(messages, model=model)

    elif PROVIDER in ["vapi", "openai"]:
        return chat_completion_remote(messages, model=model)

    else:
        return chat_completion_ollama(messages, model=model)
