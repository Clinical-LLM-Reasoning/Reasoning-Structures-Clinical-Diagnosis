# llm_debug.py
import requests, json

URL = "http://localhost:11434/api/generate"
DATA = {
    "model": "qwen3:8b",   # Make sure your local model name is visible in `ollama list`.
    "prompt": "You are a concise assistant. Answer in one short sentence: What is TSH?",
    "stream": False,
    "options": {"num_ctx": 4096, "temperature": 0.2}
}

try:
    r = requests.post(URL, json=DATA, timeout=60)
    r.raise_for_status()
    print("Status:", r.status_code)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])
except Exception as e:
    print("Error contacting Ollama:", e)
