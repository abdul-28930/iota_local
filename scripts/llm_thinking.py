"""
Shared helper for LLM calls used by the reasoning-visible features
(query rewriting, answer critique, multi-hop retrieval). Unlike llm_local.py's
raw_generate() (which suppresses thinking via /no_think for speed), these calls
keep thinking mode ON so we can surface the model's genuine reasoning to the
user -- not a synthetic log of what code ran, but the model's actual
chain-of-thought for that specific decision.
"""
import requests, time

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"


import json as _json


def generate_with_thinking_stream(prompt: str, max_tokens: int = 700, temperature: float = 0.1):
    """Yields {'type': 'thinking', 'text': ...} and {'type': 'answer', 'text': ...}
    deltas live as the model generates them -- mirrors llm_local.generate_stream()'s
    SSE parsing, but separates reasoning_content from content so callers can show
    live thinking the same way Claude's UI does, instead of waiting for one final block."""
    with requests.post(LLAMA_SERVER_URL, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        start = time.time()
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("reasoning_content"):
                        yield {"type": "thinking", "text": delta["reasoning_content"]}
                    if delta.get("content"):
                        yield {"type": "answer", "text": delta["content"]}
            except Exception:
                continue
        yield {"type": "stats", "prompt_tokens": usage.get("prompt_tokens", 0),
               "completion_tokens": usage.get("completion_tokens", 0),
               "elapsed_sec": round(time.time() - start, 2)}


def generate_with_thinking(prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> tuple[str, str]:
    """Returns (content, reasoning). Thinking mode stays enabled (no /no_think),
    so reasoning_content reflects genuine model deliberation for this call."""
    resp = requests.post(LLAMA_SERVER_URL, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=120)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    return content, reasoning
