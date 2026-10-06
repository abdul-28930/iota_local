"""
Qwen local generation via llama.cpp server (OpenAI-compatible API).
Start server separately (keeps model warm between requests):
  llama-server -m Qwen3-4B-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080
"""
import requests

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a Saudi regulatory compliance assistant. You must answer using ONLY the "
    "facts, numbers, and wording found in the CONTEXT below. Do not use any outside "
    "knowledge, training data, or general facts about cybersecurity/regulations/NTP/etc., "
    "even if you know them to be true elsewhere -- this document set is the only source "
    "of truth for this task. If the context does not contain the answer, respond exactly: "
    "\"The provided documents do not contain information to answer this question.\" "
    "Do not fill gaps with plausible-sounding general knowledge. If the recent conversation "
    "below is relevant to interpreting the current question (e.g. it uses 'it', 'that', "
    "'instead', or otherwise refers back to something earlier), use it to understand what "
    "is being asked -- but still answer only from CONTEXT, never from the conversation's "
    "own prior wording as if it were a source.\n\n"
    "IMPORTANT: After every factual claim, cite which numbered source it came from using "
    "square brackets, e.g. \"Scope includes asset management [2].\" If a sentence combines "
    "info from multiple sources, cite all of them: \"...[1][3]\". Every sentence with a "
    "factual claim MUST have at least one citation. Do not make any claim without a "
    "citation -- if you can't cite it, don't say it.\n\n"
    "LANGUAGE: Always answer in the SAME language as the QUESTION. If the question is "
    "written in Arabic, your entire answer must be in Arabic. If it is in English, "
    "answer in English. Do this regardless of what language the CONTEXT is written in "
    "-- translate/synthesize the cited facts into the question's language, do not switch "
    "languages partway through the answer."
)


import skills_loader

STYLE_INSTRUCTIONS = {
    "summarized": "Answer in 3-5 lines total. Be concise -- only the single most essential point(s), no elaboration.",
    "normal": "Answer in 5-10 lines. Cover the main points clearly, with brief supporting detail.",
    "deep": "Answer in 10-20 lines. Be comprehensive: cover all relevant points from the context, use a bulleted list for distinct points, include nuances/exceptions/scoping conditions where the sources mention them.",
}

def _build_prompt(query: str, context: str, history: str = "", long_term_memory: str = "", answer_style: str = "normal") -> str:
    memory_block = f"\n\nWHAT YOU KNOW ABOUT THIS USER (from past sessions, for context only):\n{long_term_memory}" if long_term_memory else ""
    history_block = f"\n\nRECENT CONVERSATION (for reference only, not a source):\n{history}" if history else ""
    skills_block = skills_loader.load_relevant_skills(context)
    skills_block = f"\n\n{skills_block}" if skills_block else ""
    style_instruction = STYLE_INSTRUCTIONS.get(answer_style, STYLE_INSTRUCTIONS["normal"])
    return (
        f"{SYSTEM_PROMPT}{memory_block}{history_block}{skills_block}\n\n"
        f"ANSWER LENGTH: {style_instruction}\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER (context-only):"
    )


def generate_stream(query: str, context: str, history: str = "", long_term_memory: str = "", max_tokens: int = 1200, answer_style: str = "normal"):
    """Yields answer text incrementally, then a final dict with token/timing
    stats: {'__stats__': True, 'prompt_tokens': N, 'completion_tokens': N,
    'elapsed_sec': N} -- lets the caller show input/output token counts and
    time taken, useful for diagnosing budget-exhaustion issues like the
    Root CA case where thinking ate the whole token budget silently."""
    import json as _json, time
    start = time.time()
    prompt = _build_prompt(query, context, history, long_term_memory)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    with requests.post(LLAMA_SERVER_URL, json={
        "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }, stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            print("LLAMA-SERVER ERROR BODY:", resp.text[:500])
        resp.raise_for_status()
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
                    text = choices[0].get("delta", {}).get("content") or ""
                    if text:
                        yield text
            except Exception:
                continue
    yield {
        "__stats__": True,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_sec": round(time.time() - start, 2),
    }


def generate(query: str, context: str, history: str = "", long_term_memory: str = "", max_tokens: int = 1200, answer_style: str = "normal") -> str:
    prompt = _build_prompt(query, context, history, long_term_memory, answer_style)
    return raw_generate(prompt, max_tokens=max_tokens)


def raw_generate(prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
    """No RAG wrapper — use for metadata extraction, classification, etc."""
    resp = requests.post(LLAMA_SERVER_URL, json={
        "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return content
