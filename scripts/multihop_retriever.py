"""
Feature #4: Multi-hop retrieval. After the initial retrieval pass, checks
whether the retrieved chunks look sufficient to answer the question, or
whether a second, refined search would help (e.g. the question compares two
things, or the initial chunks only partially cover it). Returns the model's
genuine reasoning for its decision.
"""
import llm_thinking


def maybe_expand_retrieval_stream(query: str, initial_chunks: list[dict], retrieve_fn):
    """Yields {'type':'thinking',...} live, then {'type':'result','chunks':...,'reasoning':...}."""
    preview = "\n".join(f"- {c['document_name']}: {c['content'][:200]}" for c in initial_chunks[:5])
    prompt = f"""A user asked a question and this content was retrieved. Decide if it's
enough to fully answer the question, or if a second, differently-worded search
would help fill a gap. Think it through.

QUESTION: {query}

RETRIEVED CONTENT (preview):
{preview}

If sufficient, end with: "DECISION: SUFFICIENT"
If not, end with: "DECISION: SEARCH: <a new search query that would help>" """

    content_acc, reasoning_acc = "", ""
    for delta in llm_thinking.generate_with_thinking_stream(prompt, max_tokens=700):
        if delta["type"] == "thinking":
            reasoning_acc += delta["text"]
            yield {"type": "thinking", "text": delta["text"]}
        elif delta["type"] == "answer":
            content_acc += delta["text"]

    full_text = content_acc + "\n" + reasoning_acc
    if "DECISION: SUFFICIENT" in full_text.upper():
        yield {"type": "result", "chunks": initial_chunks,
               "reasoning": reasoning_acc.strip() or "Initial retrieval already covers the question well."}
        return

    if "DECISION: SEARCH:" in full_text.upper():
        idx = full_text.upper().find("DECISION: SEARCH:")
        followup_query = full_text[idx + len("DECISION: SEARCH:"):].strip().split("\n")[0].strip()
        if followup_query:
            more_chunks = retrieve_fn(followup_query)
            seen_ids = {c["id"] for c in initial_chunks}
            merged = initial_chunks + [c for c in more_chunks if c["id"] not in seen_ids]
            reasoning_text = reasoning_acc.strip() or f"Initial results were incomplete, so searched again with: \"{followup_query}\""
            yield {"type": "result", "chunks": merged, "reasoning": reasoning_text}
            return

    yield {"type": "result", "chunks": initial_chunks,
           "reasoning": reasoning_acc.strip() or "Could not determine if a second search was needed; using initial results."}


def maybe_expand_retrieval(query: str, initial_chunks: list[dict], retrieve_fn) -> tuple[list[dict], str]:
    """retrieve_fn: callable(query: str) -> list[dict], same shape as retrieve_chunks().
    Returns (final_chunks, reasoning)."""
    preview = "\n".join(f"- {c['document_name']}: {c['content'][:200]}" for c in initial_chunks[:5])
    prompt = f"""A user asked a question and this content was retrieved. Decide if it's
enough to fully answer the question, or if a second, differently-worded search
would help fill a gap. Think it through.

QUESTION: {query}

RETRIEVED CONTENT (preview):
{preview}

If sufficient, end with: "DECISION: SUFFICIENT"
If not, end with: "DECISION: SEARCH: <a new search query that would help>" """

    content, reasoning = llm_thinking.generate_with_thinking(prompt, max_tokens=700)
    full_text = content + "\n" + reasoning

    if "DECISION: SUFFICIENT" in full_text.upper():
        return initial_chunks, (reasoning.strip() or "Initial retrieval already covers the question well.")

    if "DECISION: SEARCH:" in full_text.upper():
        idx = full_text.upper().find("DECISION: SEARCH:")
        followup_query = full_text[idx + len("DECISION: SEARCH:"):].strip().split("\n")[0].strip()
        if followup_query:
            more_chunks = retrieve_fn(followup_query)
            seen_ids = {c["id"] for c in initial_chunks}
            merged = initial_chunks + [c for c in more_chunks if c["id"] not in seen_ids]
            reasoning_text = reasoning.strip() or f"Initial results were incomplete, so searched again with: \"{followup_query}\""
            return merged, reasoning_text

    return initial_chunks, (reasoning.strip() or "Could not determine if a second search was needed; using initial results.")
