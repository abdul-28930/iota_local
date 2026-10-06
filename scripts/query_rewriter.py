"""
Feature #1: Query rewriting. Resolves follow-up references ("what about that
instead?") into a standalone search query, using conversation history. Returns
the model's genuine reasoning about how it interpreted the question.
"""
import llm_thinking


def rewrite_query_stream(query: str, history: str):
    """Yields {'type':'thinking','text':...} live, then finally
    {'type':'result','search_query':...,'reasoning':...}."""
    if not history:
        yield {"type": "result", "search_query": query,
               "reasoning": "No prior conversation in this chat, so the question is used as-is."}
        return

    prompt = f"""Given this conversation history and a follow-up question, rewrite the
follow-up into a standalone question that makes sense without the history.
If it's already standalone, return it unchanged.

CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION: {query}

Return ONLY the rewritten standalone question, nothing else."""

    content_acc, reasoning_acc = "", ""
    for delta in llm_thinking.generate_with_thinking_stream(prompt, max_tokens=700):
        if delta["type"] == "thinking":
            reasoning_acc += delta["text"]
            yield {"type": "thinking", "text": delta["text"]}
        elif delta["type"] == "answer":
            content_acc += delta["text"]
        # "stats" events are ignored here -- only agent_loop.py forwards usage
        # stats currently; this file's caller doesn't expose them separately.

    rewritten = content_acc.strip() or query
    reasoning = reasoning_acc.strip() or f"Interpreted as: \"{rewritten}\""
    yield {"type": "result", "search_query": rewritten, "reasoning": reasoning}


def rewrite_query(query: str, history: str) -> tuple[str, str]:
    """Returns (search_query, reasoning). If there's no history, skips the LLM
    call entirely -- nothing to resolve on a fresh chat."""
    if not history:
        return query, "No prior conversation in this chat, so the question is used as-is."

    prompt = f"""Given this conversation history and a follow-up question, rewrite the
follow-up into a standalone question that makes sense without the history.
If it's already standalone, return it unchanged.

CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION: {query}

Return ONLY the rewritten standalone question, nothing else."""

    content, reasoning = llm_thinking.generate_with_thinking(prompt, max_tokens=700)
    rewritten = content.strip() or query
    if not reasoning:
        reasoning = f"Interpreted as: \"{rewritten}\""
    return rewritten, reasoning
