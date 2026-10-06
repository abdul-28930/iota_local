"""Full local pipeline: embed -> hybrid retrieve (dense + sparse RRF) -> rerank -> generate."""
import os, math
from supabase import create_client
from dotenv import load_dotenv
import embedder, reranker, llm_local
import query_rewriter, answer_critic, multihop_retriever, agent_loop, citation_grounding

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def _sparse_score(query_sparse: dict, chunk_sparse: dict) -> float:
    return sum(w * chunk_sparse.get(tok, 0.0) for tok, w in query_sparse.items())


def _rrf_fuse(dense_ids: list, sparse_ids: list, k: int = 60) -> list:
    scores = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def get_recent_history(session_id: str | None, max_turns: int = 3) -> str:
    """Returns last max_turns Q&A pairs as a condensed text block, or '' if no session/history."""
    if not session_id:
        return ""
    rows = sb.table("messages_local").select("role, content") \
        .eq("session_id", session_id).order("created_at", desc=True) \
        .limit(max_turns * 2).execute().data
    if not rows:
        return ""
    rows = list(reversed(rows))  # chronological order
    lines = []
    for r in rows:
        label = "User" if r["role"] == "user" else "Assistant"
        # cap each historical turn's length so history can't itself blow the context budget
        content = (r["content"] or "")[:400]
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def get_user_memory(user_id: str | None) -> str:
    """Returns all stored long-term facts for this user as a text block, or '' if none."""
    if not user_id:
        return ""
    rows = sb.table("user_memory_local").select("fact") \
        .eq("user_id", user_id).order("created_at").execute().data
    if not rows:
        return ""
    return "\n".join(f"- {r['fact']}" for r in rows)


def maybe_extract_memory(user_id: str | None, session_id: str | None, query: str, answer_text: str):
    """Cheap LLM check: did this exchange reveal a durable fact/preference worth
    remembering across future sessions? Most turns won't -- only store when the
    model is confident there's something durable (a stated role, ongoing project,
    recurring preference), not one-off factual Q&A about a document."""
    if not user_id:
        return
    prompt = f"""/no_think
Does this exchange reveal a durable fact about the USER (their role, ongoing project,
a stated preference, something worth remembering in future unrelated conversations)?
This is NOT about facts from documents -- only facts ABOUT THE USER themselves.

Most exchanges reveal nothing durable. Only answer non-null if you are confident.

Return ONLY JSON: {{"fact": "short first-person-neutral statement, or null"}}

User asked: {query[:300]}
Assistant answered: {answer_text[:300]}"""
    try:
        raw = llm_local.raw_generate(prompt, max_tokens=100)
        raw = raw.strip().strip("```json").strip("```").strip()
        import re, json
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        parsed = json.loads(raw)
        fact = parsed.get("fact")
        if fact and fact.lower() != "null" and len(fact.strip()) > 5:
            sb.table("user_memory_local").insert({
                "user_id": user_id, "fact": fact.strip(), "source_session_id": session_id,
            }).execute()
    except Exception as e:
        print(f"  [memory] extraction skipped: {e}")


def retrieve_chunks(query: str, fetch_k: int = 40, top_n: int = 8) -> list[dict]:
    emb = embedder.embed([query])
    q_dense, q_sparse = emb["dense"][0], emb["sparse"][0]

    dense_hits = sb.rpc("match_chunks_local", {
        "query_embedding": q_dense, "match_count": fetch_k
    }).execute().data

    all_chunks = sb.table("sama_nora_chunks_local").select(
        "id, document_name, section_title, content, sparse_weights"
    ).limit(500).execute().data  # small test corpus; replace with better sparse index at scale
    sparse_scored = sorted(
        all_chunks, key=lambda c: _sparse_score(q_sparse, c.get("sparse_weights") or {}), reverse=True
    )[:fetch_k]

    fused_ids = _rrf_fuse([c["id"] for c in dense_hits], [c["id"] for c in sparse_scored])[:fetch_k]
    by_id = {c["id"]: c for c in dense_hits + sparse_scored}
    candidates = [by_id[i] for i in fused_ids if i in by_id]

    top_chunks = reranker.rerank(query, candidates, top_n=top_n)
    # NOTE: no unload() calls here anymore -- embedder (2.2GB) + reranker (2.2GB)
    # fit together fine (~4.4GB), and the LLM runs in a separate llama-server
    # process with its own VRAM allocation, so there's no conflict keeping
    # both warm across requests. This removes the ~10-15s reload cost per query.
    return top_chunks


def answer(query: str, session_id: str | None = None, user_id: str | None = None,
           fetch_k: int = 40, top_n: int = 8, max_tokens: int = 1200, deep_mode: bool = False, answer_style: str = "normal") -> dict:
    reasoning_trace = []
    history = get_recent_history(session_id)
    long_term = get_user_memory(user_id)

    if deep_mode:
        search_query, rw_reasoning = query_rewriter.rewrite_query(query, history)
        reasoning_trace.append({"stage": "Query interpretation", "reasoning": rw_reasoning})
    else:
        search_query = query

    top_chunks = retrieve_chunks(search_query, fetch_k, top_n)

    if deep_mode:
        top_chunks, mh_reasoning = multihop_retriever.maybe_expand_retrieval(
            search_query, top_chunks, lambda q: retrieve_chunks(q, fetch_k, top_n)
        )
        reasoning_trace.append({"stage": "Retrieval sufficiency", "reasoning": mh_reasoning})

    context, label_map = citation_grounding.build_labeled_context(top_chunks)
    ans = llm_local.generate(query, context, history=history, long_term_memory=long_term, max_tokens=max_tokens, answer_style=answer_style)

    if deep_mode:
        passed, critic_reasoning = answer_critic.check_answer_quality(query, ans, context)
        reasoning_trace.append({"stage": "Answer quality check", "reasoning": critic_reasoning})
        if not passed:
            wider_chunks = retrieve_chunks(search_query, fetch_k=fetch_k + 20, top_n=top_n + 4)
            wider_context, label_map = citation_grounding.build_labeled_context(wider_chunks)
            ans = llm_local.generate(query, wider_context, history=history, long_term_memory=long_term, max_tokens=max_tokens, answer_style=answer_style)
            top_chunks = wider_chunks
            reasoning_trace.append({"stage": "Retry", "reasoning": "First answer didn't pass the quality check, so retried with a wider set of retrieved chunks."})

    citations = citation_grounding.extract_citations(ans, label_map)

    maybe_extract_memory(user_id, session_id, query, ans)
    return {"answer": ans, "sources": top_chunks, "reasoning": reasoning_trace, "citations": citations}


def answer_stream(query: str, session_id: str | None = None, user_id: str | None = None,
                   fetch_k: int = 40, top_n: int = 8, max_tokens: int = 1200, deep_mode: bool = False, answer_style: str = "normal"):
    """Generator yielding live events as the whole pipeline runs:
      {'type': 'stage_start', 'stage': ...}
      {'type': 'thinking_delta', 'stage': ..., 'text': ...}   -- live model reasoning
      {'type': 'stage_done', 'stage': ..., 'summary': ...}
      {'type': 'sources', 'sources': [...]}
      {'type': 'token', 'text': ...}                          -- final answer, streamed
      {'type': 'done', 'full_answer': ..., 'sources': [...]}
    This replaces the old (sources, token_gen, reasoning_trace) tuple return --
    everything is now delivered live, matching how Claude streams thinking
    instead of dumping one final reasoning block after the fact."""
    history = get_recent_history(session_id)
    long_term = get_user_memory(user_id)
    search_query = query

    if deep_mode:
        yield {"type": "stage_start", "stage": "Query interpretation"}
        for ev in query_rewriter.rewrite_query_stream(query, history):
            if ev["type"] == "thinking":
                yield {"type": "thinking_delta", "stage": "Query interpretation", "text": ev["text"]}
            else:
                search_query = ev["search_query"]
                yield {"type": "stage_done", "stage": "Query interpretation", "summary": ev["reasoning"]}

    top_chunks = retrieve_chunks(search_query, fetch_k, top_n)

    if deep_mode:
        yield {"type": "stage_start", "stage": "Retrieval sufficiency"}
        for ev in multihop_retriever.maybe_expand_retrieval_stream(
            search_query, top_chunks, lambda q: retrieve_chunks(q, fetch_k, top_n)
        ):
            if ev["type"] == "thinking":
                yield {"type": "thinking_delta", "stage": "Retrieval sufficiency", "text": ev["text"]}
            else:
                top_chunks = ev["chunks"]
                yield {"type": "stage_done", "stage": "Retrieval sufficiency", "summary": ev["reasoning"]}

    source_payload = [
        {"document_name": c["document_name"], "score": round(c["rerank_score"], 3),
         "section_title": c.get("section_title"), "preview": (c.get("content") or "")[:280]}
        for c in top_chunks if c["rerank_score"] >= 0.35
    ]
    yield {"type": "sources", "sources": source_payload}

    context, label_map = citation_grounding.build_labeled_context(top_chunks)
    full_answer = ""
    for item in llm_local.generate_stream(query, context, history=history, long_term_memory=long_term, max_tokens=max_tokens, answer_style=answer_style):
        if isinstance(item, dict) and item.get("__stats__"):
            yield {"type": "usage_stats", "prompt_tokens": item["prompt_tokens"],
                   "completion_tokens": item["completion_tokens"], "elapsed_sec": item["elapsed_sec"]}
            continue
        full_answer += item
        yield {"type": "token", "text": item}

    if deep_mode:
        yield {"type": "stage_start", "stage": "Answer quality check"}
        passed, critic_reasoning = answer_critic.check_answer_quality(query, full_answer, context)
        yield {"type": "stage_done", "stage": "Answer quality check", "summary": critic_reasoning}
        # Retry-on-fail is intentionally omitted here -- the answer has already
        # streamed to the user token by token, so silently replacing it would be
        # jarring. Instead, surface a visible flag so the failure isn't silently
        # absorbed (this was previously invisible: the critic could say FAIL and
        # nothing happened). Full regeneration still only occurs in answer() below.
        if not passed:
            yield {"type": "quality_warning", "reasoning": critic_reasoning}

    citations = citation_grounding.extract_citations(full_answer, label_map)
    if citations:
        yield {"type": "citations", "citations": citations}

    yield {"type": "done", "full_answer": full_answer, "sources": source_payload}


def answer_stream_agentic(query: str, session_id: str | None = None, user_id: str | None = None,
                           max_tokens: int = 1200):
    """Agent mode: the model decides in a loop whether to search again, check
    its own draft, or finalize -- instead of the fixed single-pass pipeline
    used by deep_mode. Yields the same event shape as answer_stream() so the
    frontend doesn't need special handling, plus 'sources'/'token'/'done' at
    the end once the agent settles on a final answer."""
    history = get_recent_history(session_id)
    long_term = get_user_memory(user_id)

    def retrieve_fn(q):
        return retrieve_chunks(q, fetch_k=40, top_n=8)

    def generate_fn(q, chunks):
        ctx = "\n\n".join(c["content"][:1200] for c in chunks[:8])
        return llm_local.generate(q, ctx, history=history, long_term_memory=long_term, max_tokens=max_tokens)

    final_chunks, final_answer_text, agent_label_map = [], None, {}
    for event in agent_loop.run_agent_stream(query, retrieve_fn, generate_fn):
        if event["type"] == "agent_result":
            final_chunks = event["chunks"]
            final_answer_text = event["answer"]
            agent_label_map = event.get("label_map", {})
        else:
            yield event

    top_chunks = sorted(final_chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)[:8] if final_chunks else []
    source_payload = [
        {"document_name": c["document_name"], "score": round(c.get("rerank_score", 0), 3),
         "section_title": c.get("section_title"), "preview": (c.get("content") or "")[:280]}
        for c in top_chunks
    ]
    yield {"type": "sources", "sources": source_payload}

    # Answer was already decided by the agent loop (FINAL: ...) -- stream it
    # out token-by-token for a consistent UX with the other modes, rather than
    # dumping it all at once.
    if final_answer_text:
        for word in final_answer_text.split(" "):
            yield {"type": "token", "text": word + " "}

    if agent_label_map:
        citations = citation_grounding.extract_citations(final_answer_text or "", agent_label_map)
        if citations:
            yield {"type": "citations", "citations": citations}

    maybe_extract_memory(user_id, session_id, query, final_answer_text or "")
    yield {"type": "done", "full_answer": final_answer_text or "", "sources": source_payload}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the KYC requirements for opening a bank account?"
    result = answer(q)
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for c in result["sources"]:
        print(f"  - {c['document_name']} (rerank={c['rerank_score']:.3f})")
