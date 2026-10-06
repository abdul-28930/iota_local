"""
Agent mode: instead of a fixed pipeline (retrieve once -> maybe expand once ->
generate -> maybe critique once), the model itself decides in a loop what to
do next -- search again with a refined query, check its own draft answer's
quality, or finalize. Capped at MAX_ITERATIONS to bound worst-case latency.

Uses a text-based ReAct pattern (SEARCH: / FINAL: / CHECK: prefixes) rather
than native OpenAI-style tool-calling -- more robust for a small quantized
local model, where malformed structured tool-call JSON is a real failure mode
we've already hit today (see extract_metadata_local.py's JSON parsing
fallbacks). Easier to parse reliably, easier to debug when it goes wrong.
"""
import re, time
import llm_thinking
import structured_output

MAX_ITERATIONS = 3
STEP_TIMEOUT_SEC = 90
TOTAL_TIMEOUT_SEC = 240


def _build_agent_prompt(query: str, transcript: list[dict]) -> str:
    history_text = ""
    chunk_counter = 1
    label_map = {}
    for step in transcript:
        if step["action"] == "search":
            history_text += f"\nYou searched: \"{step['query']}\"\nResults found:\n"
            for c in step["results"][:5]:
                history_text += f"  [{chunk_counter}] (from {c['document_name']}): {c['content'][:200]}\n"
                label_map[str(chunk_counter)] = c
                chunk_counter += 1
        elif step["action"] == "check":
            history_text += f"\nYou checked your draft answer. Verdict: {step['verdict']}. Reasoning: {step['reasoning']}\n"

    return f"""You are answering a question by searching a document corpus. You have
three possible actions each turn:

SEARCH: <query>          -- search the document corpus for relevant content
REFINE: <keywords>        -- re-rank what you already found using these keywords, cheaper than a new search
CHECK: <draft answer>     -- check whether a draft answer is well-supported by what you've found
FINAL: <answer>           -- give your final answer, using ONLY information you've found via search

When you give FINAL, you MUST cite the numbered source after every factual
claim, e.g. "Scope includes asset management [2]." Do not make any claim
without a citation. Your FINAL answer must be in the SAME language as the
QUESTION below (Arabic question -> Arabic answer, English question -> English answer).

Think about what you need, then respond with exactly ONE action line.

QUESTION: {query}
{history_text if history_text else "(no searches yet)"}

Respond with exactly one line starting with SEARCH:, CHECK:, or FINAL:""", label_map


def run_agent_stream(query: str, retrieve_fn, generate_fn):
    """Yields live events, same shape as retrieve.py's other streaming stages:
      {'type': 'stage_start', 'stage': 'Agent step N'}
      {'type': 'thinking_delta', 'stage': ..., 'text': ...}
      {'type': 'stage_done', 'stage': ..., 'summary': ...}
    Finally yields {'type': 'agent_result', 'chunks': [...], 'answer': str}."""
    transcript = []
    all_chunks_seen = []
    seen_ids = set()

    loop_start = time.time()
    for i in range(MAX_ITERATIONS):
        if time.time() - loop_start > TOTAL_TIMEOUT_SEC:
            yield {"type": "stage_start", "stage": "Timeout"}
            yield {"type": "stage_done", "stage": "Timeout", "summary": "Total time budget exceeded -- finalizing with what was found."}
            break
        stage_name = f"Agent step {i + 1}"
        yield {"type": "stage_start", "stage": stage_name}

        prompt, label_map = _build_agent_prompt(query, transcript)
        content_acc, reasoning_acc = "", ""
        step_start = time.time()
        for delta in llm_thinking.generate_with_thinking_stream(prompt, max_tokens=600):
            if delta["type"] == "thinking":
                reasoning_acc += delta["text"]
                yield {"type": "thinking_delta", "stage": stage_name, "text": delta["text"]}
            elif delta["type"] == "stats":
                yield {"type": "usage_stats", "prompt_tokens": delta["prompt_tokens"],
                       "completion_tokens": delta["completion_tokens"], "elapsed_sec": delta["elapsed_sec"]}
            else:
                content_acc += delta["text"]
            if time.time() - step_start > STEP_TIMEOUT_SEC:
                reasoning_acc += " [stopped early: step timeout]"
                break

        decision_line = content_acc.strip().split("\n")[0].strip()
        yield {"type": "stage_done", "stage": stage_name, "summary": reasoning_acc.strip() or decision_line}

        action, arg = structured_output.parse_action_line(decision_line, ["SEARCH", "REFINE", "CHECK", "FINAL"])
        if action is None:
            fallback_text = content_acc.strip()
            if not fallback_text:
                # Thinking consumed the whole token budget before any decision
                # was written (e.g. reasoning through conflicting numbers across
                # 8 sources) -- content_acc ends up empty, which previously
                # meant a blank answer with nothing shown to the user. Always
                # generate a real answer from whatever was retrieved instead.
                fallback_text = generate_fn(query, all_chunks_seen) if all_chunks_seen else \
                    "I wasn't able to form an answer in time -- please try rephrasing the question."
            yield {"type": "agent_result", "chunks": all_chunks_seen, "answer": fallback_text, "label_map": label_map}
            return


        if action == "SEARCH":
            results = retrieve_fn(arg)
            for c in results:
                if c["id"] not in seen_ids:
                    all_chunks_seen.append(c)
                    seen_ids.add(c["id"])
            transcript.append({"action": "search", "query": arg, "results": results})

        elif action == "REFINE":
            # Cheap: just re-sort what's already been retrieved by simple
            # keyword overlap with the given terms, instead of paying for a
            # brand-new embedding + retrieval pass.
            keywords = [k.strip().lower() for k in arg.split() if len(k.strip()) > 2]
            def _overlap_score(c):
                text = c["content"].lower()
                return sum(text.count(k) for k in keywords)
            all_chunks_seen.sort(key=_overlap_score, reverse=True)
            transcript.append({"action": "search", "query": f"(refined existing results using: {arg})",
                                "results": all_chunks_seen[:5]})

        elif action == "CHECK":
            context = "\n\n".join(c["content"][:800] for c in all_chunks_seen[:8])
            passed, reasoning = _quick_check(query, arg, context)
            transcript.append({"action": "check", "verdict": "PASS" if passed else "FAIL", "reasoning": reasoning})

        elif action == "FINAL":
            yield {"type": "agent_result", "chunks": all_chunks_seen, "answer": arg, "label_map": label_map}
            return

    # exhausted iterations without a FINAL -- generate a best-effort answer from whatever was found
    yield {"type": "stage_start", "stage": "Finalizing"}
    yield {"type": "stage_done", "stage": "Finalizing",
           "summary": f"Reached the {MAX_ITERATIONS}-step limit without a final answer -- generating from what was found."}
    fallback_answer = generate_fn(query, all_chunks_seen)
    yield {"type": "agent_result", "chunks": all_chunks_seen, "answer": fallback_answer, "label_map": {}}


def _quick_check(query: str, draft_answer: str, context: str) -> tuple[bool, str]:
    import answer_critic
    return answer_critic.check_answer_quality(query, draft_answer, context)
