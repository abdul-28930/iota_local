"""
Generate 50-100+ test questions from sama_nora_chunks_local for benchmarking.
Samples chunks across documents/languages, asks Qwen (local) to write one
question per chunk it could answer, writes output in the same jsonl shape
as your existing generated_test_questions.jsonl so benchmark.py-style tooling
can reuse it.

Usage: python generate_questions.py [n_questions] [output_path]
"""
import sys, os, json, random, csv
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

import llm_local

def sample_chunks(n: int) -> list[dict]:
    """Pull a spread of chunks across documents/languages, not just top N rows."""
    all_rows = sb.table("sama_nora_chunks_local") \
        .select("id, document_name, content, language, token_count") \
        .execute().data
    # only chunks with enough real content to form a good question
    candidates = [r for r in all_rows if r.get("token_count", 0) and r["token_count"] >= 60]
    random.shuffle(candidates)
    return candidates[:n]

def make_question(chunk: dict) -> dict | None:
    lang = chunk.get("language") or "en"
    lang_instruction = "Arabic" if lang == "ar" else "English"
    prompt = f"""/no_think
Read this excerpt from a Saudi regulatory/cybersecurity document. Write ONE
specific, answerable question in {lang_instruction} that this excerpt directly
answers. The question should be natural (like a compliance officer would ask),
not generic. Return ONLY the question text, nothing else -- no quotes, no labels.

Excerpt:
{chunk['content'][:1200]}"""
    try:
        q = llm_local.raw_generate(prompt, max_tokens=150)
        q = q.strip().strip('"').strip()
        if len(q) < 10:
            return None
        return {
            "chunk_id": chunk["id"],
            "document_name": chunk["document_name"],
            "language": lang,
            "question": q,
            "source_preview": chunk["content"][:300],
        }
    except Exception as e:
        print(f"  [gen] failed for chunk {chunk['id']}: {e}")
        return None

def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("local_test_questions.jsonl")

    chunks = sample_chunks(n_target + 20)  # oversample, some will fail
    print(f"Sampled {len(chunks)} candidate chunks, generating up to {n_target} questions...")

    written = 0
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if written >= n_target:
                break
            q = make_question(chunk)
            if q:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                rows.append(q)
                written += 1
                print(f"[{written}/{n_target}] ({q['language']}) {q['question'][:80]}")

    csv_path = out_path.with_suffix(".csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id", "document_name", "language", "question", "source_preview"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Wrote {written} questions to {out_path} and {csv_path}")

if __name__ == "__main__":
    main()
