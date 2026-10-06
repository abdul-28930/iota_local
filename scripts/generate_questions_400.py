"""
Generates exactly 200 English + 200 Arabic test questions from random chunks
in sama_nora_chunks_local. Separate from generate_questions.py (which samples
mixed-language) -- this guarantees the exact split you asked for.

Usage: python generate_questions_400.py
Output: local_test_questions_en.jsonl/.csv, local_test_questions_ar.jsonl/.csv
"""
import os, json, random, csv
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

import llm_local

TARGET_PER_LANG = 200


def sample_chunks_by_language(lang: str, n: int) -> list[dict]:
    rows = sb.table("sama_nora_chunks_local") \
        .select("id, document_name, content, language, token_count") \
        .eq("language", lang).execute().data
    candidates = [r for r in rows if r.get("token_count", 0) and r["token_count"] >= 60]
    random.shuffle(candidates)
    return candidates[:n * 2]  # oversample -- some generations will fail/be skipped


def make_question(chunk: dict, lang: str) -> dict | None:
    lang_instruction = "Arabic" if lang == "ar" else "English"
    prompt = f"""/no_think
Read this excerpt from a Saudi regulatory/cybersecurity document. Write ONE
specific, answerable question in {lang_instruction} that this excerpt directly
answers. The question should be natural (like a compliance officer would ask),
not generic. Return ONLY the question text, nothing else -- no quotes, no labels.

Excerpt:
{chunk['content'][:1200]}"""
    try:
        q = llm_local.raw_generate(prompt, max_tokens=150).strip().strip('"').strip()
        if len(q) < 10:
            return None
        return {
            "chunk_id": chunk["id"], "document_name": chunk["document_name"],
            "language": lang, "question": q, "source_preview": chunk["content"][:300],
        }
    except Exception as e:
        print(f"  [gen] failed for chunk {chunk['id']}: {e}")
        return None


def generate_for_language(lang: str, target: int):
    print(f"\n=== Generating {target} {lang.upper()} questions ===")
    chunks = sample_chunks_by_language(lang, target)
    print(f"Sampled {len(chunks)} candidate {lang} chunks.")

    rows = []
    jsonl_path = Path(f"local_test_questions_{lang}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if len(rows) >= target:
                break
            q = make_question(chunk, lang)
            if q:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                rows.append(q)
                print(f"[{len(rows)}/{target}] {q['question'][:70]}")

    csv_path = jsonl_path.with_suffix(".csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id", "document_name", "language", "question", "source_preview"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} {lang} questions to {jsonl_path} and {csv_path}")
    if len(rows) < target:
        print(f"WARNING: only got {len(rows)}/{target} -- not enough {lang} chunks with token_count>=60 in the corpus.")


if __name__ == "__main__":
    generate_for_language("en", TARGET_PER_LANG)
    generate_for_language("ar", TARGET_PER_LANG)
    print("\nDone. 4 files written: local_test_questions_en.jsonl/.csv, local_test_questions_ar.jsonl/.csv")
