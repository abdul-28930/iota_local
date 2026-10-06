# IOTA Local Pipeline (BGE-M3 + BGE-Reranker-v2-m3 + Qwen3.5-7B)

Fully local RAG stack, tested on new tables, isolated from your production DB/pipeline.

## Setup
1. `pip install -r requirements.txt --break-system-packages` (or in a venv)
2. Copy `.env.example` -> `.env`, fill in your Supabase service-role key (from the new project: aankxlqdhxyerxkharrb)
3. Run `db/schema.sql` in Supabase SQL Editor (new project)
4. Download Qwen3.5-7B-Instruct GGUF (Q4_K_M) — see comment in `scripts/llm_local.py`
5. Start llama.cpp server: `llama-server -m <path>.gguf -ngl 99 -c 4096 --port 8080`

## Run order (staged, per your plan)
1. `python scripts/ingest.py "C:\...\PDFs" 5`   — ingest 5 test docs first
2. Check Supabase `documents_local` / `sama_nora_chunks_local` tables look right
3. `python scripts/retrieve.py "your test question"` — test full pipeline end to end
4. If retrieval/answers look good, re-run ingest.py with a higher limit (or no limit) on the rest

## Notes
- VRAM budget: embed+rerank run together (~4.4GB), then LLM loads separately (~5GB) — never all 3 at once, fits 6GB.
- Sparse search currently scores in Python against a capped 500-row pull (`retrieve.py`) — fine for a small test corpus. At full scale, move sparse scoring into Postgres (GIN-indexed jsonb) or a dedicated engine (e.g. Typesense) instead — flag this once you're past the test batch.
- Metadata extraction (`extract_metadata.py`) and chunking (`scraper.py`) are reused from your existing backend via `sys.path` — no duplication, edit the originals if you want ingestion logic changes to apply here too.
