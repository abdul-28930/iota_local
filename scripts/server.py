"""Minimal API server exposing the local RAG pipeline. Run: python server.py"""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
import retrieve, embedder, reranker

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _check_gpu_free():
    """Warns loudly if GPU memory is already heavily used -- likely a
    leftover/duplicate server.py instance still holding VRAM (this exact
    scenario caused a severe slowdown incident: two Python processes
    competing for one 6GB GPU dropped generation to ~2.6 tokens/sec)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        if used > total * 0.3:
            print(f"⚠️  WARNING: GPU already has {used}MiB/{total}MiB in use before loading models.")
            print("⚠️  This likely means another server.py (or other GPU process) is still running.")
            print("⚠️  Check Task Manager / nvidia-smi and close duplicates before continuing.")
    except Exception:
        pass  # nvidia-smi not available or check failed -- don't block startup on this


@app.on_event("startup")
def preload_models():
    _check_gpu_free()
    print("Preloading embedder + reranker (keeps them warm across requests)...")
    embedder.load()
    reranker.load()
    print("Models loaded. Ready for queries.")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/documents")
def list_documents():
    rows = sb.table("documents_local") \
        .select("id, document_name, document_type, source_type, topic_tags, total_pages") \
        .order("created_at", desc=True) \
        .execute().data
    return {"documents": rows}

@app.get("/api/stats")
def stats():
    docs = sb.table("documents_local").select("id", count="exact").execute()
    chunks = sb.table("sama_nora_chunks_local").select("id", count="exact").execute()
    return {
        "documents": docs.count or 0,
        "chunks": chunks.count or 0,
    }

class FeedbackIn(BaseModel):
    question: str
    answer: str
    rating: int  # 0 or 1

@app.post("/api/feedback")
def submit_feedback(body: FeedbackIn):
    sb.table("feedback_local").insert({
        "question": body.question,
        "answer": body.answer,
        "rating": body.rating,
    }).execute()
    return {"status": "ok"}

@app.post("/api/sessions")
def create_session(user_id: str | None = None):
    row = sb.table("sessions_local").insert({"title": "New chat", "user_id": user_id}).execute().data[0]
    return row

@app.get("/api/sessions")
def list_sessions(user_id: str | None = None):
    query = sb.table("sessions_local").select("*").order("updated_at", desc=True).limit(30)
    if user_id:
        query = query.eq("user_id", user_id)
    rows = query.execute().data
    return {"sessions": rows}

@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    rows = sb.table("messages_local").select("*").eq("session_id", session_id).order("created_at").execute().data
    return {"messages": rows}

def _maybe_set_title(session_id: str, first_query: str):
    """Only titles a session once -- if it's still 'New chat', set it from the first query."""
    row = sb.table("sessions_local").select("title").eq("id", session_id).execute().data
    current_title = row[0].get("title") if row else None
    print(f"[title-check] session={session_id} current_title={current_title!r}")
    if row and current_title == "New chat":
        title = first_query.strip()
        if len(title) > 48:
            title = title[:45].rsplit(" ", 1)[0] + "..."
        sb.table("sessions_local").update({"title": title}).eq("id", session_id).execute()
        print(f"[title-check] updated to: {title!r}")

class QueryIn(BaseModel):
    query: str
    max_tokens: int = 1200
    session_id: str | None = None
    user_id: str | None = None
    deep_mode: bool = False
    agent_mode: bool = False
    answer_style: str = "normal"

STYLE_MAX_TOKENS = {"summarized": 350, "normal": 900, "deep": 2000}

RELEVANCE_THRESHOLD = 0.35  # rerank scores below this are considered too weak to display as sources

@app.post("/api/query-stream")
def query_stream(body: QueryIn):
    from fastapi.responses import StreamingResponse
    import json as _json

    def event_gen():
        full_answer, source_payload = "", []
        generator = (
            retrieve.answer_stream_agentic(body.query, session_id=body.session_id, user_id=body.user_id, max_tokens=STYLE_MAX_TOKENS.get(body.answer_style, 900))
            if body.agent_mode else
            retrieve.answer_stream(body.query, session_id=body.session_id, user_id=body.user_id,
                                    max_tokens=STYLE_MAX_TOKENS.get(body.answer_style, 900), deep_mode=body.deep_mode, answer_style=body.answer_style)
        )
        for event in generator:
            if event["type"] == "done":
                full_answer = event["full_answer"]
                source_payload = event["sources"]
                if body.session_id:
                    sb.table("messages_local").insert({
                        "session_id": body.session_id, "role": "user", "content": body.query,
                    }).execute()
                    sb.table("messages_local").insert({
                        "session_id": body.session_id, "role": "bot", "content": full_answer,
                        "sources": source_payload,
                    }).execute()
                    _maybe_set_title(body.session_id, body.query)
                retrieve.maybe_extract_memory(body.user_id, body.session_id, body.query, full_answer)
            yield f"data: {_json.dumps(event)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/api/query")
def query(body: QueryIn):
    result = retrieve.answer(body.query, session_id=body.session_id, user_id=body.user_id,
                              max_tokens=STYLE_MAX_TOKENS.get(body.answer_style, 900), deep_mode=body.deep_mode, answer_style=body.answer_style)
    sources = [
        {
            "document_name": c["document_name"],
            "score": round(c["rerank_score"], 3),
            "section_title": c.get("section_title"),
            "preview": (c.get("content") or "")[:280],
        }
        for c in result["sources"] if c["rerank_score"] >= RELEVANCE_THRESHOLD
    ]
    if body.session_id:
        sb.table("messages_local").insert({
            "session_id": body.session_id, "role": "user", "content": body.query,
        }).execute()
        sb.table("messages_local").insert({
            "session_id": body.session_id, "role": "bot", "content": result["answer"],
            "sources": sources,
        }).execute()
        # updated_at is kept fresh via a DB trigger (see schema.sql), no manual update needed here
        _maybe_set_title(body.session_id, body.query)
    return {"answer": result["answer"], "sources": sources, "reasoning": result.get("reasoning", []), "citations": result.get("citations", [])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
