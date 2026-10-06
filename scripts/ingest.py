"""
Ingest a SMALL batch of local PDFs into the new *_local tables for testing.
Reuses extraction/chunking/metadata logic from the existing backend — import
your working scraper module by adding its folder to sys.path below.
"""
import sys, os, hashlib
sys.path.insert(0, r"C:\Users\kagune\Documents\ZetaLabs\iotav3\backend")  # adjust if needed

from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
import embedder

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# --- reuse existing extraction + chunking (still from backend); metadata is now fully local ---
from scraper import extract_pages, chunk_page
from extract_metadata_local import get_document_metadata
from ocr_local import ocr_pages
from table_extraction import page_has_table, extract_tables_as_markdown


def _file_hash(pdf_path: Path) -> str:
    """MD5 of file bytes -- catches the same PDF saved under different filenames
    (e.g. a UUID-prefixed copy vs the plain name), which filename-based dedup misses."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_one(pdf_path: Path, source_type: str = "SAMA", pages: list[dict] | None = None):
    import ingest_logger
    timer = ingest_logger.DocTimer(pdf_path)
    doc_name = pdf_path.stem
    file_hash = _file_hash(pdf_path)

    if sb.table("documents_local").select("id").eq("document_name", doc_name).execute().data:
        print(f"SKIP (exists, same name): {doc_name}")
        timer.finish("skip_exists")
        return
    if sb.table("documents_local").select("id, document_name").eq("content_hash", file_hash).execute().data:
        existing = sb.table("documents_local").select("document_name").eq("content_hash", file_hash).execute().data[0]
        print(f"SKIP (duplicate content of '{existing['document_name']}'): {doc_name}")
        timer.finish("skip_duplicate")
        return

    if pages is None:
        pages = extract_pages(pdf_path, site="local-test")
        if not pages:
            print(f"  no text layer, trying OCR: {doc_name}")
            try:
                pages = ocr_pages(pdf_path)
            except Exception as e:
                print(f"SKIP (OCR failed: {e}): {doc_name}")
                timer.finish("fail_ocr")
                return
        if not pages:
            print(f"SKIP (no text after OCR): {doc_name}")
            timer.finish("fail_no_text")
            return
    total_pages = pages[0]["total_pages"]

    meta = get_document_metadata(pdf_path.name, pages)

    doc_row = sb.table("documents_local").insert({
        "document_name": doc_name, "source_type": source_type, "total_pages": total_pages,
        "document_type": meta.get("document_type"), "reference_number": meta.get("reference_number"),
        "effective_date": meta.get("effective_date"), "effective_date_hijri": meta.get("effective_date_hijri"),
        "topic_tags": meta.get("topic_tags"),
        "extracted_references": meta.get("extracted_references"),
        "content_hash": file_hash,
    }).execute().data[0]
    doc_id = doc_row["id"]

    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_page(p["text"], p["page"], doc_name))
        # Feature #3: if this page has a real table, add it as a separate,
        # properly-formatted Markdown chunk -- prevents checkbox/matrix tables
        # (like the classified-data-handling case) from flattening into
        # meaningless run-on text during normal sentence-based chunking.
        if page_has_table(pdf_path, p["page"]):
            for table_md in extract_tables_as_markdown(pdf_path, p["page"]):
                if len(table_md) > 20:
                    all_chunks.append({
                        "document_name": doc_name, "page_start": p["page"], "page_end": p["page"],
                        "section_title": "Table (extracted)", "content": table_md,
                        "token_count": len(table_md) // 4, "language": p.get("language", "en"),
                    })
    if not all_chunks:
        print(f"SKIP (no chunks): {doc_name}")
        timer.finish("fail_no_chunks", metadata_found=bool(meta.get("document_type")))
        return

    texts = [c["content"] for c in all_chunks]
    emb = embedder.embed(texts)

    rows = []
    for c, dense, sparse in zip(all_chunks, emb["dense"], emb["sparse"]):
        rows.append({
            "document_id": doc_id, "document_name": doc_name,
            "page_start": c["page_start"], "page_end": c["page_end"],
            "section_title": c.get("section_title"), "content": c["content"],
            "token_count": c.get("token_count"), "language": c.get("language"),
            "dense_embedding": dense, "sparse_weights": sparse,
        })
    sb.table("sama_nora_chunks_local").insert(rows).execute()
    print(f"OK: {doc_name} -> {len(rows)} chunks")
    total_words = sum(len(c["content"].split()) for c in all_chunks)
    timer.finish("success", chunks=len(rows), words=total_words,
                 embedded=True, metadata_found=bool(meta.get("document_type")))


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\kagune\Documents\ZetaLabs\iotav3\backend\PDFs")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    pdfs = list(folder.rglob("*.pdf"))[:limit]
    print(f"Ingesting {len(pdfs)} PDFs (test batch)...")
    import ingest_logger
    print(f"Detailed log: {ingest_logger.get_log_path()}")

    def extract_only(pdf_path: Path):
        """CPU-bound step (text extraction / OCR) — safe to parallelize."""
        doc_name = pdf_path.stem
        if sb.table("documents_local").select("id").eq("document_name", doc_name).execute().data:
            return pdf_path, None, "exists"
        pages = extract_pages(pdf_path, site="local-test")
        if not pages:
            try:
                pages = ocr_pages(pdf_path)
            except Exception as e:
                return pdf_path, None, f"ocr_failed: {e}"
        if not pages:
            return pdf_path, None, "no_text"
        return pdf_path, pages, "ok"

    # OCR runs in parallel (CPU-bound), but each doc is embedded+inserted into
    # Supabase THE MOMENT its own OCR finishes -- not after the whole batch is
    # done. This means you see real "OK: ... -> N chunks" progress continuously
    # instead of a long silent stretch followed by a final batch of saves.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(extract_only, p): p for p in pdfs}
        for fut in as_completed(futures):
            pdf_path, pages, status = fut.result()
            if status != "ok":
                print(f"SKIP ({status}): {pdf_path.stem}")
                continue
            print(f"EXTRACTED: {pdf_path.stem} ({len(pages)} pages) -- embedding now...")
            try:
                ingest_one(pdf_path, pages=pages)
            except Exception as e:
                print(f"FAIL {pdf_path.name}: {e}")

    embedder.unload()
