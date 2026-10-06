"""Backfill metadata for documents_local rows with missing document_type/reference_number.
Re-reads first page from source PDF, re-runs extraction, UPDATEs the row. No re-embedding."""
import sys, os
sys.path.insert(0, r"C:\Users\kagune\Documents\ZetaLabs\iotav3\backend")

from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
from scraper import extract_pages
from extract_metadata_local import get_document_metadata
from ocr_local import ocr_pages

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

PDF_FOLDERS = [
    Path(r"C:\Users\kagune\Documents\ZetaLabs\iotav3\backend\PDFs"),
    Path(r"C:\Users\kagune\Documents\ZetaLabs\iotav3\backend\downloaded_pdfs"),
]

def find_pdf(doc_name: str) -> Path | None:
    for folder in PDF_FOLDERS:
        for pdf in folder.rglob("*.pdf"):
            if pdf.stem == doc_name:
                return pdf
    return None

def backfill(target_names: list[str] | None = None):
    query = sb.table("documents_local").select("id, document_name")
    if target_names:
        query = query.in_("document_name", target_names)
    else:
        query = query.is_("document_type", "null")
    rows = query.execute().data
    print(f"{len(rows)} docs to backfill.")

    for row in rows:
        doc_name = row["document_name"]
        pdf_path = find_pdf(doc_name)
        if not pdf_path:
            print(f"SKIP (source file not found): {doc_name}")
            continue

        pages = extract_pages(pdf_path, site="backfill")
        if not pages:
            try:
                pages = ocr_pages(pdf_path)
            except Exception as e:
                print(f"SKIP (OCR failed): {doc_name}: {e}")
                continue
        if not pages:
            print(f"SKIP (no text): {doc_name}")
            continue

        meta = get_document_metadata(pdf_path.name, pages)
        sb.table("documents_local").update({
            "document_type": meta.get("document_type"),
            "reference_number": meta.get("reference_number"),
            "effective_date": meta.get("effective_date"),
            "effective_date_hijri": meta.get("effective_date_hijri"),
            "topic_tags": meta.get("topic_tags"),
            "extracted_references": meta.get("extracted_references"),
        }).eq("id", row["id"]).execute()
        print(f"UPDATED: {doc_name} -> {meta}")

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    backfill(targets)
