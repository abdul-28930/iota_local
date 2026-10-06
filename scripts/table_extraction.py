"""
Feature #3: Table extraction. Detects table-like regions in PDF pages and
converts them to Markdown table syntax instead of letting them flatten into
run-on prose during normal text extraction -- this was the root cause of the
"secure handling of classified data while traveling" case, where a checkbox
matrix got flattened into meaningless text.

Ingestion-time only -- no query-time reasoning involved, since this runs once
per document, not per question.
"""
from pathlib import Path


def extract_tables_as_markdown(pdf_path: Path, page_num: int) -> list[str]:
    """Returns a list of Markdown-formatted tables found on this page (1-indexed).
    Empty list if no tables detected or pdfplumber can't parse the page."""
    try:
        import pdfplumber
    except ImportError:
        return []

    tables_md = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue  # skip noise: need at least a header + one row
                tables_md.append(_table_to_markdown(table))
    except Exception:
        return []
    return tables_md


def _table_to_markdown(table: list[list]) -> str:
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        row = row + [""] * (len(header) - len(row))  # pad short rows
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


def page_has_table(pdf_path: Path, page_num: int) -> bool:
    """Cheap check before running full extraction -- lets scraper.py decide
    whether to call extract_tables_as_markdown() at all."""
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return False
            return len(pdf.pages[page_num - 1].extract_tables()) > 0
    except Exception:
        return False
