"""
Document metadata extraction v2 — fully local.
Scans first N pages (not just page 1) and asks the LLM to extract ALL dates/
reference numbers it finds, each labeled with what it refers to (Royal Decree,
circular number, standard version, etc). Single-value regex kept as a cheap
first pass for the common stamp-box case, but LLM multi-extraction is now
primary since real documents state references in body text, not just headers.
"""
import re, json
from hijri_converter import Hijri
import llm_local

STAMP_NUM_RE  = re.compile(r'(\d{2,6}\s*/\s*\d{1,3})')
STAMP_DATE_RE = re.compile(r'(\d{4})/(\d{2})/(\d{2})')
FILENAME_RE   = re.compile(r'([A-Za-z\-]+)-(\d+)-(\d{4})H', re.I)

TYPE_KEYWORDS = {
    "تعميم": "circular",
    "إرشادات": "guideline",
    "Advance Release Calendar": "schedule",
    "Royal Decree": "royal_decree",
}

def hijri_to_gregorian(h_str: str) -> str | None:
    """Accepts YYYY/MM/DD Hijri."""
    try:
        y, m, d = map(int, h_str.split("/"))
        return Hijri(y, m, d).to_gregorian().isoformat()
    except Exception:
        return None

def _cheap_pass(filename: str, first_page_text: str) -> dict:
    """Fast regex pass -- catches the stamp-box format when present."""
    result = {}
    fm = FILENAME_RE.search(filename)
    if fm:
        result["reference_number"] = fm.group(2)
        result["effective_date_hijri"] = fm.group(3)
    num_m = STAMP_NUM_RE.search(first_page_text)
    if num_m:
        result["reference_number"] = num_m.group(1).replace(" ", "")
    date_m = STAMP_DATE_RE.search(first_page_text)
    if date_m:
        h_str = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"
        result["effective_date_hijri"] = h_str
        greg = hijri_to_gregorian(h_str)
        if greg:
            result["effective_date"] = greg
    for kw, doc_type in TYPE_KEYWORDS.items():
        if kw in first_page_text:
            result["document_type"] = doc_type
            break
    return result

def _llm_pass(combined_text: str) -> dict:
    """Primary extraction: multi-page text, asks for ALL references found, labeled."""
    prompt = f"""/no_think
Read this Saudi regulatory document excerpt (may span multiple pages). Extract ALL
reference numbers and dates you find anywhere in the text, not just the header.
Return ONLY JSON, no markdown:

{{
  "document_type": "circular|guideline|schedule|regulation|royal_decree|standard|other",
  "topic_tags": ["2-4 short keywords describing THIS document's actual subject matter"],
  "references": [
    {{"label": "what this refers to, e.g. 'Royal Decree number' or 'Circular number' or 'Standard version'",
      "value": "the number/id as written",
      "date_hijri": "DD/MM/YYYYH as written, or null if no date attached to this reference",
      "date_gregorian_hint": "your best-effort Gregorian estimate if you can compute it, else null"}}
  ]
}}

If no references are found, return "references": [].

Text:
{combined_text[:6000]}"""
    raw = llm_local.raw_generate(prompt, max_tokens=700)
    raw = raw.strip().strip("```json").strip("```").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        parsed = json.loads(raw)
        return {k: (None if v == "null" else v) for k, v in parsed.items()}
    except json.JSONDecodeError as e:
        print(f"  [metadata] LLM JSON parse failed: {e} | raw: {raw[:200]}")
        return {}

def _normalize_ref_value(v: str) -> str:
    """Strip spaces/punctuation for dedup comparison."""
    return re.sub(r"[\s/\-().]", "", v or "").lower()

def _is_garbage(v: str) -> bool:
    """Drop OCR noise: too short, no digits, or looks like a stray OCR fragment."""
    v = (v or "").strip()
    if len(v) < 3:
        return True
    if not any(ch.isdigit() for ch in v):
        # allow non-numeric standard names like "ECC" only if reasonably long/wordlike
        if len(v) < 5 or not any(ch.isalpha() for ch in v):
            return True
    # reject fragments that are mostly punctuation
    alnum_ratio = sum(ch.isalnum() for ch in v) / max(len(v), 1)
    if alnum_ratio < 0.5:
        return True
    return False

def _clean_hijri_date(raw: str) -> str | None:
    """Normalize whatever the LLM wrote into YYYY/MM/DD, or None if unparseable."""
    if not raw:
        return None
    nums = re.findall(r"\d+", raw)
    if len(nums) != 3:
        return None
    # Heuristic: the 4-digit number is the year, wherever it sits
    year = next((n for n in nums if len(n) == 4), None)
    if not year:
        return None
    rest = [n for n in nums if n != year]
    if len(rest) != 2:
        return None
    day, month = rest  # LLM order is inconsistent; assume day/month as written
    try:
        y_int = int(year)
        if not (1380 <= y_int <= 1460):  # plausible Hijri range for modern SAMA/NCA docs
            return None
        d, m = int(day), int(month)
        if m > 12:  # swapped
            d, m = m, d
        if not (1 <= d <= 30 and 1 <= m <= 12):
            return None
        return f"{year}/{m:02d}/{d:02d}"
    except Exception:
        return None

def _clean_references(refs: list[dict]) -> list[dict]:
    """Dedup near-identical values, drop garbage, recompute Gregorian dates ourselves."""
    seen = {}
    for r in refs:
        value = (r.get("value") or "").strip()
        if _is_garbage(value):
            continue
        key = (_normalize_ref_value(value), r.get("label"))
        clean_hijri = _clean_hijri_date(r.get("date_hijri"))
        greg = hijri_to_gregorian(clean_hijri) if clean_hijri else None
        entry = {
            "label": r.get("label"),
            "value": value,
            "date_hijri": clean_hijri,
            "date_gregorian": greg,  # computed deterministically, not LLM-guessed
        }
        # Keep the entry with more complete info if we see a near-duplicate
        if key not in seen or (entry["date_hijri"] and not seen[key]["date_hijri"]):
            seen[key] = entry
    return list(seen.values())

def get_document_metadata(filename: str, pages: list[dict], max_pages: int = 3) -> dict:
    """pages: list of {'text': ..., 'page': N} dicts from extract_pages()/ocr_pages()."""
    first_page_text = pages[0]["text"] if pages else ""
    cheap = _cheap_pass(filename, first_page_text)

    combined_text = "\n\n".join(p["text"] for p in pages[:max_pages])
    llm_meta = _llm_pass(combined_text)

    cleaned_refs = _clean_references(llm_meta.get("references") or [])

    meta = {
        "document_type": cheap.get("document_type") or llm_meta.get("document_type"),
        "topic_tags": llm_meta.get("topic_tags"),
        "extracted_references": cleaned_refs,
    }
    meta["reference_number"] = cheap.get("reference_number")
    meta["effective_date_hijri"] = cheap.get("effective_date_hijri")
    meta["effective_date"] = cheap.get("effective_date")
    if not meta["reference_number"] and cleaned_refs:
        first_ref = cleaned_refs[0]
        meta["reference_number"] = first_ref.get("value")
        meta["effective_date_hijri"] = first_ref.get("date_hijri")
        meta["effective_date"] = first_ref.get("date_gregorian")
    return meta
