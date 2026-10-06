"""
Citation grounding. Instead of trusting the model's summary of retrieved
chunks, this forces every claim to be tagged with which chunk it came from
([1], [2], etc.), then maps those markers back to the EXACT original chunk
text -- so the user can verify each specific claim against real source text,
not just see a list of "related" documents.
"""
import re

CITATION_INSTRUCTION = """
IMPORTANT: After every factual claim, cite which numbered source it came from
using square brackets, e.g. "Scope includes asset management [2]." If a
sentence combines info from multiple sources, cite all of them: "...[1][3]".
Every sentence with a factual claim MUST have at least one citation. Do not
make any claim without a citation -- if you can't cite it, don't say it.
"""


def build_labeled_context(chunks: list[dict]) -> tuple[str, dict]:
    """Returns (context_text_with_labels, label_to_chunk_map)."""
    labeled_parts = []
    label_map = {}
    for i, c in enumerate(chunks, 1):
        label_map[str(i)] = c
        labeled_parts.append(f"[{i}] (from {c['document_name']}):\n{c['content'][:1200]}")
    return "\n\n".join(labeled_parts), label_map


def extract_citations(answer_text: str, label_map: dict) -> list[dict]:
    """Finds every [N] marker in the answer and returns proof objects:
    the exact sentence it was attached to, and the exact source chunk text
    it points to -- so the frontend can show 'claim -> exact proof' pairs."""
    citations = []
    seen = set()

    # split into sentences (simple heuristic, good enough for citation mapping)
    sentences = re.split(r'(?<=[.!?])\s+', answer_text)

    for sentence in sentences:
        markers = re.findall(r'\[(\d+)\]', sentence)
        if not markers:
            continue
        clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
        for m in markers:
            key = (clean_sentence, m)
            if key in seen or m not in label_map:
                continue
            seen.add(key)
            chunk = label_map[m]
            citations.append({
                "claim": clean_sentence,
                "source_label": m,
                "document_name": chunk["document_name"],
                "proof_text": chunk["content"][:600],
                "section_title": chunk.get("section_title"),
            })
    return citations


def strip_citation_markers(answer_text: str) -> str:
    """For display: removes [N] markers from the visible answer text (the
    citations list carries the same info in a more useful, verifiable form)."""
    return re.sub(r'\s*\[\d+\]', '', answer_text)
