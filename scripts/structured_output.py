"""
Shared harness for parsing model output. Used by agent_loop.py (SEARCH/CHECK/
FINAL lines), citation_grounding.py (citation markers), and
extract_metadata_local.py (JSON) -- previously each had its own regex/JSON
parsing logic duplicated with slightly different fallback behavior. This
centralizes that so fixes/improvements apply everywhere at once.
"""
import re, json


def parse_json_block(raw_text: str, fallback: dict | None = None) -> dict:
    """Extracts and parses the first {...} block found in raw_text. Falls back
    to `fallback` (default {}) on any failure, never raises."""
    fallback = fallback if fallback is not None else {}
    text = raw_text.strip().strip("```json").strip("```").strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def parse_action_line(raw_text: str, valid_actions: list[str]) -> tuple[str | None, str | None]:
    """Extracts an 'ACTION: argument' style line (e.g. agent_loop's
    SEARCH:/CHECK:/FINAL:). Returns (action, argument) or (None, None) if no
    line matched any of valid_actions."""
    first_line = raw_text.strip().split("\n")[0].strip()
    pattern = "|".join(valid_actions)
    match = re.match(rf"({pattern}):\s*(.+)", first_line, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, None
    return match.group(1).upper(), match.group(2).strip()


def parse_verdict(text: str, pass_word: str = "PASS", fail_word: str = "FAIL") -> bool:
    """Extracts a PASS/FAIL-style verdict from free text. Defaults to True
    (pass) only if pass_word appears and fail_word does not -- conservative
    in favor of not blocking on ambiguous output."""
    upper = text.upper()
    if f"VERDICT: {pass_word}" in upper:
        return True
    if f"VERDICT: {fail_word}" in upper:
        return False
    return pass_word in upper and fail_word not in upper
