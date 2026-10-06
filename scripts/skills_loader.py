"""
Loads skill playbooks conditionally, based on what's actually in the context
for this query -- instead of permanently bloating the system prompt with
instructions that only matter sometimes (table handling, Hijri dates, etc).
"""
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"


def load_relevant_skills(context: str) -> str:
    """Returns concatenated skill text for any skill whose trigger condition
    matches the given context. Empty string if none apply."""
    skills_text = []

    if "|" in context and context.count("|") > 4:  # looks like a Markdown table
        skills_text.append(_read("table_handling.md"))

    if "H\"" in context or "Hijri" in context or any(f"{y}H" in context for y in range(1400, 1460)):
        skills_text.append(_read("hijri_dates.md"))

    return "\n\n".join(s for s in skills_text if s)


def load_skill(name: str) -> str:
    """Explicitly load one named skill (e.g. used by answer_critic.py for
    citation verification, which isn't context-triggered the same way)."""
    return _read(f"{name}.md")


def _read(filename: str) -> str:
    path = SKILLS_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""
