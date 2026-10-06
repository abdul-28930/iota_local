"""
Feature #2: Self-critique. After generating an answer, checks whether it
actually addresses the question using real retrieved content, or is vague/
circular (e.g. just restating the question back, as we saw with the
"classified data while traveling" case). Returns the model's genuine reasoning
for its verdict.
"""
import llm_thinking
import structured_output


def check_answer_quality(query: str, answer: str, context: str) -> tuple[bool, str]:
    """Returns (passed, reasoning)."""
    prompt = f"""Does this ANSWER genuinely address the QUESTION using real information
from CONTEXT, or is it vague, circular, or just restating the question without
real content? Think it through, then give your verdict.

QUESTION: {query}
CONTEXT (excerpt): {context[:1000]}
ANSWER: {answer}

End your response with exactly one line: "VERDICT: PASS" or "VERDICT: FAIL"."""

    content, reasoning = llm_thinking.generate_with_thinking(prompt, max_tokens=700)
    passed = structured_output.parse_verdict(reasoning + " " + content)
    verdict_reasoning = reasoning.strip() or content.strip() or "No detailed reasoning returned."
    return passed, verdict_reasoning
