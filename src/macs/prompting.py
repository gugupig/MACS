from __future__ import annotations

DEFAULT_QA_SYSTEM_PROMPT = (
    "Answer the question based on the following text. "
    "Keep your response short and simple. "
    "Do not quote the original text."
)


def build_qa_prompt(question: str, context: str, instruction: str = DEFAULT_QA_SYSTEM_PROMPT) -> str:
    return (
        f"{instruction}\n"
        "question:\n"
        f"{question}\n"
        "context:\n"
        f"{context}\n"
    )
