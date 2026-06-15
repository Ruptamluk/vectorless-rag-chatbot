"""Orchestrate vectorless retrieval + grounded generation."""

import config
import llm
import retriever

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context passages. Cite the source filename in square brackets "
    "after each claim, e.g. [handbook.md]. If the context does not contain "
    "the answer, say you don't know based on the available documents. Do not "
    "invent facts."
)


def _format_context(chunks) -> str:
    blocks = []
    for ch in chunks:
        head = f" - {ch['heading']}" if ch["heading"] else ""
        blocks.append(f"[{ch['source']}{head}]\n{ch['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, index: dict) -> dict:
    """Run the full pipeline. Returns answer text plus retrieval trace."""
    selected, reason = retriever.retrieve(question, index)

    if not selected:
        return {
            "answer": "I couldn't find anything relevant in the documents to "
                      "answer that.",
            "sources": [],
            "reason": reason,
        }

    context = _format_context(selected)
    prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\nAnswer (with [source] citations):"
    )
    text = llm.generate(
        prompt, system=ANSWER_SYSTEM, temperature=config.ANSWER_TEMPERATURE
    )
    return {
        "answer": text,
        "sources": sorted({ch["source"] for ch in selected}),
        "selected_ids": [ch["id"] for ch in selected],
        "reason": reason,
    }
