"""Vectorless retrieval: the LLM *is* the retriever.

Traditional RAG embeds the query and every chunk, then ranks by cosine
similarity. Here we instead show the model the catalogue of chunk summaries
(the machine-written table of contents) and ask it to *reason* about which
chunks could answer the question -- returning their ids as JSON.

This is slower per query than a vector lookup, but: no embedding model, no
vector DB, retrieval decisions are explainable, and it handles relational /
multi-hop questions that pure similarity search often misses.
"""

import config
import llm

RETRIEVAL_SYSTEM = (
    "You are a retrieval engine. Given a user question and a catalogue of "
    "document chunks (each with an id and a one-line summary), choose the "
    "chunks whose content is needed to answer the question.\n"
    "First, think about every distinct topic, entity, or rule the question "
    "touches -- including RELATED concepts that may be named differently in the "
    "documents (e.g. 'customer records' relates to 'confidential data'; "
    "'working from home' relates to 'remote work'). A question can require "
    "chunks from MULTIPLE documents.\n"
    "Then select the chunks covering those topics. Include a chunk if it is "
    "plausibly needed; omit clearly unrelated ones. If nothing fits, return an "
    'empty list. Respond as JSON: {"chunk_ids": ["c1", "c4"], "reason": "..."}'
)


def _format_catalogue(index: dict) -> str:
    lines = []
    for doc in index["documents"]:
        lines.append(f"# Document: {doc['source']}")
        for ch in doc["chunks"]:
            head = f" [{ch['heading']}]" if ch["heading"] else ""
            lines.append(f"  - {ch['id']}{head}: {ch['summary']}")
    return "\n".join(lines)


def _all_chunks(index: dict) -> dict:
    return {ch["id"]: {**ch, "source": doc["source"]}
            for doc in index["documents"] for ch in doc["chunks"]}


def retrieve(question: str, index: dict):
    """Return (selected_chunks, reason). Each chunk includes its full text."""
    catalogue = _format_catalogue(index)
    prompt = (
        f"Question: {question}\n\n"
        f"Catalogue of available chunks:\n{catalogue}\n\n"
        f"Select up to {config.MAX_SELECTED_CHUNKS} chunk ids that are needed "
        "to answer the question."
    )
    result = llm.generate_json(
        prompt, system=RETRIEVAL_SYSTEM, temperature=config.RETRIEVAL_TEMPERATURE
    )

    chunks_by_id = _all_chunks(index)
    chosen_ids = result.get("chunk_ids", [])[: config.MAX_SELECTED_CHUNKS]
    selected = [chunks_by_id[cid] for cid in chosen_ids if cid in chunks_by_id]
    return selected, result.get("reason", "")
