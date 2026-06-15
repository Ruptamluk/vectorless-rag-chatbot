"""Build a *vectorless* index.

Instead of embedding chunks into vectors, we build a lightweight tree:

    document -> [ chunk { id, heading, summary, text } ]

The `summary` (one line, written by the LLM at ingest time) is what the
retriever later reasons over. Think of it as a machine-generated table of
contents. No embeddings, no vector store -- just JSON on disk.
"""

import json
import os
import re

import config
import llm

SUMMARY_SYSTEM = (
    "You write a single, dense, factual sentence summarizing what a passage "
    "is about, so a reader can decide whether it answers their question. "
    "No preamble. Just the sentence."
)


def _split_into_chunks(text: str):
    """Split on markdown headings first, then by size. Yields (heading, body)."""
    # Break the doc at markdown headings so chunks align with sections.
    parts = re.split(r"\n(?=#{1,6}\s)", text.strip())
    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        if lines[0].lstrip().startswith("#"):
            heading = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            heading = ""
            body = part.strip()

        # Further split oversized sections on blank lines, respecting the cap.
        if len(body) <= config.MAX_CHUNK_CHARS:
            yield heading, body
            continue
        buf = ""
        for para in body.split("\n\n"):
            if len(buf) + len(para) > config.MAX_CHUNK_CHARS and buf:
                yield heading, buf.strip()
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            yield heading, buf.strip()


def _summarize(heading: str, body: str) -> str:
    snippet = body[:1500]
    prompt = f"Heading: {heading or '(none)'}\n\nPassage:\n{snippet}\n\nOne-sentence summary:"
    try:
        return llm.generate(prompt, system=SUMMARY_SYSTEM, temperature=0.0)
    except llm.LLMError:
        # Fall back to a cheap extractive summary if the model is unavailable.
        return (heading + ": " + body[:120]).strip(": ").replace("\n", " ")


def build_index(documents_dir: str = config.DOCUMENTS_DIR) -> dict:
    index = {"model": config.MODEL, "documents": []}
    files = sorted(
        f for f in os.listdir(documents_dir)
        if f.lower().endswith((".txt", ".md"))
    )
    if not files:
        raise FileNotFoundError(f"No .txt/.md files found in {documents_dir}")

    chunk_counter = 0
    for fname in files:
        path = os.path.join(documents_dir, fname)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()

        chunks = []
        for heading, body in _split_into_chunks(text):
            if not body:
                continue
            chunk_id = f"c{chunk_counter}"
            chunk_counter += 1
            print(f"  summarizing {fname} :: {heading or chunk_id} ...")
            chunks.append({
                "id": chunk_id,
                "heading": heading,
                "summary": _summarize(heading, body),
                "text": body,
            })

        index["documents"].append({"source": fname, "chunks": chunks})
        print(f"+ indexed {fname} ({len(chunks)} chunks)")

    return index


def save_index(index: dict, path: str = config.INDEX_PATH) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)


def load_index(path: str = config.INDEX_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    print(f"Building vectorless index with model '{config.MODEL}'...")
    idx = build_index()
    save_index(idx)
    total = sum(len(d["chunks"]) for d in idx["documents"])
    print(f"\nDone. {total} chunks across {len(idx['documents'])} documents "
          f"-> {config.INDEX_PATH}")
