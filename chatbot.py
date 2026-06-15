"""Command-line chatbot over the vectorless index.

Usage:
    python ingest.py        # build index.json (run once)
    python chatbot.py       # chat in the terminal
"""

import os
import sys

import config
import ingest
import llm
import rag


def main():
    if not llm.health_check():
        print(
            f"[!] Ollama model '{config.MODEL}' is not reachable.\n"
            f"    Start Ollama and run:  ollama pull {config.MODEL}\n"
            f"    Server expected at:    {config.OLLAMA_HOST}\n"
        )
        sys.exit(1)

    if not os.path.exists(config.INDEX_PATH):
        print("[*] No index found, building one (run `python ingest.py` to "
              "rebuild later)...")
        index = ingest.build_index()
        ingest.save_index(index)
    else:
        index = ingest.load_index()

    total = sum(len(d["chunks"]) for d in index["documents"])
    print(f"\nVectorless RAG chatbot | model={config.MODEL} | {total} chunks "
          f"in {len(index['documents'])} docs")
    print("Ask a question. Type 'exit' to quit, 'sources' after an answer to "
          "see why.\n")

    last = None
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "sources" and last:
            print(f"     retrieval reason: {last.get('reason')}")
            print(f"     chunks used:      {last.get('selected_ids')}\n")
            continue

        print("     (retrieving + thinking...)", end="\r")
        last = rag.answer(q, index)
        sources = ", ".join(last["sources"]) or "none"
        print(" " * 40, end="\r")
        print(f"bot > {last['answer']}\n      sources: {sources}\n")


if __name__ == "__main__":
    main()
