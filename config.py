"""Central configuration for the vectorless RAG chatbot.

Everything points at a local, open-source model served by Ollama
(https://ollama.com). No API keys, no cloud, no embeddings.
"""

import os

# --- Model / server ---------------------------------------------------------
# Ollama exposes an OpenAI-ish REST API on this host by default.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Any open-source chat model you have pulled with `ollama pull <name>`.
# llama3.2 (3B) is small and fast; swap for qwen2.5, mistral, phi3, etc.
MODEL = os.environ.get("VRAG_MODEL", "llama3.2")

# Lower = more deterministic. Retrieval wants 0; answering can stay low.
RETRIEVAL_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.2

# --- Corpus / index ---------------------------------------------------------
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.json")

# How big each chunk (a.k.a. "node") may get, in characters, before we split.
MAX_CHUNK_CHARS = 1200

# How many chunks the LLM-retriever is allowed to select for one question.
MAX_SELECTED_CHUNKS = 4
