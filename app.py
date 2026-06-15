"""Streamlit chat UI for the vectorless RAG bot.

Run:  streamlit run app.py
"""

import os

import streamlit as st

import config
import ingest
import llm
import rag

st.set_page_config(page_title="Vectorless RAG Chatbot", page_icon="📄")
st.title("📄 Vectorless RAG Chatbot")
st.caption(f"Open-source model `{config.MODEL}` via Ollama — no embeddings, "
           "no vector database.")


@st.cache_resource(show_spinner="Loading / building index...")
def get_index():
    if not os.path.exists(config.INDEX_PATH):
        idx = ingest.build_index()
        ingest.save_index(idx)
        return idx
    return ingest.load_index()


if not llm.health_check():
    st.error(
        f"Ollama model `{config.MODEL}` is not reachable at "
        f"`{config.OLLAMA_HOST}`.\n\nStart Ollama and run "
        f"`ollama pull {config.MODEL}`."
    )
    st.stop()

index = get_index()

with st.sidebar:
    st.subheader("Corpus")
    for doc in index["documents"]:
        st.write(f"- **{doc['source']}** ({len(doc['chunks'])} chunks)")
    st.divider()
    st.write("Retrieval is done by the LLM reasoning over chunk summaries — "
             "expand **Retrieval trace** under any answer to see which chunks "
             "it picked and why.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Retrieval trace"):
                st.write(f"**Reason:** {msg['trace']['reason']}")
                st.write(f"**Chunks used:** {msg['trace'].get('selected_ids')}")
                st.write(f"**Sources:** {', '.join(msg['trace']['sources'])}")

if prompt := st.chat_input("Ask about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + answering..."):
            result = rag.answer(prompt, index)
        st.markdown(result["answer"])
        with st.expander("Retrieval trace"):
            st.write(f"**Reason:** {result['reason']}")
            st.write(f"**Chunks used:** {result.get('selected_ids')}")
            st.write(f"**Sources:** {', '.join(result['sources']) or 'none'}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "trace": result,
    })
