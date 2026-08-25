"""Streamlit web UI for the local RAG assistant (Week 4, Option B).

Run with:
    streamlit run app.py

Uses the same pipeline as the CLI: retrieval.get_top_chunks() for context
and llm.chat() for the grounded answer. Models are loaded once per server
process and reused across questions.
"""

import streamlit as st

import config
import db
import llm
import retrieval
from main import FALLBACK_ANSWER, build_user_prompt

st.set_page_config(page_title="Local RAG Assistant", page_icon="📚")


@st.cache_resource(show_spinner="Loading local models (first run may download them)...")
def load_models() -> int:
    """Load both Foundry Local models once; return the indexed chunk count."""
    llm.init_embedding_model()
    llm.init_chat_model()
    conn = db.get_connection()
    n_chunks = db.count_chunks(conn)
    conn.close()
    return n_chunks


st.title("📚 Local RAG Assistant")
st.caption(
    f"Fully offline Q&A over your documents — powered by Foundry Local "
    f"({config.CHAT_MODEL_ALIAS} + {config.EMBEDDING_MODEL_ALIAS})."
)

n_chunks = load_models()
if n_chunks == 0:
    st.error("Knowledge base is empty. Run `python ingest.py` first, then reload.")
    st.stop()

st.sidebar.header("Knowledge base")
st.sidebar.write(f"**{n_chunks}** chunks indexed in `{config.DB_PATH.name}`")
st.sidebar.write("To add documents, drop `.txt`/`.md` files into `documents/` "
                 "and re-run `python ingest.py`.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, chunks)

# Replay past turns so the conversation stays visible across reruns.
for question, answer, chunks in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if chunks:
            with st.expander("Retrieved context"):
                for c in chunks:
                    st.markdown(f"**{c['source']}** (score {c['score']:.3f})")
                    st.text(c["content"][:500])

question = st.chat_input("Ask a question about your documents...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            chunks = retrieval.get_top_chunks(question)
        if not chunks:
            answer = FALLBACK_ANSWER
            st.write(answer)
        else:
            stream = llm.chat_stream(
                config.SYSTEM_PROMPT, build_user_prompt(question, chunks)
            )
            answer = st.write_stream(stream)
        if chunks:
            with st.expander("Retrieved context"):
                for c in chunks:
                    st.markdown(f"**{c['source']}** (score {c['score']:.3f})")
                    st.text(c["content"][:500])

    st.session_state.history.append((question, answer, chunks))
