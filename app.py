"""Streamlit chat interface for asking questions about an indexed PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st

from ingest import COLLECTION_NAME, DEFAULT_DB_PATH, ingest_pdf, load_embedding_model
from rag import answer

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EXAMPLE_QUESTIONS = (
    "Summarize this PDF",
    "What are the key findings?",
    "Explain the main concept",
)
PAGE_CITATION = re.compile(r"\[Page\s+(\d+)\]")


@st.cache_resource
def cached_embedding_model() -> Any:
    """Load the embedding model once per Streamlit server process."""
    return load_embedding_model()


def indexed_chunk_count() -> int:
    """Return the number of indexed chunks, or zero for an empty database."""
    if not DEFAULT_DB_PATH.exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=str(DEFAULT_DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def render_bubble(text: str, role: str) -> None:
    """Render escaped chat text and styled page citations in a chat bubble."""
    safe_text = html.escape(text)
    safe_text = PAGE_CITATION.sub(
        r'<span class="page-badge">Page \1</span>', safe_text
    )
    safe_text = safe_text.replace("\n", "<br>")
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(
        f'<div class="{bubble_class}">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def render_sources(chunks: list[dict[str, Any]]) -> None:
    """Show matching page numbers, scores, and quoted source text."""
    with st.expander("Sources", expanded=False):
        if not chunks:
            st.caption("No source chunks were retrieved.")
            return
        for chunk in chunks:
            match_percent = max(0.0, min(1.0, float(chunk["score"]))) * 100
            st.markdown(f"**Page {chunk['page']} | match {match_percent:.0f}%**")
            quote = "\n".join(f"> {line}" for line in chunk["text"].splitlines())
            st.markdown(quote)


def queue_example_question(question: str) -> None:
    """Queue an example prompt for the normal chat-answer flow."""
    st.session_state.pending_question = question


def main() -> None:
    """Render the PDF chat application."""
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --accent: #2563eb;
            --accent-soft: #eaf1ff;
            --ink: #17243b;
            --muted: #65748b;
            --line: #dce4ef;
        }
        [data-testid="stAppViewContainer"] { background: #f8faff; }
        [data-testid="stSidebar"] {
            background: #f1f5fb;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px 16px;
        }
        [data-testid="stSidebar"] h2 { color: var(--ink); }
        button[kind="primary"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #fff !important;
        }
        button[kind="primary"]:hover { background: #1d4ed8 !important; }
        div[data-testid="stChatMessage"] {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin: 12px 0;
        }
        div[data-testid="stChatMessage"]:has(.user-bubble) {
            flex-direction: row-reverse;
        }
        div[data-testid="stChatMessage"]:has(.user-bubble)
        [data-testid="stChatMessageContent"] {
            display: flex;
            justify-content: flex-end;
        }
        .user-bubble, .assistant-bubble {
            box-sizing: border-box;
            width: fit-content;
            max-width: min(100%, 760px);
            padding: 12px 16px;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }
        .user-bubble {
            margin-left: auto;
            color: #17366e;
            background: var(--accent-soft);
            border: 1px solid #d4e2ff;
            border-radius: 16px 16px 4px 16px;
        }
        .assistant-bubble {
            color: var(--ink);
            background: #fff;
            border: 1px solid var(--line);
            border-left: 3px solid var(--accent);
            border-radius: 4px 16px 16px 16px;
        }
        .page-badge {
            display: inline-block;
            margin: 0 3px;
            padding: 1px 8px;
            color: #1e40af;
            background: #eaf1ff;
            border: 1px solid #c9d9ff;
            border-radius: 999px;
            font-size: 0.78em;
            font-weight: 600;
            white-space: nowrap;
        }
        @media (max-width: 700px) {
            .user-bubble, .assistant-bubble { max-width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("PDF RAG Chatbot")
    st.caption("Ask questions about your document. Answers include page citations.")

    with st.sidebar:
        st.header("Document")
        chunk_count = indexed_chunk_count()
        st.metric("Indexed chunks", chunk_count)
        top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=10, value=5)
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
        index_clicked = st.button(
            "Index PDF",
            type="primary",
            use_container_width=True,
            disabled=uploaded_pdf is None,
        )
        if index_clicked and uploaded_pdf is not None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = Path(uploaded_pdf.name).name
            pdf_path = DATA_DIR / safe_name
            pdf_path.write_bytes(uploaded_pdf.getvalue())
            try:
                with st.spinner("Extracting pages and building the search index..."):
                    count = ingest_pdf(
                        pdf_path,
                        embedding_model=cached_embedding_model(),
                        replace_collection=True,
                    )
                st.session_state.messages = []
                st.success(f"Indexed {count} chunks from {safe_name}.")
                st.rerun()
            except Exception as error:
                st.error(f"Could not index this PDF: {error}")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("pending_question", None)
            st.rerun()
        if chunk_count == 0:
            st.info("Upload and index a text-based PDF to start chatting.")

    example_columns = st.columns(3)
    for column, example in zip(example_columns, EXAMPLE_QUESTIONS):
        column.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
            disabled=chunk_count == 0,
            on_click=queue_example_question,
            args=(example,),
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_bubble(message["content"], message["role"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    typed_question = st.chat_input("Ask a question about your PDF")
    example_question = st.session_state.pop("pending_question", None)
    question = typed_question or example_question
    if question:
        prior_history = st.session_state.messages.copy()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            render_bubble(question, "user")
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document..."):
                    response, sources = answer(
                        question,
                        history=prior_history,
                        k=top_k,
                        embedding_model=cached_embedding_model(),
                    )
                render_bubble(response, "assistant")
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "sources": sources}
                )
            except Exception as error:
                message = str(error)
                if "busy or rate-limited" in message:
                    message = (
                        "Gemini is temporarily busy after several retries. "
                        "Please try again shortly."
                    )
                else:
                    message = (
                        f"I couldn't complete that request: {message}. "
                        "Check your API key and document index, then try again."
                    )
                render_bubble(message, "assistant")
                st.session_state.messages.append(
                    {"role": "assistant", "content": message, "sources": []}
                )


if __name__ == "__main__":
    main()
