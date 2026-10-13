"""Retrieval-augmented question answering for the indexed PDF collection."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ingest import COLLECTION_NAME, DEFAULT_DB_PATH, load_embedding_model
from prompts import GRADER_PROMPT, REWRITE_PROMPT, SYSTEM_PROMPT, USER_TEMPLATE

load_dotenv()
NOT_FOUND = "Mujhe document mein iska jawab nahi mila."
MODEL_NAME = "gemini-3.8-flash"
FALLBACK_MODEL_NAME = "gemini-3.5-flash-lite"


def _api_key() -> str:
    """Read the API key from the environment or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        key = st.secrets.get("GEMINI_API_KEY")
    except (ImportError, RuntimeError, KeyError, FileNotFoundError):
        key = None
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY in your environment or Streamlit secrets."
        )
    return str(key)


def generate(system: str, user: str) -> str:
    """Generate text with Gemini, retrying brief rate-limit or outage errors.

    Time is dominated by remote model calls (up to 3 attempts per model).
    Client-side space is O(input + output text), excluding SDK internals.
    """
    client = genai.Client(api_key=_api_key())
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.0,
        max_output_tokens=1024,
    )
    models = (MODEL_NAME, FALLBACK_MODEL_NAME)
    for model_index, model_name in enumerate(models):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user,
                    config=config,
                )
                return (response.text or "").strip()
            except Exception as error:
                status_code = getattr(error, "code", None) or getattr(
                    error, "status_code", None
                )
                try:
                    status_code = int(status_code)
                except (TypeError, ValueError):
                    status_code = None
                if status_code not in {429, 503}:
                    raise
                if attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                if model_index == len(models) - 1:
                    raise RuntimeError(
                        "Gemini is still busy or rate-limited after retries on "
                        "both models. Please try again shortly."
                    ) from error
    return ""


def retrieve(
    question: str,
    k: int = 5,
    *,
    embedding_model: SentenceTransformer | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return the top-k chunks with page metadata and cosine similarity.

    For Q question tokens, C indexed chunks, D embedding dimensions, and k
    results, time is E(Q) plus the vector index's approximate search cost
    (often near O(log C + k), but index-dependent). Space is O(D + k * S),
    where S is the returned chunk size; model memory is excluded.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    client = chromadb.PersistentClient(path=str(db_path))
    collection_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if COLLECTION_NAME not in collection_names:
        return []
    collection = client.get_collection(COLLECTION_NAME)
    if collection.count() == 0:
        return []

    model = embedding_model or load_embedding_model()
    query_embedding = model.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": text,
                "page": int(metadata["page"]),
                "source": metadata.get("source", ""),
                "score": 1.0 - float(distance),
            }
        )
    return chunks


def build_context(chunks: Sequence[dict[str, Any]]) -> str:
    """Format retrieved chunks with page labels.

    For K chunks and L total text characters, time is O(K + L) and the
    returned context uses O(L) space.
    """
    return "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}" for chunk in chunks
    )


def rewrite_query(history: Sequence[dict[str, str]], question: str) -> str:
    """Rewrite a follow-up question so it can be searched on its own.

    Prompt preparation is O(H + Q) time and space for H history and Q
    question text; total time is dominated by the remote generation call.
    """
    if not history:
        return question
    history_text = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in history[-8:]
    )
    prompt = REWRITE_PROMPT.format(history=history_text, question=question)
    rewritten = generate("Rewrite the question as instructed.", prompt)
    return rewritten or question


def answer(
    question: str,
    history: Sequence[dict[str, str]] | None = None,
    k: int = 5,
    *,
    embedding_model: SentenceTransformer | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[str, list[dict[str, Any]]]:
    """Answer a question and return the answer with its retrieved sources.

    Cost is retrieval plus prompt construction O(L) for L context characters
    and a remote generation call. Working space is O(L + k * S), including
    the retrieved chunks and formatted context.
    """
    search_question = rewrite_query(history or [], question)
    chunks = retrieve(
        search_question,
        k=k,
        embedding_model=embedding_model,
        db_path=db_path,
    )
    if not chunks:
        return NOT_FOUND, chunks

    minimum_similarity = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
    if max(chunk["score"] for chunk in chunks) < minimum_similarity:
        return NOT_FOUND, chunks

    user_prompt = USER_TEMPLATE.format(context=build_context(chunks), question=question)
    response = generate(SYSTEM_PROMPT, user_prompt)
    cited_pages = {int(page) for page in re.findall(r"\[Page\s+(\d+)\]", response)}
    available_pages = {chunk["page"] for chunk in chunks}
    if not response or not cited_pages or not cited_pages.issubset(available_pages):
        return NOT_FOUND, chunks
    return response, chunks


def grade_answer(
    question: str,
    reference_answer: str,
    model_answer: str,
    reference_page: int,
) -> dict[str, Any]:
    """Ask Gemini to score answer correctness and page citation accuracy."""
    user_prompt = json.dumps(
        {
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": model_answer,
            "reference_page": reference_page,
        },
        ensure_ascii=False,
    )
    raw_result = generate(GRADER_PROMPT, user_prompt)
    match = re.search(r"\{.*\}", raw_result, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Grader did not return valid JSON: {raw_result}")
    result = json.loads(match.group(0))
    result["score"] = float(result["score"])
    citation_result = result["citation_correct"]
    result["citation_correct"] = (
        citation_result if isinstance(citation_result, bool)
        else str(citation_result).lower() == "yes"
    )
    return result
