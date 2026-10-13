"""Extract, chunk, embed, and store PDF pages in a persistent ChromaDB."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "pdf_chunks"
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "chroma_db"
CHUNK_TOKENS = 600
CHUNK_OVERLAP = 100
BATCH_SIZE = 32


def load_embedding_model() -> SentenceTransformer:
    """Load multilingual E5.

    Time: O(M) to read M model parameters from cache or disk.
    Space: O(M) for the model weights held in memory.
    """
    return SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, tokenizer: Any) -> list[str]:
    """Split text into overlapping chunks.

    For N input tokens and fixed chunk/overlap sizes, time is O(N) and
    space is O(N), including the returned chunks.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start : start + CHUNK_TOKENS]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_TOKENS >= len(token_ids):
            break
    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    *,
    embedding_model: SentenceTransformer | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    replace_collection: bool = False,
) -> int:
    """Index a PDF and return the number of stored text chunks.

    Let N be extracted tokens, C chunks, D embedding dimensions, B batch
    size, and M model parameters. Time is O(N + C * E(600) + V(C)), where
    E is the fixed-model cost to embed one 600-token chunk and V is the
    Chroma index insertion cost. Space is O(N + B * D + M) in RAM; stored
    vectors use O(C * D) disk space. PDF decoding and disk I/O are excluded.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    model = embedding_model or load_embedding_model()
    records: list[dict[str, Any]] = []
    with fitz.open(path) as pdf:
        page_count = len(pdf)
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text").strip()
            if not page_text:
                print(f"Skipping empty page {page_index + 1}/{page_count}")
                continue
            for chunk_index, chunk in enumerate(chunk_text(page_text, model.tokenizer)):
                record_id = hashlib.sha256(
                    f"{path.name}:{page_index + 1}:{chunk_index}:{chunk}".encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "id": record_id,
                        "text": chunk,
                        "metadata": {
                            "page": page_index + 1,
                            "source": path.name,
                        },
                    }
                )
            print(f"Processed page {page_index + 1}/{page_count}")

    if not records:
        raise ValueError("No extractable text was found in the PDF.")

    client = chromadb.PersistentClient(path=str(db_path))
    if replace_collection:
        collection_names = {
            item if isinstance(item, str) else item.name
            for item in client.list_collections()
        }
        if COLLECTION_NAME in collection_names:
            client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # Re-ingesting the same named PDF updates it instead of creating duplicates.
    collection.delete(where={"source": path.name})

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        embeddings = model.encode(
            [f"passage: {record['text']}" for record in batch],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
            embeddings=embeddings.tolist(),
        )
        print(f"Stored {min(batch_start + len(batch), len(records))}/{len(records)} chunks")

    print(f"Done: indexed {len(records)} chunks from {path.name}.")
    return len(records)


def main() -> None:
    """Parse command-line arguments and index the selected PDF."""
    parser = argparse.ArgumentParser(description="Index a PDF for RAG chat.")
    parser.add_argument("pdf_path", help="Path to the PDF file to index")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear the existing collection before indexing this PDF",
    )
    args = parser.parse_args()
    ingest_pdf(args.pdf_path, replace_collection=args.replace)


if __name__ == "__main__":
    main()
