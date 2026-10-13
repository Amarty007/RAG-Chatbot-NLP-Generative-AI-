# PDF RAG Chatbot

A small retrieval-augmented generation (RAG) app for asking questions about a PDF. It extracts text page by page, retrieves relevant passages, and asks Gemini to answer only from those passages with page citations. The answer fallback is `Mujhe document mein iska jawab nahi mila.` when evidence is missing or citations are invalid.

## Architecture

**Offline ingestion:** PDF pages are read with PyMuPDF, blank pages are skipped, and each page is split into 600-token chunks with 100-token overlap using the multilingual E5 tokenizer. The chunks are embedded in batches with the `passage: ` prefix and stored in a local persistent ChromaDB collection with source and one-based page metadata.

**Question answering:** The optional history rewrite creates a standalone search question. The app embeds it with the E5 `query: ` prefix, retrieves the top-k passages, and sends the context and original question to Gemini 3.8 Flash at temperature 0. Rate-limit (429) and temporary high-demand (503) responses are retried up to twice with short delays; if the primary model stays unavailable, the app tries Gemini 3.5 Flash Lite. Retrieved excerpts and similarity scores are available under each answer.

## Setup

Requires Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`. Alternatively, export that variable in your shell. The multilingual embedding model is downloaded the first time ingestion or chat needs it.

Put a PDF in `data/` and build the index:

```bash
python ingest.py data/your-document.pdf
```

By default, indexing adds/replaces that named PDF in `./chroma_db`. To replace the entire collection:

```bash
python ingest.py data/your-document.pdf --replace
```

Start the chat UI:

```bash
streamlit run app.py
```

You can also upload a PDF in the sidebar. Uploading from the UI replaces the current collection so responses only use the newly selected PDF.

## Evaluation

Edit `eval/questions.json` with questions and reference answers from the indexed PDF. `page` uses the PDF's one-based page number:

```json
[
	{
		"question": "What is the main finding?",
		"reference_answer": "A short answer based on the document.",
		"page": 2
	}
]
```

Run the evaluation after indexing the matching PDF:

```bash
python eval/run_eval.py
```

The script prints a rubric-based mean answer score and citation accuracy. These are LLM-graded estimates, not a substitute for reviewing the answers. The default `eval/questions.json` is empty because reference questions depend on your PDF.

## Streamlit Cloud

Push this repository to GitHub, create a Streamlit Community Cloud app using `app.py` as the entry point, and add `GEMINI_API_KEY` under the app's **Settings > Secrets** as TOML:

```toml
GEMINI_API_KEY = "your-key"
```

The app can index a PDF uploaded through the sidebar. Streamlit Cloud's local filesystem is not durable across app restarts, so its `chroma_db` index may need to be rebuilt after a restart. For a persistent production deployment, use durable external storage or a hosted vector database.

## Results

| Metric | Result |
| --- | --- |
| Mean answer score | _Run evaluation_ |
| Citation accuracy | _Run evaluation_ |

## Limitations

- Scanned PDFs need OCR; image-only pages are skipped as empty text.
- PDF tables and complex layouts are not parsed reliably.
- Similarity thresholds and LLM grading are heuristics; inspect sources and evaluation results.
- Streamlit Cloud's local ChromaDB storage may be lost on restart.

## Future Work

- Add hybrid keyword and vector search.
- Rerank retrieved passages with a cross-encoder.
- Add OCR and stronger table extraction for scanned or structured PDFs.