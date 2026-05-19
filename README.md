# DocEval RAG — Retrieval-Augmented Generation with Self-Evaluation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A production-oriented RAG pipeline that ingests PDFs (including PDFs with images), answers questions using a hybrid retrieval strategy, and continuously self-evaluates and retries to improve answer quality before returning a response.

---

## Features

- **Multimodal PDF ingestion** — extracts text blocks and images; images are described via a Vision Language Model (Gemini) before being stored
- **Hybrid retrieval** — combines dense vector search (Chroma + Sentence Transformers) and sparse keyword search (BM25), with configurable weights
- **Cross-encoder reranking** — reranks retrieved chunks using a cross-encoder model for higher precision
- **Query rewriting** — rewrites follow-up questions into standalone queries using conversation history
- **Self-evaluation loop** — scores each answer on Faithfulness, Answer Relevancy, Answer Correctness, and Context Precision, then retries with targeted feedback if any metric falls below its threshold
- **Gradio UI** — a browser-based chat interface with visible evaluation scores, sources, and the rewritten query

---

## Architecture Overview

```
User Question
      │
      ▼
Query Rewriting (DeepSeek)
      │
      ▼
Hybrid Retrieval (BM25 + Chroma)
      │
      ▼
Context Precision Check ──── below threshold? ──► Re-retrieve (adjusted weights) ──┐
      │                                                                              │
      └──────────────────────────────────────────────────────────────────────────────┘
      │  (best context selected)
      ▼
Answer Generation (DeepSeek)
      │
      ▼
Post-Answer Evaluation (Faithfulness / Relevancy / Correctness)
      │
      ├── all PASS? ──► Return Answer
      │
      └── any FAIL? ──► Retry with targeted feedback prompt ──► Re-evaluate ──► Return best Answer
```

---

## Project Structure

```
.
├── app.py          # Gradio UI and entry point
├── config.py       # All configurable parameters and API keys
├── eval.py         # Query rewriting, retrieval orchestration, answer generation, evaluation loop
├── ingest.py       # PDF parsing, VLM image enrichment, chunking, vector store and BM25 storage
├── prompts.py      # All LLM system prompts
├── rag.py          # Hybrid retriever and cross-encoder reranker
├── requirements.txt
├── pdfs/           # Place your PDF files here before running
├── chroma_db/      # Auto-created; persisted vector store
├── bm25_corpus.json        # Auto-created; BM25 text corpus
└── ingested_files.json     # Auto-created; tracks already-ingested PDFs
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/dogukansagir/doceval.git
cd doceval
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Add PDFs

Create a `pdfs/` folder and place your PDF files in the `pdfs/` directory:

```bash
mkdir pdfs
cp your_document.pdf pdfs/
```

### 5. Run the app

```bash
python app.py
```

The Gradio interface will be available at `http://localhost:7860` by default.

---

## Configuration

All parameters are in `config.py`. Key ones are:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Token size of each text chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between consecutive chunks |
| `CROSSENCODER_KIN` | 30 | Chunks passed into the cross-encoder |
| `CROSSENCODER_KOUT` | 5 | Top chunks selected after reranking |
| `BM25_WEIGHT` | 0.4 | Initial weight for BM25 retriever |
| `COSINE_WEIGHT` | 0.6 | Initial weight for vector retriever |
| `RETRY_COUNT` | 3 | Max answer regeneration retries |
| `CONTEXT_PRECISION_RETRY_COUNT` | 3 | Max retrieval retries |
| `FAITHFULNESS_THRESHOLD` | 0.70 | Minimum passing score |
| `ANSWER_RELEVANCY_THRESHOLD` | 0.70 | Minimum passing score |
| `ANSWER_CORRECTNESS_THRESHOLD` | 0.70 | Minimum passing score |
| `CONTEXT_PRECISION_THRESHOLD` | 0.80 | Minimum passing score |

---

## Models Used

| Role | Model | Provider |
|---|---|---|
| Main LLM (answer generation) | `deepseek-v4-flash` | DeepSeek |
| Judge LLM (evaluation) | `deepseek-v4-flash` | DeepSeek |
| Retry Judge LLM | `deepseek-v4-pro` | DeepSeek |
| Image description (VLM) | `gemini-3.1-flash-lite` | Google |
| Text embeddings | `all-MiniLM-L6-v2` | Sentence Transformers |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` | Sentence Transformers |

---

## Evaluation Metrics

Each response is scored by a judge LLM on four metrics:

- **Faithfulness** — Is the answer supported by the retrieved context? Claims not found in the retrieved chunks lower this score.
- **Answer Relevancy** — Does the answer actually address the question asked?
- **Answer Correctness** — Is the answer factually correct? Scored as `Unavailable` when ground truth is not available.
- **Context Precision** — Are the retrieved chunks relevant to the question? Evaluated before answer generation; triggers re-retrieval with adjusted BM25/cosine weights if below threshold.

When an answer fails on any metric, the system retries with a targeted system prompt indicating which metric failed. 
if the scores are still under the threshold after generating an answer `RETRY_COUNT` times, the best-scoring answer across all attempts is returned.

---

## Adding New PDFs at Runtime

You can upload PDFs directly through the Gradio UI using the **Upload PDF** panel. The file will be saved to `pdfs/`, ingested, and added to the existing vector store and BM25 corpus without reprocessing previously ingested files.

---

## Notes

- The Gemini VLM call includes a 4-second sleep between image requests to stay within the free-tier rate limit (15 RPM). Adjust or remove this in `ingest.py` if you are on a higher quota.
- `ingested_files.json` prevents re-ingesting the same PDF on restart. Delete it (along with `chroma_db/` and `bm25_corpus.json`) to do a full re-ingest from scratch.
- `Answer Correctness` cannot be evaluated without an external ground truth, so the judge returns `Unavailable` when the retrieved context is insufficient to verify correctness. This does not count as a FAIL.
- The evaluation uses a two-tier judge strategy: the first evaluation pass uses `deepseek-v4-flash` for speed (~5s). 
  If the answer fails and enters the retry loop, `deepseek-v4-pro` takes over as the judge for higher reliability. 
  This keeps the happy path fast while ensuring retries are evaluated with maximum accuracy.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
