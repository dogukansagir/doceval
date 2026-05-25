# DocEval RAG — Retrieval-Augmented Generation with Self-Evaluation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A production-grade RAG pipeline that ingests PDFs (including PDFs with images), answers questions using a hybrid retrieval strategy with ColBERT reranking, and continuously self-evaluates and retries to improve answer quality before returning a response. Exposes both a Gradio chat UI and a FastAPI REST API.

---

## Features

- **Multimodal PDF ingestion** — extracts text blocks and images; images are described via a Vision Language Model (Gemini) before being stored
- **Hybrid retrieval** — combines dense vector search and sparse BM25 search natively in Qdrant, with configurable weighted RRF fusion
- **ColBERT late interaction reranking** — reranks retrieved chunks server-side in Qdrant Cloud using MaxSim for higher precision
- **Query rewriting** — rewrites follow-up questions into standalone queries using conversation history
- **Self-evaluation loop** — scores each answer on Faithfulness, Answer Relevancy, Answer Correctness, and Context Precision, then retries with targeted feedback if any metric falls below its threshold
- **Persistent storage** — PDFs and ingestion tracking stored in AWS S3, survives container restarts
- **Secrets management** — API keys stored in AWS Secrets Manager, falls back to `.env` for local development
- **Gradio UI** — browser-based chat interface with visible evaluation scores, sources, rewritten query, and database management buttons
- **FastAPI REST API** — `/evaluate`, `/ingest`, `/reset`, and `/health` endpoints for programmatic access

---

## Architecture Overview

```
User Question
      │
      ▼
Query Rewriting (DeepSeek)
      │
      ▼
Hybrid Retrieval (Dense + Sparse BM25 in Qdrant)
      │
      ▼
Weighted RRF Fusion
      │
      ▼
Context Precision Check ──── below threshold? ──► Re-retrieve (adjusted weights) ──┐
      │                                                                            │
      └────────────────────────────────────────────────────────────────────────────┘
      │  (best context selected)
      ▼
ColBERT Late Interaction Reranking (Qdrant Cloud)
      │
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

### Service Layer

```
uvicorn main:app
├── POST /evaluate        ← REST API for programmatic access
├── POST /ingest          ← Trigger ingestion of PDFs from S3
├── POST /reset           ← Wipe Qdrant collection and S3 PDF tracking
├── GET  /health          ← Health check endpoint
├── /ui                   ← Gradio chat interface (mounted)
└── /docs                 ← Auto-generated API docs
```

---

## Project Structure

```
.
├── app.py                # Standalone Gradio entry point (local only)
├── main.py               # FastAPI entry point with Gradio mounted at /ui
├── gradio_ui.py          # Shared Gradio UI definition
├── config.py             # All configurable parameters and API keys
├── eval.py               # Query rewriting, retrieval orchestration, answer generation, evaluation loop
├── ingest.py             # PDF parsing, VLM image enrichment, chunking, Qdrant + S3 storage
├── prompts.py            # All LLM system prompts
├── rag.py                # Hybrid retriever with weighted RRF and ColBERT reranking
├── requirements.txt
├── Dockerfile
└── .dockerignore
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
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cluster_url
```

On AWS, these can optionally be stored in Secrets Manager under a secret named `doceval/secrets` as key/value pairs — the app will automatically prefer Secrets Manager over `.env` when running on AWS.

### 4. AWS S3 Setup (required)

S3 is used for persistent PDF storage and ingestion tracking.

- Create an S3 bucket in your preferred region
- Block all public access
- Create an IAM user with `AmazonS3FullAccess` and configure it locally:
```bash
aws configure
```
- Update `config.py` with your bucket name and region:
```python
AWS_REGION = "your-region"
S3_BUCKET_NAME = "your-bucket-name"
```

### 5. Qdrant Cloud Setup

- Create a Qdrant Cloud cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
- Copy the cluster URL and API key to your `.env`
- The collection is created automatically on first run

### 6. Run locally

**Gradio UI only:**
```bash
python app.py
```
Available at `http://localhost:7860`

**FastAPI + Gradio:**
```bash
uvicorn main:app --reload
```

| Path | Description |
|---|---|
| `http://localhost:8000/ui` | Gradio chat interface |
| `http://localhost:8000/evaluate` | REST API endpoint |
| `http://localhost:8000/docs` | Auto-generated API docs |

---

## Cloud Deployment

The app is fully containerized via the provided `Dockerfile` and deployable on any container hosting platform. For AWS ECS Fargate deployment specifically, refer to the [AWS ECS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started.html).

---

## REST API

### `POST /evaluate`

**Request body:**
```json
{
  "query": "What is the faithfulness metric?",
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "Faithfulness measures whether...",
  "scores": {
    "Faithfulness": 0.95,
    "Answer Relevancy": 0.90,
    "Answer Correctness": 0.85
  },
  "context_precision_score": 0.88,
  "rewritten_query": "What is the faithfulness metric?",
  "sources": [
    {
      "content": "chunk text...",
      "source": "paper.pdf",
      "page": 3
    }
  ],
  "chat_history": [
    {"role": "user", "content": "What is the faithfulness metric?"},
    {"role": "assistant", "content": "Faithfulness measures whether..."}
  ]
}
```

### `POST /ingest`

Triggers ingestion of all new PDFs found in S3 that haven't been ingested yet. Runs as a background task.

### `POST /reset`

Wipes the Qdrant collection and clears `ingested_files.json` from S3. Use before re-ingesting from scratch.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Token size of each text chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between consecutive chunks |
| `CROSSENCODER_KIN` | 30 | Chunks passed into ColBERT reranker |
| `CROSSENCODER_KOUT` | 5 | Top chunks selected after reranking |
| `BM25_WEIGHT` | 1.0 | Initial weight for sparse retriever |
| `COSINE_WEIGHT` | 1.5 | Initial weight for dense retriever |
| `RETRY_COUNT` | 3 | Max answer regeneration retries |
| `CONTEXT_PRECISION_RETRY_COUNT` | 3 | Max retrieval retries |
| `FAITHFULNESS_THRESHOLD` | 0.70 | Minimum passing score |
| `ANSWER_RELEVANCY_THRESHOLD` | 0.70 | Minimum passing score |
| `ANSWER_CORRECTNESS_THRESHOLD` | 0.70 | Minimum passing score |
| `CONTEXT_PRECISION_THRESHOLD` | 0.80 | Minimum passing score |
| `AWS_REGION` | eu-north-1 | AWS region for S3 |
| `S3_BUCKET_NAME` | doceval-pdf-storage | S3 bucket for PDFs and tracking |
| `QDRANT_COLLECTION_NAME` | doceval | Qdrant collection name |

---

## Models Used

| Role | Model | Provider |
|---|---|---|
| Main LLM (answer generation) | `deepseek-v4-flash` | DeepSeek |
| Judge LLM (evaluation) | `deepseek-v4-flash` | DeepSeek |
| Retry Judge LLM | `deepseek-v4-pro` | DeepSeek |
| Image description (VLM) | `gemini-3.1-flash-lite` | Google |
| Dense embeddings | `all-MiniLM-L6-v2` | FastEmbed |
| Sparse embeddings (BM25) | `Qdrant/bm25` | FastEmbed |
| Late interaction reranker | `answerai-colbert-small-v1` | FastEmbed |

---

## Evaluation Metrics

Each response is scored by a judge LLM on four metrics:

- **Faithfulness** — Is the answer supported by the retrieved context? Claims not found in the retrieved chunks lower this score.
- **Answer Relevancy** — Does the answer actually address the question asked?
- **Answer Correctness** — Is the answer factually correct? Scored as `Unavailable` when ground truth cannot be determined.
- **Context Precision** — Are the retrieved chunks relevant to the question? Evaluated before answer generation; triggers re-retrieval with adjusted BM25/cosine weights if below threshold.

When an answer fails on any metric, the system retries with a targeted system prompt indicating which metric failed. The best-scoring answer across all attempts is returned.

The evaluation uses a two-tier judge strategy: the first pass uses `deepseek-v4-flash` for speed. If the answer fails and enters the retry loop, `deepseek-v4-pro` takes over for higher reliability.

---

## Adding New PDFs

**Via Gradio UI:** Use the Upload PDF panel. Files are uploaded to S3 and ingested immediately.

**Via AWS S3 Console:**
1. Open your S3 bucket
2. Navigate to the `pdfs/` folder (create it if it doesn't exist)
3. Click **Upload** and select your PDF files
4. Then click **Ingest PDFs from Server** in the Gradio UI or call `POST /ingest`

**Via AWS CLI:**
```bash
aws s3 cp yourfile.pdf s3://your-bucket-name/pdfs/yourfile.pdf
```
Then click **Ingest PDFs from Server** in the Gradio UI or call `POST /ingest`.

---

## Notes

- The Gemini VLM call includes a 4-second sleep between image requests to stay within the free-tier rate limit (15 RPM). Adjust or remove this in `ingest.py` if you are on a higher quota.
- `Answer Correctness` cannot be evaluated without an external ground truth, so the judge returns `Unavailable` when the retrieved context is insufficient to verify correctness. This does not count as a FAIL.
- Weights in the RRF fusion are relative — `COSINE_WEIGHT=1.5` and `BM25_WEIGHT=1.0` means dense retrieval is weighted 1.5x over sparse, not that they sum to 1.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
