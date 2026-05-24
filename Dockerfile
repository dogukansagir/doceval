FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding; \
    TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'); \
    SparseTextEmbedding('Qdrant/bm25'); \
    LateInteractionTextEmbedding('answerdotai/answerai-colbert-small-v1')"
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]