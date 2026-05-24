from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from eval import evaluate
from ingest import rag_client, ingest_pdf, vectors_config, sparse_vectors_config, s3_client
import config
from gradio_ui import create_demo
import gradio as gr

state = {}

def dict_to_langchain(chat_history):
    langchain_history = []
    for message in chat_history:
        if message["role"] == "user":
            langchain_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_history.append(AIMessage(content=message["content"]))
        else:
            raise ValueError(f"Unknown role: {message['role']}")
    return langchain_history

def langchain_to_dict(chat_history):
    dict_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            dict_history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            dict_history.append({"role": "assistant", "content": message.content})
    return dict_history

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag_client"] = rag_client
    print("RAG client is ready")
    yield
    state.clear()

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

class Response(BaseModel):
    answer: str
    scores: dict
    context_precision_score: float
    rewritten_query: str
    sources: list[dict]
    chat_history: list[dict]

app = FastAPI(lifespan=lifespan)
demo = create_demo(state)
app = gr.mount_gradio_app(app, demo, path="/ui")

@app.post("/evaluate")
def run_evaluation(request: QueryRequest) -> Response:
    rag_client = state["rag_client"]
    chat_history = dict_to_langchain(request.chat_history)

    answer, scores, context_precision, retrieved_docs, rewritten_query, chat_history = evaluate(
    query=request.query,
    rag_client=rag_client,
    chat_history=chat_history
    )

    chat_history = langchain_to_dict(chat_history)

    sources = [
    {"content": doc.page_content, "source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
    for doc in retrieved_docs
    ]

    return Response(
        answer=answer,
        scores=scores,
        context_precision_score=context_precision,
        rewritten_query=rewritten_query,
        sources=sources,
        chat_history=chat_history  
    )

@app.post("/ingest")
def run_ingest():
    rag_client = ingest_pdf()
    state["rag_client"] = rag_client
    return {"status": "ingestion complete"}

@app.post("/reset")
def reset():
    rag_client.delete_collection(collection_name=config.QDRANT_COLLECTION_NAME)
    rag_client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config
    )
    s3_client.delete_object(Bucket=config.S3_BUCKET_NAME, Key="ingested_files.json")
    return {"status": "reset complete"}
