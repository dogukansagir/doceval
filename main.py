from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from eval import evaluate
from ingest import ingest_pdf
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
    vectorstore, bm25_retriever = ingest_pdf()
    state["vectorstore"] = vectorstore
    state["bm25_retriever"] = bm25_retriever
    print("Retrievers are ready")
    yield
    state.clear()

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

class Response(BaseModel):
    answer: str
    scores: dict
    context_precision_score: float | str
    rewritten_query: str
    sources: list[dict]
    chat_history: list[dict]

app = FastAPI(lifespan=lifespan)
demo = create_demo(state)
app = gr.mount_gradio_app(app, demo, path="/ui")

@app.post("/evaluate")
def run_evaluation(request: QueryRequest) -> Response:
    vectorstore = state["vectorstore"]
    bm25_retriever = state["bm25_retriever"]
    chat_history = dict_to_langchain(request.chat_history)

    answer, scores, context_precision, retrieved_docs, rewritten_query, chat_history = evaluate(
    query=request.query,
    vectorstore=vectorstore,
    bm25_retriever=bm25_retriever,
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

