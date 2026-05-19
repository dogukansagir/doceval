from ingest import ingest_pdf
from gradio_ui import create_demo

state = {}
vectorstore, bm25_retriever = ingest_pdf()
state["vectorstore"] = vectorstore
state["bm25_retriever"] = bm25_retriever

demo = create_demo(state)
demo.launch()
