from langchain_core.messages import AIMessage, HumanMessage
from ingest import ingest_pdf
from eval import evaluate
from gradio_ui import create_demo
import os
import shutil

def parse_gradio_history(history):
    langchain_history = []
    for entry in history:
        if entry["role"] == "user":
            langchain_history.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            langchain_history.append(AIMessage(content=entry["content"]))

    return langchain_history

def upload_pdf(file):
    if file is None:
        return "No file uploaded."
    
    filename = os.path.basename(file)
    save_path = f"./pdfs/{filename}"
    shutil.copy(file, save_path)

    # Ingest the new PDF and update the vectorstore and bm25_retriever
    global vectorstore, bm25_retriever
    vectorstore, bm25_retriever = ingest_pdf()
    
    return f"File '{filename}' uploaded and ingested successfully."

def chat(message, history):
    langchain_history = parse_gradio_history(history)
    answer, scores, cp_score, source_docs, rewritten_query, _ = evaluate(message, vectorstore, bm25_retriever, chat_history=langchain_history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    if all(v == "Unavailable" for v in scores.values()):
        sources_text = "No relevant sources found."
    else:
        sources_text = "\n\n---\n\n".join([
            f"📄 {doc.metadata.get('source', 'Unknown')} — Page {doc.metadata.get('page', '?')}\n{doc.page_content[:300]}..."
            for doc in source_docs
        ])

    return history, str(scores["Faithfulness"]), str(scores["Answer Relevancy"]), str(scores["Answer Correctness"]), str(cp_score), sources_text, rewritten_query, ""

vectorstore, bm25_retriever = ingest_pdf()
demo = create_demo(vectorstore, bm25_retriever)
demo.launch()
