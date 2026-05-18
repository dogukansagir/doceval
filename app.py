from langchain_core.messages import AIMessage, HumanMessage
import gradio as gr
from ingest import ingest_pdf
from eval import evaluate
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

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about the ingested documents.")
    rewritten_query_box = gr.Textbox(label="Rewritten Query", interactive=False)
    with gr.Accordion("See Sources", open=False):
        sources_box = gr.Textbox(label="Sources", interactive=False, lines=10, max_lines=10)
    with gr.Row():
        faithfulness = gr.Textbox(label="Faithfulness")
        relevancy = gr.Textbox(label="Answer Relevancy")
        correctness = gr.Textbox(label="Answer Correctness")
        cp_score = gr.Textbox(label="Context Precision")
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload and Ingest")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    msg.submit(chat, [msg, chatbot], [chatbot, faithfulness, relevancy, correctness, cp_score, sources_box, rewritten_query_box, msg])
    upload_btn.click(upload_pdf, inputs=pdf_upload, outputs=upload_status)
demo.launch()
