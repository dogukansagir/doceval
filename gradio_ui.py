import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
import os
from ingest import ingest_pdf, s3_client, rag_client, vectors_config, sparse_vectors_config
from eval import evaluate
import config

def create_demo(state):

    def parse_gradio_history(history):
        langchain_history = []
        for entry in history:
            if entry["role"] == "user":
                langchain_history.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                langchain_history.append(AIMessage(content=entry["content"]))

        return langchain_history
    
    def run_ingest():
        ingest_pdf()
        state["rag_client"] = rag_client
        return "Ingestion complete."

    def run_reset():
        rag_client.delete_collection(collection_name=config.QDRANT_COLLECTION_NAME)
        rag_client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )
        s3_client.delete_object(Bucket=config.S3_BUCKET_NAME, Key="ingested_files.json")
        state["rag_client"] = rag_client
        return "Reset complete."

    def upload_pdf(files):
        if files is None:
            return "No files uploaded."
        
        for file in files:
            filename = os.path.basename(file)
            s3_client.upload_file(file, config.S3_BUCKET_NAME, f"pdfs/{filename}")

        ingest_pdf()
        state["rag_client"] = rag_client

        return f"{len(files)} file(s) uploaded and ingested successfully."

    def chat(message, history):
        rag_client = state["rag_client"]
        langchain_history = parse_gradio_history(history)
        answer, scores, cp_score, source_docs, rewritten_query, _ = evaluate(message, rag_client, chat_history=langchain_history)
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
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="multiple")
            upload_btn = gr.Button("Upload and Ingest")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        with gr.Row():
            ingest_btn = gr.Button("Ingest PDFs from Server")
            reset_btn = gr.Button("Reset Database", variant="stop")
            operation_status = gr.Textbox(label="Status", interactive=False)

        msg.submit(chat, [msg, chatbot], [chatbot, faithfulness, relevancy, correctness, cp_score, sources_box, rewritten_query_box, msg])
        upload_btn.click(upload_pdf, inputs=pdf_upload, outputs=upload_status)
        ingest_btn.click(run_ingest, inputs=None, outputs=operation_status)
        reset_btn.click(run_reset, inputs=None, outputs=operation_status)
    return demo