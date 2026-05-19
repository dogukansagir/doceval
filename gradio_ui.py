import gradio as gr
from app import chat, upload_pdf

def create_demo(vectorstore, bm25_retriever):
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
    return demo