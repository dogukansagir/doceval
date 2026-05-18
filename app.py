from langchain_core.messages import AIMessage, HumanMessage
import gradio as gr
from ingest import ingest_pdf
from eval import evaluate

def parse_gradio_history(history):
    langchain_history = []
    for entry in history:
        if entry["role"] == "user":
            langchain_history.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            langchain_history.append(AIMessage(content=entry["content"]))

    return langchain_history

def chat(message, history):
    langchain_history = parse_gradio_history(history)
    answer, scores, cp_score, source_docs, rewritten_query, _ = evaluate(message, vectorstore, bm25_retriever, chat_history=langchain_history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    sources_text = "\n\n---\n\n".join([
    f"📄 {doc.metadata.get('source', 'Unknown')} — Page {doc.metadata.get('page', '?')}\n{doc.page_content[:300]}..." for doc in source_docs])

    return history, str(scores["Faithfulness"]), str(scores["Answer Relevancy"]), str(scores["Answer Correctness"]), str(cp_score), sources_text, rewritten_query, ""

vectorstore, bm25_retriever = ingest_pdf()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about the ingested documents.")
    rewritten_query_box = gr.Textbox(label="Rewritten Query", interactive=False)
    with gr.Accordion("See Sources", open=False):
        sources_box = gr.Textbox(label="Sources", interactive=False, lines=10)
    with gr.Row():
        faithfulness = gr.Textbox(label="Faithfulness")
        relevancy = gr.Textbox(label="Answer Relevancy")
        correctness = gr.Textbox(label="Answer Correctness")
        cp_score = gr.Textbox(label="Context Precision")
    msg.submit(chat, [msg, chatbot], [chatbot, faithfulness, relevancy, correctness, cp_score, sources_box, rewritten_query_box, msg])
demo.launch()
