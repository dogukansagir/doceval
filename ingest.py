import pymupdf as fitz
from google import genai
from google.genai import types
import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import os
import json
import time

client = genai.Client(api_key=config.GEMINI_API_KEY)
rag_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])
embeddings = HuggingFaceEmbeddings(model_name=config.SENTENCETRANSFORMER_MODEL)

def doc_to_dict(document):
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }

def dict_to_doc(dictionary):
    return Document(
        page_content=dictionary["page_content"],
        metadata=dictionary["metadata"]
    )

def extract_blocks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    output = []
    flags = fitz.TEXT_PRESERVE_IMAGES
    for page in doc:
        blocks = page.get_text("blocks", flags=flags)
        for block in blocks:
            if block[6] == 0:  # Text block
                output.append({"page": page.number, "blocks": {"type": "text", "content": block[4]}})

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)

            width = base_image["width"]
            height = base_image["height"]
            area = width * height
            aspect_ratio = max(width, height) / min(width, height)

            if area < 10000 or aspect_ratio > 5:
                continue

            output.append({"page": page.number, "blocks": {"type": "image", "content": base_image["image"], "ext": base_image["ext"]}})

    return output

def enrich_blocks(blocks):
    for block in blocks:
        if block["blocks"]["type"] == "image":
            block["blocks"]["content"] = call_vlm_model(block["blocks"])
            time.sleep(4) # to avoid rate limits (15 requests per minute)
            block["blocks"]["type"] = "text"
    return blocks

def call_vlm_model(block):
    response = client.models.generate_content(
    model=config.GEMINI_MODEL,
    contents=[
      types.Part.from_bytes(
        data=block["content"],
        mime_type=f'image/{block["ext"]}',
      ),
      'Extract all data, values, labels and relationships visible in the image. Analyze the data and give me a string about your analysis. you will create strings that are going inside a RAG system. do not omit any information. describe the type and structure of the visual content (table, chart, diagram, figure) before describing its contents.'
        ],
    )
    return response.text

def chunk_and_store(blocks, pdf_name):
    all_documents = []
    
    for page_num in set([block["page"] for block in blocks]):
        page_text = ""
        for block in blocks:
            if block["page"] == page_num:
                page_text += block["blocks"]["content"]

        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            all_documents.append(Document(
                page_content=chunk,
                metadata={"source": pdf_name, "page": page_num + 1}
            ))

    print(f"Total Chunks: {len(all_documents)}")
    
    vectorstore = QdrantVectorStore(client=rag_client, collection_name="doceval", embedding=embeddings)
    vectorstore.add_documents(all_documents)

    bm25_file_name = "bm25_corpus.json"
    if os.path.exists(bm25_file_name):
        with open(bm25_file_name, "r") as f:
            bm25_corpus = json.load(f)
            bm25_corpus.extend([doc_to_dict(doc) for doc in all_documents])
        with open(bm25_file_name, "w") as f:
            json.dump(bm25_corpus, f)
    else:
        bm25_corpus = [doc_to_dict(doc) for doc in all_documents]
        with open(bm25_file_name, "w") as f:
            json.dump(bm25_corpus, f)
    
    bm25_corpus = [dict_to_doc(doc) for doc in bm25_corpus]
    bm25_retriever = BM25Retriever.from_documents(bm25_corpus, k=config.CROSSENCODER_KIN)

    return vectorstore, bm25_retriever

def ingest_pdf(pdf_folder = "./pdfs"):
    vectorstore, bm25_retriever = None, None

    if os.path.exists("ingested_files.json"):
        with open("ingested_files.json", "r") as f:
            ingested_files = json.load(f)
    else:
        ingested_files = []
        
    for pdf in os.listdir(pdf_folder):
        if pdf.endswith(".pdf") and pdf not in ingested_files:
            pdf_path = os.path.join(pdf_folder, pdf)
            blocks = extract_blocks_from_pdf(pdf_path)
            enriched_blocks = enrich_blocks(blocks)
            vectorstore, bm25_retriever = chunk_and_store(enriched_blocks,pdf)
            ingested_files.append(pdf)
            with open("ingested_files.json", "w") as f:
                json.dump(ingested_files, f)
    
    if vectorstore == None and bm25_retriever == None:
        vectorstore = QdrantVectorStore(client=rag_client, collection_name="doceval", embedding=embeddings)
        with open("bm25_corpus.json", "r") as f:
            bm25_corpus = json.load(f)
        
        bm25_corpus = [dict_to_doc(doc) for doc in bm25_corpus]
        bm25_retriever = BM25Retriever.from_documents(bm25_corpus, k=config.CROSSENCODER_KIN)

    return vectorstore, bm25_retriever