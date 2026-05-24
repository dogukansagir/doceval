import pymupdf as fitz
from google import genai
from google.genai import types
import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from uuid import uuid4
import os
import json
import time

client = genai.Client(api_key=config.GEMINI_API_KEY)
rag_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, cloud_inference=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])

vectors_config={"dense": models.VectorParams(size=384, distance=Distance.COSINE),
                "multi": models.VectorParams(
                    size=96,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                    hnsw_config=models.HnswConfigDiff(m=0))
                }
sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}

dense_embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_embedding_model = SparseTextEmbedding(model_name="qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding(model_name="answerdotai/answerai-colbert-small-v1")

if rag_client.collection_exists(collection_name=config.QDRANT_COLLECTION_NAME) == False:
    rag_client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config
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

    pointlist = []
    texts = [doc.page_content for doc in all_documents]
    dense_vectors = list(dense_embedding_model.embed(texts))
    sparse_vectors = list(sparse_embedding_model.embed(texts))
    late_vectors = list(late_interaction_embedding_model.embed(texts))
    for doc, sparse_vec, dense_vec, late_vec in zip(all_documents, sparse_vectors, dense_vectors, late_vectors):
        sparse_vec = models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())
        pointlist.append(PointStruct(
            id=str(uuid4()),
            vector={"dense": dense_vec, "multi": late_vec, "sparse": sparse_vec},
            payload={"page_content": doc.page_content, **doc.metadata}
        ))
    
    rag_client.upload_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        points=pointlist,
        batch_size=25,
        max_retries=3
    )
    return rag_client

def ingest_pdf(pdf_folder = "./pdfs"):
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
            chunk_and_store(enriched_blocks,pdf)
            ingested_files.append(pdf)
            with open("ingested_files.json", "w") as f:
                json.dump(ingested_files, f)

    return rag_client