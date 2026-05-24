from ingest import dense_embedding_model, sparse_embedding_model, late_interaction_embedding_model
from qdrant_client.models import models, Prefetch
from langchain_core.documents import Document
import config

def retrieve(query, rag_client, cosine_weight=config.COSINE_WEIGHT, bm25_weight=config.BM25_WEIGHT):

    dense_query_embedding = list(dense_embedding_model.embed([query]))[0]
    sparse_query_embedding = list(sparse_embedding_model.embed([query]))[0]
    sparse_query_embedding = models.SparseVector(indices=sparse_query_embedding.indices.tolist(), values=sparse_query_embedding.values.tolist())
    late_query_embedding = list(late_interaction_embedding_model.embed([query]))[0]

    prefetch=[
        Prefetch(
            prefetch=[
                Prefetch(query=dense_query_embedding, using="dense", limit=config.CROSSENCODER_KIN),
                Prefetch(query=sparse_query_embedding, using="sparse", limit=config.CROSSENCODER_KIN),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=[cosine_weight, bm25_weight])),
            limit=config.CROSSENCODER_KIN
        )
    ]

    results = rag_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=late_query_embedding,
        using="multi",
        with_payload=True,
        limit=config.CROSSENCODER_KOUT
        )
    
    result_docs = []
    for result in results.points:
        doc = Document(
            page_content=result.payload["page_content"],
            metadata={k: v for k, v in result.payload.items() if k != "page_content"}
        )
        result_docs.append(doc)


    return result_docs