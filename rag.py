from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import config

cross_encoder = CrossEncoder(model_name = config.CROSSENCODER_MODEL)

def retrieve(query, vectorstore, bm25_retriever):

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vectorstore.as_retriever(search_kwargs={"k": config.CROSSENCODER_KIN}), bm25_retriever], weights=[config.COSINE_WEIGHT, config.BM25_WEIGHT])
    retrieved_chunks = hybrid_retriever.get_relevant_documents(query)

    scores = cross_encoder.predict([(query, chunk.page_content) for chunk in retrieved_chunks])
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in scored_chunks[:config.CROSSENCODER_KOUT]]
    
    return top_chunks