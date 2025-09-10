# MMR (Maximal Marginal Relevance) for document selection
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils import embedding_functions
from config import EMBEDDING_MODEL


def mmr(collection, documents: list[dict], query: str, relevance_weight=0.5, top_k=5):
    doc_embeddings = collection.get(ids=[doc["id"] for doc in documents], include=["embeddings"])["embeddings"]
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=EMBEDDING_MODEL
    )
    query_embedding = ollama_ef([query])[0]
    return [documents[i] for i in raw_mmr(doc_embeddings, query_embedding, relevance_weight, top_k)]

def raw_mmr(doc_embeddings, query_embedding, lambda_param=0.5, top_k=5):

    if len(doc_embeddings) == 0:
        return []

    doc_embeddings = np.array(doc_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Compute similarity between documents and the query
    doc_query_similarities = cosine_similarity(doc_embeddings, query_embedding).flatten()
    # Compute similarity among documents
    doc_doc_similarities = cosine_similarity(doc_embeddings)

    selected_indices = []
    unselected_indices = list(range(len(doc_embeddings)))

    # Select the document most similar to the query first
    first_selected = unselected_indices[np.argmax(doc_query_similarities)]
    selected_indices.append(first_selected)
    unselected_indices.remove(first_selected)

    while len(selected_indices) < top_k and unselected_indices:
        mmr_scores = []
        for idx in unselected_indices:
            relevance = doc_query_similarities[idx]
            redundancy = max([doc_doc_similarities[idx][sel_idx] for sel_idx in selected_indices]) if selected_indices else 0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, idx))
        
        # Select the document with the highest MMR score
        next_selected = max(mmr_scores, key=lambda x: x[0])[1]
        selected_indices.append(next_selected)
        unselected_indices.remove(next_selected)

    return selected_indices