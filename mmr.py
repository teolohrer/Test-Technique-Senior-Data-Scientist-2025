import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


from embedding import EmbeddingWrapper


class MMR:
    def __init__(self, embedding_function: EmbeddingWrapper | None = None):
        self.embedding_function = embedding_function

    def get_bm25_similarities(self, documents: list[str]) -> list:
        tokenized_corpus = [doc.split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        return [bm25.get_scores(d) for d in tokenized_corpus]

    def _check_embedding_function(self):
        if self.embedding_function is None:
            raise ValueError(
                "Embedding function must be provided either during initialization or as a parameter."
            )

    def mmr_collection(
        self, documents: list[dict], query: str, relevance_weight=0.5, top_k=5
    ) -> list[dict]:
        self._check_embedding_function()
        if self.embedding_function:
            documents_texts = [doc["document"] for doc in documents]
            doc_embeddings = self.embedding_function(documents_texts)
            query_embedding = self.embedding_function([query])[0]

            bm25_similarities = self.get_bm25_similarities(documents_texts)

            selected_indices = self.mmr_embeddings(
                doc_embeddings, query_embedding, relevance_weight, top_k, bm25_similarities
            )
            return [documents[i] for i in selected_indices]
        return []

    def mmr_documents(
        self, documents: list[str], query: str, relevance_weight=0.5, top_k=5
    ) -> list[str]:
        self._check_embedding_function()
        if self.embedding_function:
            doc_embeddings = self.embedding_function(documents)
            query_embedding = self.embedding_function([query])[0]
            bm25_similarities = self.get_bm25_similarities(documents)
            selected_indices = self.mmr_embeddings(
                doc_embeddings, query_embedding, relevance_weight, top_k, bm25_similarities
            )
            return [documents[i] for i in selected_indices]
        return []

    def mmr_embeddings(
        self,
        doc_embeddings,
        query_embedding,
        lambda_param=0.5,
        top_k=5,
        bm25_similarities: list | None = None,
    ) -> list[int]:
        # modified MMR implementation to use both dense and sparse similarities for redundancy

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
                redundancy = (
                    max([doc_doc_similarities[idx][sel_idx] for sel_idx in selected_indices])
                    if selected_indices
                    else 0
                )
                if bm25_similarities:
                    redundancy_bm25 = (
                        max([bm25_similarities[idx][sel_idx] for sel_idx in selected_indices])
                        if selected_indices
                        else 0
                    )
                    redundancy = max(redundancy, redundancy_bm25)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                mmr_scores.append((mmr_score, idx))

            # Select the document with the highest MMR score
            next_selected = max(mmr_scores, key=lambda x: x[0])[1]
            selected_indices.append(next_selected)
            unselected_indices.remove(next_selected)

        return selected_indices
