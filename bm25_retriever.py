from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, collection):
        self.collection = collection
        frozen_data = collection.get(include=["documents"])
        corpus, self.ids = frozen_data["documents"], frozen_data["ids"]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 10):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)

        top_n_indices = doc_scores.argsort()[-k:][::-1]
        top_n_ids = [self.ids[i] for i in top_n_indices]
        results = self.collection.get(ids=top_n_ids)

        return [{"id": par_id, "document": doc, "metadata": meta} for par_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])]