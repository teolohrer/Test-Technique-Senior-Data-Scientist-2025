
class DenseRetriever:
    def __init__(self, collection):
        self.collection = collection

    def retrieve(self, query, k: int):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
    )
        return [{"id": par_id, "document": doc, "metadata": meta} for par_id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])]

    def top_k_documents_with_prev_next(self, query, k: int):
        results = self.retrieve(query, k)
        # Complete documents texts with previous and next paragraphs
        for result in results:
            if "prev_par_id" in result["metadata"] and result["metadata"]["prev_par_id"]:
                prev = self.collection.get(ids=[result["metadata"]["prev_par_id"]])
                if prev and len(prev["documents"]) > 0:
                    result["document"] = prev["documents"][0] + "\n" + result["document"]
            if "next_par_id" in result["metadata"] and result["metadata"]["next_par_id"]:
                next_ = self.collection.get(ids=[result["metadata"]["next_par_id"]])
                if next_ and len(next_["documents"]) > 0:
                    result["document"] = result["document"] + "\n" + next_["documents"][0]
        return results