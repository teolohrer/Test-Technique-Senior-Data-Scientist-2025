from typing import Any
from sentence_transformers import CrossEncoder

from config import RAGConfig
from defaults import RRF_K

class CrossEncoderWrapper:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
    
    @staticmethod
    def from_config(config: RAGConfig):
        return CrossEncoderWrapper(model_name=config.cross_encoder_model)

    def score(self, query: str, document: str) -> float:
        return float(self.model.predict([[query, document]])[0])
    
    def batch_score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, doc] for doc in documents]
        return [float(score) for score in self.model.predict(pairs)]
    
    def rank_documents(self, query: str, documents: list[dict], key: str="document") -> list[dict]:
        if len(documents) == 0:
            return documents

        pairs = [[query, doc[key]] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["metadata"]["cross_encoder_score"] = float(score)

        return sorted(documents, key=lambda d: d["metadata"].get("cross_encoder_score", 0), reverse=True)

def rank_documents_by_key(documents: list[dict], key: str="document_score", reverse: bool = True) -> list[dict]:
    return sorted(documents, key=lambda d: d["metadata"].get(key, 0), reverse=reverse)


def rrf_fusion(all_results: list, top_k=10):
    scores = {}
    all_docs = {doc["id"]: doc for results in all_results for doc in results}

    for results in all_results:
        for rank, doc in enumerate(results, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [all_docs[doc] for doc, _score in ranked]