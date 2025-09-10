from typing import Any
from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, RRF_K


def rerank_cutoff_by_key(documents: list[dict], key: str="document_score", reverse: bool = True) -> list[dict]:
    return sorted(documents, key=lambda d: d["metadata"].get(key, 0), reverse=reverse)

def cross_encoder_rerank(documents: list[dict], query: str, model_name: str = CROSS_ENCODER_MODEL) -> list[dict]:

    if len(documents) == 0:
        return documents

    cross_encoder = CrossEncoder(model_name)

    pairs = [[query, doc["document"]] for doc in documents]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["metadata"]["cross_encoder_score"] = float(score)

    return sorted(documents, key=lambda d: d["metadata"].get("cross_encoder_score", 0), reverse=True)

def rrf_fusion(all_results: list, top_k=10):
    scores = {}
    all_docs = {doc["id"]: doc for results in all_results for doc in results}

    for results in all_results:
        for rank, doc in enumerate(results, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [all_docs[doc] for doc, _score in ranked]