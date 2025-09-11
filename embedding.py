from chromadb.utils import embedding_functions
from chromadb.api.types import Embedding

from defaults import EMBEDDING_MODEL, OLLAMA_BASE_URL


class EmbeddingWrapper:
    def __init__(self, model_name: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.ef = embedding_functions.OllamaEmbeddingFunction(url=base_url, model_name=model_name)

    def __call__(self, texts: list[str]) -> list[Embedding]:
        return self.ef(texts)
