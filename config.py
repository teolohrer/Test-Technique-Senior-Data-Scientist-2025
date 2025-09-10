from dataclasses import dataclass

from ollama_client import OllamaClient
import defaults

@dataclass
class RAGConfig:
    client: OllamaClient
    embedding_model: str
    generation_model: str
    chroma_persist: str
    base_ollama_url: str
    question_expansion_model: str
    question_expansion_temperature: float
    cross_encoder_model: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    initial_context_size: int
    max_expanded_questions: int
    max_context_docs: int
    subquestion_max_documents: int
    subquestion_fusion_max_documents: int
    kmeans_n_clusters: int
    hdbscan_min_cluster_size: int
    cluster_laplace_smoothing: float
    cluster_sample_size: int
    mmr_relevance_weight: float
    mmr_top_k: int
    ollama_connection_timeout: int
    verbose: bool

    @staticmethod
    def from_dict(config_dict: dict) -> 'RAGConfig':
        base_config = RAGConfig.default()
        if "base_ollama_url" in config_dict:
            base_config.client = OllamaClient(base_url=config_dict["base_ollama_url"])
        for k, v in config_dict.items():
            if hasattr(base_config, k):
                setattr(base_config, k, v)
        return base_config
    
    @staticmethod
    def from_args(args) -> 'RAGConfig':
        base_config = RAGConfig.default()
        if getattr(args, 'base_ollama_url', None):
            base_config.client = OllamaClient(base_url=args.base_ollama_url)
        for k, v in vars(args).items():
            if v is not None and hasattr(base_config, k):
                setattr(base_config, k, v)
        return base_config
    
    @staticmethod
    def default() -> 'RAGConfig':
        return RAGConfig(
            client=OllamaClient(base_url=defaults.OLLAMA_BASE_URL),
            embedding_model=defaults.EMBEDDING_MODEL,
            generation_model=defaults.GENERATION_MODEL,
            chroma_persist=defaults.CHROMA_PERSIST,
            base_ollama_url=defaults.OLLAMA_BASE_URL,
            question_expansion_model=defaults.QUESTION_EXPANSION_MODEL,
            question_expansion_temperature=defaults.QUESTION_EXPANSION_TEMPERATURE,
            cross_encoder_model=defaults.CROSS_ENCODER_MODEL,
            temperature=defaults.TEMPERATURE,
            max_tokens=defaults.MAX_TOKENS,
            top_p=defaults.TOP_P,
            top_k=defaults.TOP_K,
            max_context_docs=defaults.MAX_CONTEXT_DOCS,
            initial_context_size=defaults.INITIAL_CONTEXT_SIZE,
            subquestion_max_documents=defaults.SUBQUESTION_MAX_DOCUMENTS,
            max_expanded_questions=defaults.MAX_EXPANDED_QUESTIONS,
            subquestion_fusion_max_documents=defaults.SUBQUESTION_FUSION_MAX_DOCUMENTS,
            kmeans_n_clusters=defaults.KMEANS_N_CLUSTERS,
            hdbscan_min_cluster_size=defaults.HDBSCAN_MIN_CLUSTER_SIZE,
            cluster_sample_size=defaults.CLUSTER_SAMPLE_SIZE,
            cluster_laplace_smoothing=defaults.CLUSTER_LAPLACE_SMOOTHING,
            mmr_relevance_weight=defaults.MMR_RELEVANCE_WEIGHT,
            mmr_top_k=defaults.MMR_TOP_K,
            ollama_connection_timeout=defaults.OLLAMA_CONNECTION_TIMEOUT,
            verbose=defaults.VERBOSE
        )
    
    