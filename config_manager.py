"""
Comprehensive Configuration Manager with Pydantic Validation
===========================================================

This module provides a unified configuration management system for the RAG application.
It supports loading configuration from YAML files and command line arguments with
proper validation using Pydantic models.

Usage:
    # Load from YAML with CLI overrides
    config = load_config(yaml_file="config.yaml", cli_args=args_dict)

    # Convert to existing RAGConfig
    rag_config = config.to_rag_config(client)
"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

import defaults
from ollama_client import OllamaClient
from typing import Optional


class Config(BaseModel):
    """
    Comprehensive configuration model with Pydantic validation.

    This model encompasses all configurable parameters for the RAG system,
    providing validation, type checking, and default value management.
    """

    # === Client Configuration ===
    client: Optional[OllamaClient] = Field(
        default=None, description="Ollama client instance for LLM communication"
    )

    # === Models Configuration ===
    embedding_model: str = Field(
        default=defaults.EMBEDDING_MODEL, description="Embedding model name for document embeddings"
    )
    generation_model: str = Field(
        default=defaults.GENERATION_MODEL, description="Generation model name for LLM responses"
    )
    question_expansion_model: str = Field(
        default=defaults.QUESTION_EXPANSION_MODEL, description="Model for question expansion"
    )
    cross_encoder_model: str = Field(
        default=defaults.CROSS_ENCODER_MODEL, description="Cross-encoder model for reranking"
    )

    # === Infrastructure Configuration ===
    ollama_base_url: str = Field(
        default=defaults.OLLAMA_BASE_URL, description="Ollama server base URL"
    )
    ollama_timeout: int = Field(
        default=defaults.OLLAMA_TIMEOUT, gt=0, description="Ollama request timeout in seconds"
    )
    ollama_connection_timeout: int = Field(
        default=defaults.OLLAMA_CONNECTION_TIMEOUT,
        gt=0,
        description="Ollama connection timeout in seconds",
    )
    chroma_persist: str = Field(
        default=defaults.CHROMA_PERSIST, description="ChromaDB persistence directory path"
    )
    data_filepath: str = Field(
        default=defaults.DATA_FILEPATH, description="Data file path for corpus"
    )

    # === LLM Generation Parameters ===
    temperature: float = Field(
        default=defaults.TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="LLM temperature for response generation",
    )
    max_tokens: int = Field(
        default=defaults.MAX_TOKENS, gt=0, description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=defaults.TOP_P, ge=0.0, le=1.0, description="Top-p nucleus sampling parameter"
    )
    top_k: int = Field(default=defaults.TOP_K, gt=0, description="Top-k sampling parameter")

    # === RAG & Retrieval Configuration ===
    max_context_docs: int = Field(
        default=defaults.MAX_CONTEXT_DOCS,
        gt=0,
        description="Maximum number of documents in final context",
    )
    initial_context_size: int = Field(
        default=defaults.INITIAL_CONTEXT_SIZE,
        gt=0,
        description="Initial context size for question expansion seeding",
    )
    max_expanded_questions: int = Field(
        default=defaults.MAX_EXPANDED_QUESTIONS,
        gt=0,
        description="Maximum number of expanded questions to generate",
    )
    subquestion_max_documents: int = Field(
        default=defaults.SUBQUESTION_MAX_DOCUMENTS,
        gt=0,
        description="Maximum documents per sub-question",
    )
    subquestion_fusion_max_documents: int = Field(
        default=defaults.SUBQUESTION_FUSION_MAX_DOCUMENTS,
        gt=0,
        description="Maximum documents after sub-question fusion",
    )

    # === Clustering Configuration ===
    kmeans_n_clusters: int = Field(
        default=defaults.KMEANS_N_CLUSTERS,
        gt=0,
        description="Number of clusters for K-means clustering",
    )
    hdbscan_min_cluster_size: int = Field(
        default=defaults.HDBSCAN_MIN_CLUSTER_SIZE,
        gt=0,
        description="Minimum cluster size for HDBSCAN",
    )
    cluster_laplace_smoothing: float = Field(
        default=defaults.CLUSTER_LAPLACE_SMOOTHING,
        ge=0.0,
        description="Laplace smoothing factor for cluster sampling",
    )
    cluster_sample_size: int = Field(
        default=defaults.CLUSTER_SAMPLE_SIZE,
        gt=0,
        description="Number of documents to sample from clusters",
    )

    # === MMR Configuration ===
    mmr_relevance_weight: float = Field(
        default=defaults.MMR_RELEVANCE_WEIGHT,
        ge=0.0,
        le=1.0,
        description="MMR relevance weight (0=diversity, 1=relevance)",
    )
    mmr_top_k: int = Field(
        default=defaults.MMR_TOP_K, gt=0, description="Top-k documents for MMR processing"
    )

    # === Question Expansion Configuration ===
    question_expansion_temperature: float = Field(
        default=defaults.QUESTION_EXPANSION_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="Temperature for question expansion generation",
    )
    question_expansion_max_tokens: int = Field(
        default=defaults.QUESTION_EXPANSION_MAX_TOKENS,
        gt=0,
        description="Maximum tokens for question expansion",
    )
    question_expansion_top_p: float = Field(
        default=defaults.QUESTION_EXPANSION_TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p for question expansion generation",
    )

    # === Runtime Control Configuration ===
    verbose: bool = Field(
        default=defaults.VERBOSE, description="Enable verbose output and debugging information"
    )

    # === CLI-Only Parameters (not configurable via YAML) ===
    model: Optional[str] = Field(default=None, description="Override model selection (CLI only)")
    list_models: bool = Field(
        default=False, description="List available models and exit (CLI only)"
    )

    class Config:
        extra = "forbid"  # Prevent unknown parameters
        validate_assignment = True  # Validate on assignment
        arbitrary_types_allowed = True  # Allow OllamaClient type
        json_schema_extra = {
            "example": {
                "embedding_model": "nomic-embed-text:v1.5",
                "generation_model": "qwen3:8b",
                "temperature": 0.3,
                "max_tokens": 5000,
                "max_context_docs": 10,
                "verbose": True,
            }
        }

    @field_validator("ollama_base_url")
    @classmethod
    def validate_url(cls, v):
        """Validate that Ollama URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        return v

    @field_validator("data_filepath", "chroma_persist")
    @classmethod
    def validate_paths(cls, v):
        """Validate that paths are reasonable (not empty)."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_document_limits(self):
        """Ensure document limit parameters are logically consistent."""
        # Ensure the pipeline flow makes sense
        if self.max_context_docs > self.mmr_top_k:
            raise ValueError("max_context_docs should not exceed mmr_top_k")
        if self.max_context_docs > self.cluster_sample_size:
            raise ValueError("max_context_docs should not exceed cluster_sample_size")

        return self


def load_config(
    yaml_file: Optional[str] = None, cli_args: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration with precedence: CLI > YAML > defaults.

    Args:
        yaml_file: Path to YAML configuration file (optional)
        cli_args: Dictionary of CLI arguments (optional)

    Returns:
    Config: Validated configuration instance

    Raises:
        ValidationError: If configuration validation fails
        FileNotFoundError: If YAML file is specified but doesn't exist
        yaml.YAMLError: If YAML file is malformed
    """
    config_data = {}

    # 1. Load from YAML file if provided
    if yaml_file:
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_file}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data.update(yaml_data)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

    # 2. Override with CLI arguments (excluding None values and special params)
    if cli_args:
        # Filter out CLI-only parameters and None values
        cli_only_params = {"question", "list_models", "config"}  # 'config' is only for loading YAML
        cli_overrides = {
            k: v for k, v in cli_args.items() if v is not None and k not in cli_only_params
        }

        # Handle special CLI-only parameters
        if "list_models" in cli_args:
            cli_overrides["list_models"] = cli_args["list_models"]
        if "model" in cli_args:
            cli_overrides["model"] = cli_args["model"]

        config_data.update(cli_overrides)

    # 3. Create and validate with Pydantic
    return Config(**config_data)


def create_sample_config(output_path: str = "config.yaml") -> None:
    """
    Create a sample YAML configuration file with all available options.

    Args:
        output_path: Path where to save the sample configuration file
    """
    # Create sample config without client for YAML generation
    sample_config = Config(client=None)

    config_content = f"""# RAG Application Configuration
# =============================
# This file contains all configurable parameters for the RAG system.
# CLI arguments will override values specified here.

# === Model Configuration ===
embedding_model: "{sample_config.embedding_model}"
generation_model: "{sample_config.generation_model}"
question_expansion_model: "{sample_config.question_expansion_model}"
cross_encoder_model: "{sample_config.cross_encoder_model}"

# === Infrastructure ===
ollama_base_url: "{sample_config.ollama_base_url}"
ollama_timeout: {sample_config.ollama_timeout}
ollama_connection_timeout: {sample_config.ollama_connection_timeout}
chroma_persist: "{sample_config.chroma_persist}"
data_filepath: "{sample_config.data_filepath}"

# === LLM Generation Parameters ===
temperature: {sample_config.temperature}
max_tokens: {sample_config.max_tokens}
top_p: {sample_config.top_p}
top_k: {sample_config.top_k}

# === RAG & Retrieval ===
max_context_docs: {sample_config.max_context_docs}
initial_context_size: {sample_config.initial_context_size}
max_expanded_questions: {sample_config.max_expanded_questions}
subquestion_max_documents: {sample_config.subquestion_max_documents}
subquestion_fusion_max_documents: {sample_config.subquestion_fusion_max_documents}

# === Clustering ===
kmeans_n_clusters: {sample_config.kmeans_n_clusters}
hdbscan_min_cluster_size: {sample_config.hdbscan_min_cluster_size}
cluster_laplace_smoothing: {sample_config.cluster_laplace_smoothing}
cluster_sample_size: {sample_config.cluster_sample_size}

# === MMR (Diversity) ===
mmr_relevance_weight: {sample_config.mmr_relevance_weight}
mmr_top_k: {sample_config.mmr_top_k}

# === Question Expansion ===
question_expansion_temperature: {sample_config.question_expansion_temperature}
question_expansion_max_tokens: {sample_config.question_expansion_max_tokens}
question_expansion_top_p: {sample_config.question_expansion_top_p}

# === Runtime Control ===
verbose: {str(sample_config.verbose).lower()}

# Note: 'model' and 'list_models' are CLI-only parameters and cannot be set here
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"Sample configuration saved to: {output_path}")


if __name__ == "__main__":
    # Create sample configuration for users
    create_sample_config()
