# Model Configuration
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
# EMBEDDING_MODEL = "snowflake-arctic-embed2:568m"
GENERATION_MODEL = "qwen3:8b"
# GENERATION_MODEL = "mistral:7b"
QUESTION_EXPANSION_MODEL = "qwen3:8b"
# QUESTION_EXPANSION_MODEL = "mistral:7b"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"

# Data and Storage Configuration
CHROMA_PERSIST = "chroma_db"
DATA_FILEPATH = "data/corpus.csv"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds
OLLAMA_CONNECTION_TIMEOUT = 5  # seconds

# RAG Pipeline Configuration
MAX_CONTEXT_DOCS = 10  # Final number of context documents to include for generation
INITIAL_CONTEXT_SIZE = 50  # For question expansion seeding
MAX_EXPANDED_QUESTIONS = 10
SUBQUESTION_MAX_DOCUMENTS = 100  # per sub-question
SUBQUESTION_FUSION_MAX_DOCUMENTS = 200  # after fusion of sub-questions

# Clustering Configuration
KMEANS_N_CLUSTERS = 3
HDBSCAN_MIN_CLUSTER_SIZE = 5
CLUSTER_LAPLACE_SMOOTHING = 0.3
CLUSTER_SAMPLE_SIZE = 100

# Reranking Configuration
RRF_K = 60

# MMR Configuration
MMR_RELEVANCE_WEIGHT = 0.9
MMR_TOP_K = 50

# Question Expansion Configuration
QUESTION_EXPANSION_TEMPERATURE = 0.9
QUESTION_EXPANSION_MAX_TOKENS = 500
QUESTION_EXPANSION_TOP_P = 0.9
QUESTION_EXPANSION_TRIES = 3

# Default LLM Generation Parameters
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 5000
TOP_K = 5

# Verbosity
VERBOSE = True