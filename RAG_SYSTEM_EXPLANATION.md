# RAG System Implementation Explanation

This document explains the RAG system implemented in this project, which uses a multi-stage pipeline combining hybrid retrieval, question expansion, clustering, and reranking techniques.

## System Overview

The RAG system processes documents at multiple granularities and employs a sophisticated 9-stage pipeline to retrieve and rank the most relevant context for question answering using LLMs.

**Key Components:**
- Multi-granularity document processing (paragraphs, chunks, sentences)
- Hybrid retrieval (dense + sparse)
- Question expansion using LLM
- Multi-stage reranking with RRF, cross-encoder, and MMR
- Document clustering for diversity
- French-language prompting for answer generation

## Document Processing (`process_data.py`)

The system creates three document collections with different granularities:

### 1. Paragraphs (`prepare_paragraphs`)
- Loads CSV data with paragraph-level documents
- Links consecutive paragraphs with `prev_par_id` and `next_par_id` metadata
- Preserves document structure and ordering

### 2. Chunks (`prepare_chunks`) 
- Creates fixed-size overlapping text chunks (500 chars, 50 char overlap)
- Reconstructs full articles by joining paragraphs
- Adds chunk metadata: index, start position, size, total chunks

### 3. Sentences (`prepare_sentences`)
- Splits paragraphs into individual sentences using period delimiter
- Creates new IDs with format `s{par_id}_{sentence_index}`
- Maintains paragraph metadata for each sentence

### 4. Sliding Sentences (Currently Disabled)
- Feature available but commented out in current implementation
- Would create overlapping sentence windows for additional retrieval granularity

## Retrieval Components

### Dense Retriever (`dense_retriever.py`)
- Uses ChromaDB's vector similarity search with Ollama embeddings
- [`retrieve()`](dense_retriever.py:5): Queries collection and returns documents with metadata
- [`top_k_documents_with_prev_next()`](dense_retriever.py:14): Expands context by including adjacent paragraphs

### Sparse Retriever (`bm25_retriever.py`)
- Implements BM25 using `rank_bm25` library
- Pre-computes BM25 index from collection documents
- [`retrieve()`](bm25_retriever.py:12): Returns top-k documents based on BM25 scores

## Pipeline Stages (`pipeline.py`)

The [`RAGPipeline.run()`](pipeline.py:147) method implements a 9-stage process:

### Stage 1: Initial Context Retrieval
**Goal**: Get preliminary context using lexical matching to understand the question scope
```python
initial_retrievers = [
    self.retrievers["sentences"]["sparse"],
    self.retrievers["chunks"]["sparse"],
]
```
- Retrieves `initial_context_size` documents from 2 sparse retrievers (sentences + chunks)
- Uses only BM25 lexical matching for fast initial retrieval
- Fuses results using RRF to get balanced initial context

### Stage 2: Question Expansion
**Goal**: Break down complex or ambiguous questions into specific, targeted sub-queries
- [`question_expansion()`](pipeline.py:61): Uses LLM to generate expanded queries from initial context
- Generates up to `max_expanded_questions` sub-queries
- Uses French prompt template to create keyword-focused search queries

### Stage 3: Multi-Query Retrieval
**Goal**: Cast a wide net using paragraph-level documents for comprehensive coverage
```python
retrievers = [
    self.retrievers["paragraphs"]["sparse"],
    self.retrievers["paragraphs"]["dense"],
]
```
- Retrieves `subquestion_max_documents` for each expanded question using 2 paragraph-level retrievers
- Total: `subquestion_max_documents × 2 × max_expanded_questions` documents

### Stage 4: Per-Question Fusion
**Goal**: Get the most relevant documents for each expanded sub-question using precise relevance scoring
- Flattens all retriever results for each sub-question into single list
- [`rerank_context_cross_encoder()`](pipeline.py:184): Uses cross-encoder for accurate relevance ranking
- Reduces to `subquestion_max_documents` per question

### Stage 5: Cross-Question Fusion
**Goal**: Combine the best documents across all sub-questions while avoiding redundancy
- Applies RRF to merge top documents from all sub-questions
- Reduces to `subquestion_fusion_max_documents` documents
- Deduplicates by document content to remove duplicates from different collections

### Stage 6: Clustering and Sampling
**Goal**: Ensure topical diversity by organizing documents into semantic clusters
- [`cluster_context()`](pipeline.py:208): K-means clustering into `kmeans_n_clusters` semantic groups
- Cross-encoder ranking within each cluster for relevance ordering
- [`sample_clusters()`](pipeline.py:215): Proportional sampling with Laplace smoothing to ensure representation from all topics
- Reduces to `cluster_sample_size` documents

### Stage 7: Diversity Optimization
**Goal**: Remove redundant information while maintaining high relevance
- [`mmr_context()`](pipeline.py:232): MMR balances relevance vs diversity to avoid repetitive content
- Reduces to `mmr_top_k` documents

### Stage 8: Source Diversity Filter
**Goal**: Ensure source diversity by limiting documents per source
- Filters to at most 1 document per `document_netloc` source
- Prevents over-representation of any single source
- Maintains source diversity in final context

### Stage 9: Final Ranking and Validation
**Goal**: Final precision ranking with quality assurance
- [`rerank_context_cross_encoder()`](pipeline.py:251): Ultimate relevance scoring using cross-encoder
- Reduces to final `max_context_docs` for LLM context window
- **Quality checks**: Returns error message if no relevant context found
- **Warning system**: Alerts when context is limited (< 3 documents)

## Reranking Components

### RRF Fusion (`rerank.py`)
- [`rrf_fusion()`](rerank.py:46): Combines multiple result lists using reciprocal rank fusion
- Score formula: `1/(RRF_K + rank)` where `RRF_K = 60`
- Returns top-k documents by aggregated scores

### Cross-Encoder (`rerank.py`)
- Uses `sentence-transformers` CrossEncoder model
- [`rank_documents()`](rerank.py:23): Scores query-document pairs and sorts by relevance
- Adds `cross_encoder_score` to document metadata

### MMR Diversity (`mmr.py`)
- [`mmr_collection()`](mmr.py:24): Applies MMR to document collections
- Combines dense embeddings and sparse BM25 similarities for redundancy detection
- Balances relevance vs diversity using `relevance_weight` parameter

## Clustering (`clustering.py`)

### K-Means Implementation
- [`kmeans_clustering()`](clustering.py:39): Groups documents using sklearn KMeans
- Uses document embeddings as features
- Returns list of document clusters with added `kmeans_label` metadata

### Cluster Sampling
- [`smooth_sample_clusters()`](clustering.py:57): Proportional sampling with Laplace smoothing
- Ensures representation from all clusters while respecting cluster sizes
- Formula: `(cluster_size + smoothing) / (total + n_clusters × smoothing)`

## Answer Generation

### Prompt Engineering (`prompts.py`)
- [`format_rag_prompt()`](prompts.py:192): Uses structured French prompt with examples
- Requires source citation format: `[id / netloc / date]`
- Emphasizes factual grounding and neutral tone

### LLM Integration
- [`generate_answer()`](pipeline.py:118): Calls Ollama client for text generation
- Uses configured model (default: `qwen3:8b`) with temperature and token limits
- Formats response using [`print_utils.py`](print_utils.py) functions

## Configuration

The system is configured via [`config/config.yaml`](config/config.yaml) with parameters for:
- Model names (embedding, generation, cross-encoder)
- Retrieval limits (context sizes, top-k values)
- Clustering parameters (n_clusters, smoothing)
- MMR settings (relevance weight)
- LLM generation parameters (temperature, max_tokens)

## Key Design Decisions

**Simplified Retrieval Strategy**: The current implementation focuses on sparse retrieval for initial context and paragraph-level documents for main retrieval, prioritizing efficiency over exhaustive coverage.

**Source Diversity**: A dedicated filtering stage ensures no single source dominates the final context, promoting balanced information representation.

**Quality Assurance**: Built-in validation prevents empty responses and warns users when context is insufficient for reliable answers.

**Configurable Pipeline**: All components are initialized through the [`Config`](config_manager.py) class with CLI override support.