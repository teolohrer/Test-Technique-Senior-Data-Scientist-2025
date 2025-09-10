from dataclasses import dataclass
import time
import sys
from typing import Any

from print_utils import format_response, print_error, print_info, print_warning, timing_decorator
from bm25_retriever import BM25Retriever
from clustering import Clustering
from dense_retriever import DenseRetriever
from mmr import MMR
from process_data import DataLoader, prepare_chunks, prepare_paragraphs, prepare_sentences, prepare_sliding_sentences
from prompts import format_rag_prompt
from question_expansion import QuestionExpansion
from rerank import CrossEncoderWrapper, rrf_fusion
from config import RAGConfig


class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.loader = DataLoader.from_config(config)
        self.embedding_function = self.loader.embedding_function
        self.mmr = MMR(self.embedding_function)
        self.clustering = Clustering(self.embedding_function)
        self.cross_encoder = CrossEncoderWrapper.from_config(config)
    
    @timing_decorator
    def load(self, input_file_path: str):
        collection_paragraphs = self.loader.load_data(input_file_path, "paragraphs", data_prepare_func=prepare_paragraphs)
        collection_chunks = self.loader.load_data(input_file_path, "chunks", data_prepare_func=prepare_chunks)
        collection_sentences = self.loader.load_data(input_file_path, "sentences", data_prepare_func=prepare_sentences)
        collection_sliding_sentences = self.loader.load_data(input_file_path, "sliding_sentences", data_prepare_func=prepare_sliding_sentences)
        collections = {"paragraphs": collection_paragraphs, "chunks": collection_chunks, "sentences": collection_sentences, "sliding_sentences": collection_sliding_sentences}
        self.retrievers = {k: {"sparse": BM25Retriever(v), "dense": DenseRetriever(v)} for k, v in collections.items()}

    @timing_decorator
    def question_expansion(self, question: str, context: list[dict], max_questions: int) -> list[str]:
        expander = QuestionExpansion.from_config(self.config)
        result = expander.expand(question, context)[:max_questions]
        if self.config.verbose:
            print_info("=== Expanded Questions ===")
            for i, q in enumerate(result):
                print_info(f"{i+1}. {q}")
            print_info("=== End of Expanded Questions ===")
        return result

    def retrieve_context(self, retrievers, question: str, top_k: int) -> list[list[dict]]:
        context = []
        for retriever in retrievers:
            context.append(retriever.retrieve(question, top_k))
        return context

    def rerank_context_rrf(self, context: list[list[dict]], top_k: int) -> list[dict]:
        return rrf_fusion(context, top_k=top_k)

    @timing_decorator
    def rerank_context_cross_encoder(self, question: str, context: list[dict], top_k: int) -> list[dict]:
        return self.cross_encoder.rank_documents(query=question, documents=context, key="document")[:top_k]
    
    @timing_decorator
    def mmr_context(self, question: str, context: list[dict], top_k: int, relevance_weight: float) -> list[dict]:
        return self.mmr.mmr_collection(context, question, relevance_weight=relevance_weight, top_k=top_k)
    
    def mmr_questions(self, original_question: str, questions: list[str], top_k: int, relevance_weight: float) -> list[str]:
        return self.mmr.mmr_documents(questions, original_question, relevance_weight=relevance_weight, top_k=top_k)

    @timing_decorator
    def cluster_context(self, context: list[dict], n_clusters: int) -> list[list[dict]]:
        # k-means is empirically better than hdbscan here
        clusters = self.clustering.kmeans_clustering(context, n_clusters=n_clusters)
        # clusters = self.clustering.hdbscan_clustering(context, min_cluster_size=self.config.hdbscan_min_cluster_size)
        return clusters

    def sample_clusters(self, clustered_context: list[list[dict]], smoothing: float, top_k: int) -> list[dict]:
        return self.clustering.smooth_sample_clusters(clusters=clustered_context, laplace_smoothing=smoothing, top_k=top_k)
    
    @timing_decorator
    def generate_answer(self, question: str, context: list[dict]) -> str:
        prompt = format_rag_prompt(context, question)
        if self.config.verbose:
            print_info("=== Prompt to the LLM ===")
            print(prompt)
            print_info("=== End of prompt ===")
        try:
            result = self.config.client.generate(
                prompt=prompt,
                model=self.config.generation_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                think=False,
            )

            # Affichage du résultat
            formatted_response = format_response(result, self.config.verbose)
            return formatted_response
        except KeyboardInterrupt:
            print_warning("\nInterrompu par l'utilisateur")
            sys.exit(130)
        except Exception as e:
            print_error(f"Erreur inattendue: {str(e)}")
            sys.exit(1)

    
    @timing_decorator
    def run(self, question: str) -> Any:
        initial_retrievers = [
            self.retrievers["sentences"]["sparse"],
            self.retrievers["chunks"]["sparse"],
            self.retrievers["sentences"]["dense"],
            self.retrievers["chunks"]["dense"],
        ]

        initial_context_by_retriever = self.retrieve_context(initial_retrievers, question, self.config.initial_context_size)
        initial_context = self.rerank_context_rrf(initial_context_by_retriever, top_k=self.config.initial_context_size)
        expanded_questions = self.question_expansion(question, initial_context, self.config.max_expanded_questions)

        retrievers = [
            self.retrievers["chunks"]["sparse"],
            self.retrievers["paragraphs"]["dense"],
            self.retrievers["chunks"]["dense"],
            # self.retrievers["sliding_sentences"]["sparse"],
            # self.retrievers["sentences"]["sparse"],
        ]

        context_by_question_by_retriever = {}
        for q in expanded_questions:
            context_by_question_by_retriever[q] = self.retrieve_context(retrievers, q, self.config.subquestion_max_documents)

        # We now have subquestion_max_documents * n_retrievers * max_expanded_questions documents
        
        context_by_question = []
        for subquestion, question_context in context_by_question_by_retriever.items():
            merged_context = self.rerank_context_rrf(context=question_context, top_k=self.config.subquestion_max_documents)
            context_by_question.append(merged_context)
        
        # We now have subquestion_max_documents * max_expanded_questions documents

        context = self.rerank_context_rrf(context=context_by_question, top_k=self.config.subquestion_fusion_max_documents)

        # Make context unique by content, not id
        # Some documents may appear multiple times with different ids (from different collections)

        seen_contents = set()
        unique_context = []
        for doc in context:
            if doc["document"] not in seen_contents:
                seen_contents.add(doc["document"])
                unique_context.append(doc)
        context = unique_context

        # We now have subquestion_fusion_max_documents documents

        clustered_context = self.cluster_context(context, self.config.kmeans_n_clusters)

        # Rank whithin each cluster with cross-encoder
        ranked_clustered_context = [self.cross_encoder.rank_documents(question, cluster) for cluster in clustered_context]

        context = self.sample_clusters(ranked_clustered_context, self.config.cluster_laplace_smoothing, self.config.cluster_sample_size)

        # We now have cluster_sample_size documents

        context = self.mmr_context(question, context, top_k=self.config.mmr_top_k, relevance_weight=self.config.mmr_relevance_weight)

        # We now have mmr_top_k documents

        context = self.rerank_context_cross_encoder(question, context, top_k=self.config.max_context_docs)

        answer = self.generate_answer(question, context)

        return answer

def search_word(context, word):
    counter = 0
    for doc in context:
        if word.lower() in doc["document"].lower():
            counter += 1
    return counter