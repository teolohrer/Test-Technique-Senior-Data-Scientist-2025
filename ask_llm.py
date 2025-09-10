"""
Script pour interroger un LLM local via Ollama
===============================================

Usage:
    python ask_llm.py "Qu'est-ce que le machine learning ?"
    python ask_llm.py "Explique le RAG" --model mistral:7b-instruct
    python ask_llm.py "Bonjour" --verbose

Ou avec uv:
    uv run ask_llm.py "Ma question"
"""

import argparse
import sys
import time
from typing import Any, Dict, Optional
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

from bm25_retriever import BM25Retriever
from clustering import hdbscan_clustering
from process_data import load_data, prepare_chunks, prepare_paragraphs, prepare_sentences, prepare_sliding_paragraphs
from question_expansion import QuestionExpansion
from rerank import cross_encoder_rerank, rrf_fusion
from clustering import hdbscan_clustering, kmeans_clustering
from mmr import mmr

from dense_retriever import DenseRetriever
from prompts import format_rag_prompt

from config import DATA_FILEPATH, MAX_CONTEXT_DOCS


class OllamaClient:
    """Client simple pour interagir avec Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_tags = f"{base_url}/api/tags"

    def is_available(self) -> bool:
        """Vérifie si Ollama est disponible"""
        try:
            response = requests.get(self.api_tags, timeout=3)
            return response.status_code == 200
        except (ConnectionError, Timeout, RequestException):
            return False

    def get_models(self) -> list:
        """Récupère la liste des modèles disponibles"""
        try:
            response = requests.get(self.api_tags, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model.get("name", "") for model in data.get("models", [])]
            return []
        except (ConnectionError, Timeout, RequestException):
            return []

    def get_default_model(self) -> Optional[str]:
        """Retourne le premier modèle disponible"""
        models = self.get_models()
        if models:
            return models[0]
        return None

    def generate(self, prompt: str, model: Optional[str] = None, think: bool = False, **options) -> Dict[str, Any]:
        """
        Génère une réponse à partir d'un prompt

        Args:
            prompt: La question/prompt à envoyer
            model: Le modèle à utiliser (auto-détecté si None)
            **options: Options additionnelles (temperature, max_tokens, etc.)

        Returns:
            Dict contenant la réponse et les métadonnées
        """
        # Auto-détection du modèle si non spécifié
        if model is None:
            model = self.get_default_model()
            if model is None:
                raise ValueError(
                    "Aucun modèle disponible. Installez un modèle avec 'ollama pull phi3:instruct'"
                )

        # Préparation du payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {
                "temperature": options.get("temperature", 0.7),
                "top_p": options.get("top_p", 0.9),
                "max_tokens": options.get("max_tokens", 1000),
                **{
                    k: v
                    for k, v in options.items()
                    if k not in ["temperature", "top_p", "max_tokens"]
                },
            },
        }

        try:
            start_time = time.time()
            response = requests.post(self.api_generate, json=payload, timeout=120)  # 2 minutes max
            generation_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model,
                    "generation_time": round(generation_time, 2),
                    "done": result.get("done", True),
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count"),
                }
            else:
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}: {response.text}",
                    "model": model,
                    "generation_time": generation_time,
                }

        except Timeout:
            return {
                "success": False,
                "error": "Timeout: Le modèle met trop de temps à répondre",
                "model": model,
                "generation_time": time.time() - start_time,
            }
        except ConnectionError:
            return {
                "success": False,
                "error": "Impossible de se connecter à Ollama. Vérifiez qu'il est démarré.",
                "model": model,
            }
        except RequestException as e:
            return {"success": False, "error": f"Erreur de requête: {str(e)}", "model": model}


def print_error(message: str):
    """Affiche un message d'erreur en rouge"""
    print(f"\033[91m❌ Erreur: {message}\033[0m", file=sys.stderr)


def print_success(message: str):
    """Affiche un message de succès en vert"""
    print(f"\033[92m✅ {message}\033[0m")


def print_info(message: str):
    """Affiche un message d'information en bleu"""
    print(f"\033[94mℹ️  {message}\033[0m")


def print_warning(message: str):
    """Affiche un message d'avertissement en jaune"""
    print(f"\033[93m⚠️  {message}\033[0m")


def format_response(result: Dict[str, Any], verbose: bool = False) -> str:
    """Formate la réponse pour l'affichage"""
    if not result["success"]:
        return f"❌ {result['error']}"

    response_text = result["response"].strip()

    if verbose:
        # Mode verbose avec métadonnées
        output = []
        output.append(f"🤖 Modèle: {result['model']}")
        output.append(f"⏱️  Temps de génération: {result['generation_time']}s")

        if result.get("prompt_eval_count"):
            output.append(f"📝 Tokens prompt: {result['prompt_eval_count']}")
        if result.get("eval_count"):
            output.append(f"🔤 Tokens générés: {result['eval_count']}")

        output.append("\n" + "=" * 50)
        output.append("💬 Réponse:")
        output.append("=" * 50)
        output.append(response_text)

        return "\n".join(output)
    else:
        # Mode simple
        return response_text


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Interroge un LLM local via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  %(prog)s "Qu'est-ce que le machine learning ?"
  %(prog)s "Explique le RAG" --model mistral:7b-instruct
  %(prog)s "Résume ce texte: ..." --verbose --temperature 0.1
  %(prog)s "Code Python pour trier une liste" --max-tokens 200
        """,
    )

    parser.add_argument("question", help="La question à poser au LLM")

    parser.add_argument("--model", "-m", help="Modèle à utiliser (auto-détecté si non spécifié)")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Affichage détaillé avec métadonnées"
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Température pour la génération (0.0-1.0, défaut: 0.7)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Nombre maximum de tokens à générer (défaut: 5000)",
    )

    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p pour la génération (défaut: 0.9)"
    )

    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        help="URL du serveur Ollama (défaut: http://localhost:11434)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Affiche la liste des modèles disponibles et quitte",
    )

    args = parser.parse_args()

    # Validation des paramètres
    if args.temperature < 0 or args.temperature > 1:
        print_error("La température doit être entre 0.0 et 1.0")
        sys.exit(1)

    if args.max_tokens < 1:
        print_error("Le nombre de tokens doit être positif")
        sys.exit(1)
    
    if args.question.strip() == "":
        print_error("La question ne peut pas être vide")
        sys.exit(1)

    # Initialisation du client
    client = OllamaClient(args.url)

    # Vérification de la disponibilité d'Ollama
    if not client.is_available():
        print_error("Ollama n'est pas accessible")
        print_info("Vérifiez qu'Ollama est démarré avec: ollama serve")
        sys.exit(1)

    # Mode liste des modèles
    if args.list_models:
        models = client.get_models()
        if models:
            print_info("Modèles disponibles:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print_warning("Aucun modèle installé")
            print_info("Installez un modèle avec: ollama pull phi3:instruct")
        sys.exit(0)

    # Vérification qu'au moins un modèle est disponible
    if not args.model:
        available_models = client.get_models()
        if not available_models:
            print_error("Aucun modèle disponible")
            print_info("Installez un modèle avec: ollama pull phi3:instruct")
            print_info("Ou listez les modèles disponibles avec: %(prog)s --list-models")
            sys.exit(1)

    # Affichage de la question (mode verbose)
    if args.verbose:
        print_info(f"Question: {args.question}")
        if args.model:
            print_info(f"Modèle: {args.model}")
        print_info("Génération en cours...")
        print()

    collection = load_data(DATA_FILEPATH, "paragraphes", data_prepare_func=prepare_paragraphs)
    collection_chunks = load_data(DATA_FILEPATH, "chunks", data_prepare_func=prepare_chunks)
    collection_sentences = load_data(DATA_FILEPATH, "sentences", data_prepare_func=prepare_sentences)

    bm25 = BM25Retriever(collection)
    bm25_chunks = BM25Retriever(collection_chunks)
    dense_retriever = DenseRetriever(collection)
    dense_chunks_retriever = DenseRetriever(collection_chunks)
    dense_sentences_retriever = DenseRetriever(collection_sentences)
    bm25_sentences = BM25Retriever(collection_sentences)

    retrievers = [
        bm25,
        # bm25_chunks,
        dense_retriever,
        # dense_chunks_retriever,
        # dense_sentences_retriever,
    ]

    sentence_retrievers = [
        bm25_sentences,
        dense_sentences_retriever,
    ]

    t0 = time.time()

    t_start_question_expansion = time.time()
    # Step -1 : Naive retrieval with sentenceretrievers on the original question
    # to seed the question expansion
    initial_contexts = []
    n_initial_context = 50
    for retriever in sentence_retrievers:
        context = retriever.retrieve(args.question, k=n_initial_context)
        initial_contexts.append(context)
    initial_context = rrf_fusion(initial_contexts, top_k=n_initial_context)


    # Step 0: Question expansion

    question_expansion = QuestionExpansion(client, args.model)
    questions = question_expansion.expand(args.question, initial_context=initial_context)
    questions = questions[:10]  # max 10 questions to avoid too many calls

    print_info(f"Temps d'expansion de la question: {(time.time() - t_start_question_expansion):.2f}s")
    print_info(f"Questions reformulées ({len(questions)}): \n- {'\n- '.join(questions)}")


    # Step 1: Retrieval
    # For each question, retrieve from each retriever
    t_start_retrieval = time.time()

    contexts = []
    context_by_question_by_retriever = {}
    for question in questions:
        context_by_question_by_retriever[question] = []
        for retriever in retrievers:
            context = retriever.retrieve(question, k=5*MAX_CONTEXT_DOCS)
            context_by_question_by_retriever[question].append(context)
            
    
    # Step 2: Fusion of all contexts
    # We now have, for each question, a list of lists of contexts (one list per retriever)
    # Flatten by retriever, rerank with rrf then cross-encoder, keep top 5*MAX_CONTEXT_DOCS
    # RRF is OK here because we are scoring the same question

    retrieve_size = 5*MAX_CONTEXT_DOCS

    for question, list_of_contexts in context_by_question_by_retriever.items():
        merged_context = rrf_fusion(list_of_contexts, top_k=retrieve_size)
        contexts.append(merged_context)
    
    # We now have a list of contexts, one per question

    context = rrf_fusion(contexts, top_k=retrieve_size)

    # We could re-rank by document_score as well to balance sources
    # context_by_source = sorted(context, key=lambda d: d["metadata"].get("document_score", 0), reverse=True)
    # context = rrf_fusion([context, context_by_source], top_k=retrieve_size)

    # We now have a single context with up to retrieve_size documents
    # It should be balanced between questions and retrievers

    print_info(f"Temps de récupération des documents: {(time.time() - t_start_retrieval):.2f}s")
    print_info(f"Nombre de documents récupérés avant filtrage: {len(context)}")

    t_start_clustering = time.time()


    # We "pre-clustered" by using different retrievers and questions
    # akin to a supervised clustering
    # Now we can cluster unsupervised to diversify the context to help outliers

    # clusters = hdbscan_clustering(collection, context, min_cluster_size=5)
    # empirically better with kmeans here
    clusters = kmeans_clustering(collection, context, n_clusters=6)

    clustered_context = []

    sorted_clusters = []
    for cluster in clusters:
        sorted_cluster = cross_encoder_rerank(cluster, args.question)
        sorted_clusters.append(sorted_cluster)

    clustered_context_sizes = [len(c) for c in sorted_clusters]
    normalized_cluster_sizes = [s / sum(clustered_context_sizes) for s in clustered_context_sizes]
    laplace_smoothing = 0.1
    normalized_cluster_sizes = [(s + laplace_smoothing) / (1 + laplace_smoothing * len(normalized_cluster_sizes)) for s in normalized_cluster_sizes]
    smoothed_cluster_sizes = [int(s * len(context)) + 1 for s in normalized_cluster_sizes]
    smoothed_clusters = [cluster[:size] for cluster, size in zip(sorted_clusters, smoothed_cluster_sizes)]
    # flatten
    smoothed_clusters = [doc for cluster in smoothed_clusters for doc in cluster]
    from random import shuffle
    shuffle(smoothed_clusters)
    
    final_pool_size = MAX_CONTEXT_DOCS * 5
    clustered_context = smoothed_clusters[:final_pool_size]
    
    context = clustered_context
    print_info(f"Temps de clustering: {(time.time() - t_start_clustering):.2f}s")

    t_start_mmr_ce_rerank = time.time()

    context = mmr(collection, context, args.question, top_k=MAX_CONTEXT_DOCS*2, relevance_weight=0.8)
    context = cross_encoder_rerank(context, args.question)[:MAX_CONTEXT_DOCS]

    print_info(f"Temps de rerank MMR + cross-encoder: {(time.time() - t_start_mmr_ce_rerank):.2f}s")
    

    context = sorted(context, key=lambda d: d["metadata"].get("document_score", 0), reverse=False)

    t_start_question_expansion = time.time()
    # Génération de la réponse
    prompt = format_rag_prompt(context, args.question)
    print("===")
    print(prompt)
    print("===")

    try:
        result = client.generate(
            prompt=prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            think=False,
        )

        # Affichage du résultat
        formatted_response = format_response(result, args.verbose)
        print_info(f"Temps de génération de la réponse: {(time.time() - t_start_question_expansion):.2f}s")
        t1 = time.time()
        print(formatted_response)
        print_info(f"Temps total écoulé: {(t1-t0):.2f}s")

        # Code de sortie
        sys.exit(0 if result["success"] else 1)

    except KeyboardInterrupt:
        print_warning("\nInterrompu par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print_error(f"Erreur inattendue: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
