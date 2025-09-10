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
from typing import Any, Dict

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import RAGConfig
from pipeline import RAGPipeline


from ollama_client import OllamaClient
from print_utils import print_error, print_info, print_warning

import defaults



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
        default=defaults.TEMPERATURE,
        help=f"Température pour la génération (0.0-1.0, défaut: {defaults.TEMPERATURE})",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.MAX_TOKENS,
        help=f"Nombre maximum de tokens à générer (défaut: {defaults.MAX_TOKENS})",
    )

    parser.add_argument(
        "--top-p", type=float, default=defaults.TOP_P, help=f"Top-p pour la génération (défaut: {defaults.TOP_P})"
    )

    parser.add_argument(
        "--url",
        default=defaults.OLLAMA_BASE_URL,
        help=f"URL du serveur Ollama (défaut: {defaults.OLLAMA_BASE_URL})",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Affiche la liste des modèles disponibles et quitte",
    )

    # RAG Configuration Parameters
    parser.add_argument(
        "--max-context-docs",
        type=int,
        default=defaults.MAX_CONTEXT_DOCS,
        help=f"Nombre maximum de documents dans le contexte final (défaut: {defaults.MAX_CONTEXT_DOCS})",
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=defaults.MAX_EXPANDED_QUESTIONS,
        help=f"Nombre maximum de questions reformulées (défaut: {defaults.MAX_EXPANDED_QUESTIONS})",
    )

    parser.add_argument(
        "--mmr-relevance-weight",
        type=float,
        default=defaults.MMR_RELEVANCE_WEIGHT,
        help=f"Poids de pertinence pour MMR (0.0-1.0, défaut: {defaults.MMR_RELEVANCE_WEIGHT})",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=defaults.KMEANS_N_CLUSTERS,
        help=f"Nombre de clusters pour le clustering k-means (défaut: {defaults.KMEANS_N_CLUSTERS})",
    )

    parser.add_argument(
        "--disable-clustering",
        action="store_true",
        help="Désactive le clustering des documents",
    )

    parser.add_argument(
        "--disable-mmr",
        action="store_true",
        help="Désactive la diversification MMR",
    )

    parser.add_argument(
        "--disable-question-expansion",
        action="store_true",
        help="Désactive l'expansion de questions",
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

    if args.max_context_docs < 1:
        print_error("Le nombre maximum de documents doit être positif")
        sys.exit(1)

    if args.max_questions < 1:
        print_error("Le nombre maximum de questions doit être positif")
        sys.exit(1)

    if args.mmr_relevance_weight < 0 or args.mmr_relevance_weight > 1:
        print_error("Le poids de pertinence MMR doit être entre 0.0 et 1.0")
        sys.exit(1)

    if args.n_clusters < 1:
        print_error("Le nombre de clusters doit être positif")
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
            print_info(f"Installez un modèle avec: ollama pull {defaults.GENERATION_MODEL}")
        sys.exit(0)

    # Vérification qu'au moins un modèle est disponible
    if not args.model:
        available_models = client.get_models()
        if not available_models:
            print_error("Aucun modèle disponible")
            print_info(f"Installez un modèle avec: ollama pull {defaults.GENERATION_MODEL}")
            print_info("Ou listez les modèles disponibles avec: %(prog)s --list-models")
            sys.exit(1)

    # Affichage de la question (mode verbose)
    if args.verbose:
        print_info(f"Question: {args.question}")
        if args.model:
            print_info(f"Modèle: {args.model}")
        print_info("Génération en cours...")
        print()
    
    config = RAGConfig.from_args(args)
    config.client = client
    pipeline = RAGPipeline(config)
    pipeline.load(defaults.DATA_FILEPATH)
    answer = pipeline.run(args.question)
    print(answer)


if __name__ == "__main__":
    main()
