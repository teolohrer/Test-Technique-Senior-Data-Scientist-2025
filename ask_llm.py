"""
Script pour interroger un LLM local via Ollama avec configuration YAML
=====================================================================

Usage:
    python ask_llm.py "Qu'est-ce que le machine learning ?"
    python ask_llm.py "Explique le RAG" --config config.yaml --model mistral:7b-instruct
    python ask_llm.py "Bonjour" --verbose --temperature 0.1

Ou avec uv:
    uv run ask_llm.py "Ma question"
"""

import argparse
import sys
from typing import Any, Dict

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from config_manager import load_config, Config
from pipeline import RAGPipeline
from ollama_client import OllamaClient
from print_utils import print_error, print_info, print_warning

import defaults


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Interroge un LLM local via Ollama avec support de configuration YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  %(prog)s "Qu'est-ce que le machine learning ?"
  %(prog)s "Explique le RAG" --config config.yaml --model mistral:7b-instruct
  %(prog)s "Résume ce texte: ..." --verbose --temperature 0.1 --max-tokens 200

Configuration:
  Utilisez --config pour charger un fichier YAML avec vos paramètres préférés.
  Les arguments de ligne de commande prennent priorité sur la configuration YAML.
  Générez un fichier d'exemple avec: python config_manager.py
        """,
    )

    # Required argument
    parser.add_argument("question", help="La question à poser au LLM")

    # Configuration file
    parser.add_argument("--config", "-c", help="Fichier de configuration YAML")

    # CLI-only parameters
    parser.add_argument("--model", "-m", help="Modèle à utiliser (auto-détecté si non spécifié)")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Affiche la liste des modèles disponibles et quitte",
    )

    # Override any configuration parameter via CLI
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Affichage détaillé avec métadonnées"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, help=f"Température pour la génération (0.0-1.0)"
    )
    parser.add_argument("--max-tokens", type=int, help="Nombre maximum de tokens à générer")
    parser.add_argument("--top-p", type=float, help="Top-p pour la génération")
    parser.add_argument("--ollama-base-url", help="URL du serveur Ollama")
    parser.add_argument(
        "--max-context-docs", type=int, help="Nombre maximum de documents dans le contexte final"
    )
    parser.add_argument(
        "--max-expanded-questions", type=int, help="Nombre maximum de questions reformulées"
    )
    parser.add_argument(
        "--mmr-relevance-weight", type=float, help="Poids de pertinence pour MMR (0.0-1.0)"
    )
    parser.add_argument(
        "--kmeans-n-clusters", type=int, help="Nombre de clusters pour le clustering k-means"
    )

    return parser


def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Convert args to dict, handling dash/underscore conversion
        args_dict = vars(args)
        # Convert CLI argument names to match config field names
        if "ollama_base_url" not in args_dict and "ollama-base-url" in args_dict:
            args_dict["ollama_base_url"] = args_dict.pop("ollama-base-url", None)
        if "max_context_docs" not in args_dict and "max-context-docs" in args_dict:
            args_dict["max_context_docs"] = args_dict.pop("max-context-docs", None)
        if "max_expanded_questions" not in args_dict and "max-expanded-questions" in args_dict:
            args_dict["max_expanded_questions"] = args_dict.pop("max-expanded-questions", None)
        if "mmr_relevance_weight" not in args_dict and "mmr-relevance-weight" in args_dict:
            args_dict["mmr_relevance_weight"] = args_dict.pop("mmr-relevance-weight", None)
        if "kmeans_n_clusters" not in args_dict and "kmeans-n-clusters" in args_dict:
            args_dict["kmeans_n_clusters"] = args_dict.pop("kmeans-n-clusters", None)
        if "list_models" not in args_dict and "list-models" in args_dict:
            args_dict["list_models"] = args_dict.pop("list-models", None)
        
        if not args.verbose:
            # pop verbose if not set in CLI to avoid overriding config
            args_dict.pop("verbose", None)

        # Load configuration (YAML + CLI overrides)
        config = load_config(yaml_file=args.config, cli_args=args_dict)
    except Exception as e:
        print_error(f"Erreur de configuration: {e}")
        sys.exit(1)

    # Validation des paramètres critiques (Pydantic handles most validation)
    if not args.question.strip():
        print_error("La question ne peut pas être vide")
        sys.exit(1)

    # Initialisation du client Ollama
    client = OllamaClient(config.ollama_base_url)

    # Vérification de la disponibilité d'Ollama
    if not client.is_available():
        print_error("Ollama n'est pas accessible")
        print_info("Vérifiez qu'Ollama est démarré avec: ollama serve")
        sys.exit(1)

    # Mode liste des modèles
    if config.list_models:
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
    if not config.model:
        available_models = client.get_models()
        if not available_models:
            print_error("Aucun modèle disponible")
            print_info(f"Installez un modèle avec: ollama pull {defaults.GENERATION_MODEL}")
            sys.exit(1)

    # Override generation model if specified in CLI
    if config.model:
        config.generation_model = config.model

    # Affichage de la question (mode verbose)
    if config.verbose:
        print_info(f"Question: {args.question}")
        print_info(f"Configuration: {args.config or 'defaults'}")
        if config.model:
            print_info(f"Modèle: {config.model}")
        print_info("Génération en cours...")
        print()

    # Set the client directly in config
    config.client = client

    # Exécution du pipeline RAG
    pipeline = RAGPipeline(config)
    pipeline.load(config.data_filepath)
    answer = pipeline.run(args.question)
    print(answer)


if __name__ == "__main__":
    main()
