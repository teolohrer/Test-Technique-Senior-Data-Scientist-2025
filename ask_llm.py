#!/usr/bin/env python3
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

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout


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

    def generate(self, prompt: str, model: str = None, **options) -> Dict[str, Any]:
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
                    "Aucun modèle disponible. Installez un modèle avec 'ollama pull phi3:mini'"
                )

        # Préparation du payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
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
        default=1000,
        help="Nombre maximum de tokens à générer (défaut: 1000)",
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
            print_info("Installez un modèle avec: ollama pull phi3:mini")
        sys.exit(0)

    # Vérification qu'au moins un modèle est disponible
    if not args.model:
        available_models = client.get_models()
        if not available_models:
            print_error("Aucun modèle disponible")
            print_info("Installez un modèle avec: ollama pull phi3:mini")
            print_info("Ou listez les modèles disponibles avec: %(prog)s --list-models")
            sys.exit(1)

    # Affichage de la question (mode verbose)
    if args.verbose:
        print_info(f"Question: {args.question}")
        if args.model:
            print_info(f"Modèle: {args.model}")
        print_info("Génération en cours...")
        print()

    # Génération de la réponse
    try:
        result = client.generate(
            prompt=args.question,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )

        # Affichage du résultat
        formatted_response = format_response(result, args.verbose)
        print(formatted_response)

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
