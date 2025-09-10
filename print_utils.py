import sys
import time
from typing import Any, Dict

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

        output.append("💬 Réponse:")
        output.append(response_text)

        return "\n".join(output)
    else:
        # Mode simple
        return response_text

def timing_decorator(func):
    def wrapper(self, *args, **kwargs):
        config = getattr(self, 'config', None)
        verbose = config and getattr(config, 'verbose', False)
        if verbose:
            print_info(f"Starting '{func.__name__}'...")
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if verbose:
            print_info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds")
        return result
    return wrapper
