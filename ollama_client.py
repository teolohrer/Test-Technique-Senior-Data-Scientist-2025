import time
from typing import Any, Dict, Optional

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


import requests
from requests.exceptions import ConnectionError, RequestException, Timeout


from defaults import (
    GENERATION_MODEL,
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    OLLAMA_CONNECTION_TIMEOUT,
    TEMPERATURE,
    TOP_P,
)


class OllamaClient:
    """Client simple pour interagir avec Ollama"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_tags = f"{base_url}/api/tags"

    def is_available(self) -> bool:
        """Vérifie si Ollama est disponible"""
        try:
            response = requests.get(self.api_tags, timeout=OLLAMA_CONNECTION_TIMEOUT)
            return response.status_code == 200
        except (ConnectionError, Timeout, RequestException):
            return False

    def get_models(self) -> list:
        """Récupère la liste des modèles disponibles"""
        try:
            response = requests.get(self.api_tags, timeout=OLLAMA_CONNECTION_TIMEOUT)
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

    def generate(
        self, prompt: str, model: Optional[str] = None, think: bool = False, **options
    ) -> Dict[str, Any]:
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
                    f"Aucun modèle disponible. Installez un modèle avec 'ollama pull {GENERATION_MODEL}'"
                )

        # Préparation du payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {
                "temperature": options.get("temperature", TEMPERATURE),
                "top_p": options.get("top_p", TOP_P),
                "max_tokens": options.get("max_tokens", MAX_TOKENS),
                **{
                    k: v
                    for k, v in options.items()
                    if k not in ["temperature", "top_p", "max_tokens"]
                },
            },
        }

        try:
            start_time = time.time()
            response = requests.post(self.api_generate, json=payload, timeout=OLLAMA_TIMEOUT)
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
