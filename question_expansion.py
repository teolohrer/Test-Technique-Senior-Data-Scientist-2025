import json
import numpy as np
import re
from typing import Any

from chromadb import Collection
from defaults import (
    QUESTION_EXPANSION_TEMPERATURE,
    QUESTION_EXPANSION_MAX_TOKENS,
    QUESTION_EXPANSION_TOP_P,
    QUESTION_EXPANSION_TRIES,
)
from chromadb.utils import embedding_functions
from config_manager import Config

QUESTION_REFORMULATION_PROMPT = """
Vous êtes un assistant d'extraction de mots-clés à partir de questions pour un système de RAG, en fonction d'un contexte donné. Un utilisateur vous fournit une question qui peut être vague, ambiguë, ou possèdant de mutliples sous-questions. À partir d'un contexte fourni, votre tâche est de diviser cette question en sous requêtes claires, précises et spécifiques, adaptées à une base de connaissances. Ces reformulations ciblent chacune une sous-partie différente et précise de la question originale en relation avec le contexte. Ces requêtes ne sont pas des questions, mais des phrases ou des expressions. 

Répondez sous la forme d'une liste de requêtes de recherches précises, 5-8 mots-clés et synonymes, pas de questions, en français, au format JSON, ordonnées par pertinence.

Ne produisez pas plus de requêtes que nécessaire. Chaque requête doit pouvoir apporter une information pertinente et éclairante sur la question originale.

Contexte :

{context}

Question originale :

{question}

Requêtes de mots-clés, en liste JSON :
"""


class QuestionExpansion:
    def __init__(self, client: Any, model, embedding_model, temperature=0.9, tries=3):
        self.client = client
        self.model = model
        self.tries = tries
        self.embedding_model = embedding_model
        self.temperature = temperature

    @staticmethod
    def from_config(config: Config):
        embedding_model = embedding_functions.OllamaEmbeddingFunction(
            model_name=config.embedding_model,
            url=config.ollama_base_url,
        )
        return QuestionExpansion(
            client=config.client,
            model=config.question_expansion_model,
            embedding_model=embedding_model,
            temperature=config.question_expansion_temperature,
            tries=QUESTION_EXPANSION_TRIES,
        )

    def format_context(self, contexts: list[dict]) -> str:
        return "\n\n".join([f"[{c['id'][:8]}] {c['document']}" for c in contexts])

    def centroid_embedding_query(self, collection: Collection, question: str, lambda_: float=0.5, top_k: int = 5):
        question_embedding = self.embedding_model.embed_query(question)
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
        )
        if not results or "embeddings" not in results or not results["embeddings"]:
            return question_embedding
        doc_embeddings = np.array(results["embeddings"][0])
        centroid = np.mean(doc_embeddings, axis=0)

        tuned_question_embedding = (1 - lambda_) * question_embedding + lambda_ * centroid

        results = collection.query(
            query_embeddings=[tuned_question_embedding],
            n_results=top_k,
        )

        if results and "documents" in results and results["documents"]:
            return results["documents"][0]

    def expand(self, question: str, initial_context: list[dict]) -> list[str]:
        reformulations_prompt = QUESTION_REFORMULATION_PROMPT.format(
            question=question, context=self.format_context(initial_context)
        )
        tries = self.tries
        while tries > 0:
            try:
                reformulation_result = self.client.generate(
                    prompt=reformulations_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=QUESTION_EXPANSION_MAX_TOKENS,
                    top_p=QUESTION_EXPANSION_TOP_P,
                    think=False,
                )

                # use a regex to remove any non-json content before or after the json list
                regex = r"(\[.*\])"
                if "response" not in reformulation_result:
                    print("La réponse de reformulation ne contient pas de champ 'response'")
                    raise ValueError(
                        "La réponse de reformulation ne contient pas de champ 'response'"
                    )
                match = re.search(
                    regex, reformulation_result["response"].replace("\n", ""), re.DOTALL
                )
                if not match:
                    print("Aucune liste JSON trouvée dans la réponse, regex failed")
                    raise ValueError("Aucune liste JSON trouvée dans la réponse")

                stripped_response = match.group(1)

                additional_questions = json.loads(stripped_response)

                if not isinstance(additional_questions, list):
                    raise ValueError("Les reformulations ne sont pas une liste")
                if not all(isinstance(q, str) for q in additional_questions):
                    raise ValueError(
                        "Toutes les reformulations ne sont pas des chaînes de caractères"
                    )

                questions = [question] + additional_questions
                break
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Erreur de décodage JSON des reformulations: {str(e)}")
                questions = [question]
        return questions
