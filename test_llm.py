"""
Test de vérification du LLM local via Ollama (avec uv)
======================================================

Ce script vérifie que :
1. Ollama est installé et en cours d'exécution
2. Au moins un modèle est disponible
3. L'API fonctionne correctement
4. Le modèle peut générer des réponses cohérentes

Usage:
    uv run test_llm.py
    # ou
    uv sync && python test_llm.py
"""

import subprocess
import sys
import time
from typing import Dict, List

import requests
from colorama import Fore, Style, init
from tabulate import tabulate

# Initialisation des couleurs
init(autoreset=True)


class OllamaLLMTester:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_tags = f"{base_url}/api/tags"
        self.api_show = f"{base_url}/api/show"
        self.available_models = []

    def print_status(self, message: str, status: str = "info"):
        """Affiche un message avec couleur selon le statut"""
        colors = {
            "success": Fore.GREEN + "✅ ",
            "error": Fore.RED + "❌ ",
            "warning": Fore.YELLOW + "⚠️  ",
            "info": Fore.BLUE + "ℹ️  ",
        }
        print(f"{colors.get(status, '')}{message}{Style.RESET_ALL}")

    def print_header(self, title: str):
        """Affiche un en-tête de section"""
        print(f"\n{Fore.CYAN}{'=' * 50}")
        print(f"{title.center(50)}")
        print(f"{'=' * 50}{Style.RESET_ALL}\n")

    def check_uv_installation(self) -> bool:
        """Test préliminaire: Vérifier que uv est installé"""
        self.print_header("TEST PRÉLIMINAIRE: UV")

        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status(f"uv installé: {version}", "success")

                # Vérifier que les dépendances sont installées
                try:
                    import colorama
                    import requests
                    import tabulate

                    self.print_status("Toutes les dépendances sont disponibles", "success")
                    return True
                except ImportError as e:
                    self.print_status(f"Dépendance manquante: {e}", "warning")
                    self.print_status("Exécutez: uv sync", "info")
                    return True  # uv est installé même si les deps manquent

            else:
                self.print_status("uv non fonctionnel", "error")
                return False

        except FileNotFoundError:
            self.print_status("uv n'est pas installé", "error")
            self.print_status(
                "Installation: curl -LsSf https://astral.sh/uv/install.sh | sh", "info"
            )
            return False
        except subprocess.TimeoutExpired:
            self.print_status("Timeout lors de la vérification uv", "error")
            return False
        except Exception as e:
            self.print_status(f"Erreur lors de la vérification uv: {str(e)}", "error")
            return False

    def test_ollama_connection(self) -> bool:
        """Test 1: Vérifier la connexion à Ollama"""
        self.print_header("TEST 1: CONNEXION OLLAMA")

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.print_status("Ollama est en cours d'exécution", "success")
                return True
            else:
                self.print_status(f"Ollama répond avec le code {response.status_code}", "error")
                return False
        except requests.exceptions.ConnectionError:
            self.print_status("Impossible de se connecter à Ollama", "error")
            self.print_status("Vérifiez qu'Ollama est démarré : 'ollama serve'", "warning")
            return False
        except requests.exceptions.Timeout:
            self.print_status("Timeout lors de la connexion à Ollama", "error")
            return False
        except Exception as e:
            self.print_status(f"Erreur de connexion : {str(e)}", "error")
            return False

    def get_available_models(self) -> List[Dict]:
        """Test 2: Lister les modèles disponibles"""
        self.print_header("TEST 2: MODÈLES DISPONIBLES")

        try:
            response = requests.get(self.api_tags, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                if not models:
                    self.print_status("Aucun modèle installé", "warning")
                    self.print_status("Installez un modèle : 'ollama pull phi3:instruct'", "info")
                    return []

                # Affichage tableau des modèles
                model_table = []
                for model in models:
                    name = model.get("name", "Unknown")
                    size = model.get("size", 0)
                    size_gb = round(size / (1024**3), 2) if size else 0
                    modified = model.get("modified_at", "Unknown")

                    model_table.append([name, f"{size_gb} GB", modified])
                    self.available_models.append(name)

                print(
                    tabulate(model_table, headers=["Modèle", "Taille", "Modifié"], tablefmt="grid")
                )

                self.print_status(f"{len(models)} modèle(s) disponible(s)", "success")
                return models

            else:
                self.print_status(
                    f"Erreur lors de la récupération des modèles: {response.status_code}", "error"
                )
                return []

        except Exception as e:
            self.print_status(f"Erreur lors de la récupération des modèles: {str(e)}", "error")
            return []

    def test_model_generation(self, model_name: str) -> bool:
        """Test 3: Tester la génération avec un modèle"""
        self.print_header(f"TEST 3: GÉNÉRATION AVEC {model_name}")

        test_prompts = [
            {
                "prompt": "Bonjour, peux-tu me dire ton nom en une phrase ?",
                "description": "Test de base",
                "expected_keywords": ["bonjour", "nom", "assistant", "ia"],
            },
            {
                "prompt": "Qu'est-ce que le machine learning ? Réponds en maximum 2 phrases.",
                "description": "Test de connaissance",
                "expected_keywords": ["machine learning", "apprentissage", "données", "algorithme"],
            },
            {
                "prompt": "Résume ce texte : 'Le RAG (Retrieval-Augmented Generation) combine la recherche de documents pertinents avec la génération de texte par un modèle de langage.'",
                "description": "Test de résumé",
                "expected_keywords": ["rag", "recherche", "génération", "documents"],
            },
        ]

        success_count = 0

        for i, test in enumerate(test_prompts, 1):
            self.print_status(f"Test {i}/3: {test['description']}", "info")
            print(f"   Prompt: {test['prompt'][:60]}...")

            try:
                start_time = time.time()

                payload = {
                    "model": model_name,
                    "prompt": test["prompt"],
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 150},
                }

                response = requests.post(self.api_generate, json=payload, timeout=60)

                generation_time = round(time.time() - start_time, 2)

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "").strip()

                    if generated_text:
                        print(
                            f"   {Fore.GREEN}✅ Réponse générée en {generation_time}s{Style.RESET_ALL}"
                        )
                        print(f"   Réponse: {generated_text[:100]}...")

                        # Vérification basique de la cohérence
                        text_lower = generated_text.lower()
                        found_keywords = [
                            kw for kw in test["expected_keywords"] if kw.lower() in text_lower
                        ]

                        if found_keywords:
                            print(
                                f"   {Fore.GREEN}✅ Mots-clés trouvés: {', '.join(found_keywords)}{Style.RESET_ALL}"
                            )
                            success_count += 1
                        else:
                            print(
                                f"   {Fore.YELLOW}⚠️  Réponse peu pertinente (mots-clés manquants){Style.RESET_ALL}"
                            )
                    else:
                        print(f"   {Fore.RED}❌ Réponse vide{Style.RESET_ALL}")
                else:
                    print(f"   {Fore.RED}❌ Erreur HTTP {response.status_code}{Style.RESET_ALL}")

            except requests.exceptions.Timeout:
                print(f"   {Fore.RED}❌ Timeout (>60s){Style.RESET_ALL}")
            except Exception as e:
                print(f"   {Fore.RED}❌ Erreur: {str(e)}{Style.RESET_ALL}")

            print()  # Ligne vide entre les tests

        success_rate = (success_count / len(test_prompts)) * 100

        if success_rate >= 66:
            self.print_status(
                f"Modèle fonctionnel ({success_count}/{len(test_prompts)} tests réussis)", "success"
            )
            return True
        else:
            self.print_status(
                f"Modèle partiellement fonctionnel ({success_count}/{len(test_prompts)} tests réussis)",
                "warning",
            )
            return False

    def test_performance(self, model_name: str) -> Dict:
        """Test 4: Mesurer les performances"""
        self.print_header(f"TEST 4: PERFORMANCES DE {model_name}")

        performance_results = {"latency_avg": 0, "throughput": 0, "success_rate": 0}

        test_prompt = "Explique le concept de RAG en une phrase."
        num_tests = 3
        successful_requests = 0
        total_time = 0
        total_tokens = 0

        self.print_status(
            f"Exécution de {num_tests} requêtes pour mesurer les performances...", "info"
        )

        for i in range(num_tests):
            try:
                start_time = time.time()

                payload = {
                    "model": model_name,
                    "prompt": f"{test_prompt} (Test {i + 1})",
                    "stream": False,
                }

                response = requests.post(self.api_generate, json=payload, timeout=30)

                if response.status_code == 200:
                    request_time = time.time() - start_time
                    total_time += request_time
                    successful_requests += 1

                    result = response.json()
                    response_text = result.get("response", "")
                    total_tokens += len(response_text.split())

                    print(f"   Test {i + 1}: {request_time:.2f}s ✅")
                else:
                    print(f"   Test {i + 1}: Erreur HTTP {response.status_code} ❌")

            except Exception as e:
                print(f"   Test {i + 1}: Erreur {str(e)} ❌")

        if successful_requests > 0:
            avg_latency = total_time / successful_requests
            avg_throughput = total_tokens / total_time if total_time > 0 else 0
            success_rate = (successful_requests / num_tests) * 100

            performance_results = {
                "latency_avg": round(avg_latency, 2),
                "throughput": round(avg_throughput, 1),
                "success_rate": round(success_rate, 1),
            }

            # Affichage résultats
            perf_table = [
                ["Latence moyenne", f"{avg_latency:.2f} secondes"],
                ["Débit moyen", f"{avg_throughput:.1f} tokens/sec"],
                ["Taux de succès", f"{success_rate:.1f}%"],
            ]

            print("\n" + tabulate(perf_table, headers=["Métrique", "Valeur"], tablefmt="grid"))

            # Évaluation des performances
            if avg_latency < 5 and success_rate >= 90:
                self.print_status("Performances excellentes", "success")
            elif avg_latency < 15 and success_rate >= 70:
                self.print_status("Performances acceptables", "warning")
            else:
                self.print_status("Performances insuffisantes", "error")
        else:
            self.print_status("Impossible de mesurer les performances", "error")

        return performance_results

    def generate_uv_usage_example(self):
        """Génère un exemple d'usage avec uv"""
        self.print_header("EXEMPLE D'USAGE AVEC UV")

        example_code = f'''
# test_rag_simple.py
import requests

def query_llm(prompt, model="{self.available_models[0] if self.available_models else "phi3:instruct"}"):
    """Interroge le LLM local via Ollama"""
    response = requests.post('http://localhost:11434/api/generate',
                           json={{'model': model, 'prompt': prompt, 'stream': False}})
    return response.json()['response']

# Test simple
if __name__ == "__main__":
    question = "Qu'est-ce que le RAG ?"
    reponse = query_llm(question)
    print(f"Q: {{question}}")
    print(f"R: {{reponse}}")
'''

        print(f"{Fore.CYAN}Créez un fichier test_rag_simple.py :{Style.RESET_ALL}")
        print(example_code)

        print(f"{Fore.CYAN}Exécution avec uv :{Style.RESET_ALL}")
        print(f"{Fore.GREEN}uv run test_rag_simple.py{Style.RESET_ALL}")
        print(f"# ou")
        print(f"{Fore.GREEN}uv sync && python test_rag_simple.py{Style.RESET_ALL}")

    def run_full_test(self) -> bool:
        """Exécute tous les tests"""
        print(f"{Fore.MAGENTA}{'=' * 60}")
        print("🧪 TEST COMPLET DU LLM LOCAL (OLLAMA + UV)".center(60))
        print(f"{'=' * 60}{Style.RESET_ALL}")

        # Test préliminaire: uv
        if not self.check_uv_installation():
            self.print_status("Tests partiels - uv non disponible", "warning")

        # Test 1: Connexion
        if not self.test_ollama_connection():
            self.print_status("Tests interrompus - Ollama non accessible", "error")
            return False

        # Test 2: Modèles disponibles
        models = self.get_available_models()
        if not models:
            self.print_status("Tests interrompus - Aucun modèle disponible", "error")
            return False

        # Test 3 & 4: Test du premier modèle disponible
        test_model = self.available_models[0]
        self.print_status(f"Test avec le modèle: {test_model}", "info")

        generation_success = self.test_model_generation(test_model)
        performance_results = self.test_performance(test_model)

        # Génération exemple d'usage
        self.generate_uv_usage_example()

        # Résumé final
        self.print_header("RÉSUMÉ FINAL")

        if generation_success and performance_results["success_rate"] >= 70:
            self.print_status("🎉 Environnement LLM PRÊT pour le test technique RAG !", "success")
            self.print_status(f"Modèle recommandé: {test_model}", "info")
            self.print_status("Gestionnaire de paquets: uv ⚡", "info")
            return True
        else:
            self.print_status("⚠️ Environnement partiellement fonctionnel", "warning")
            self.print_status(
                "Le test technique RAG reste possible mais avec des limitations", "warning"
            )
            return False


def main():
    """Point d'entrée principal"""
    tester = OllamaLLMTester()

    try:
        success = tester.run_full_test()

        if success:
            print(
                f"\n{Fore.GREEN}🚀 Vous pouvez maintenant commencer le développement RAG avec uv !{Style.RESET_ALL}"
            )
            print(f"\n{Fore.CYAN}Commandes uv utiles :{Style.RESET_ALL}")
            print(
                f"{Fore.GREEN}uv add sentence-transformers  {Style.RESET_ALL}# Ajouter une dépendance"
            )
            print(
                f"{Fore.GREEN}uv run script.py              {Style.RESET_ALL}# Exécuter un script"
            )
            print(
                f"{Fore.GREEN}uv sync                       {Style.RESET_ALL}# Synchroniser les dépendances"
            )
            print(
                f"{Fore.GREEN}uv lock                       {Style.RESET_ALL}# Verrouiller les versions"
            )
        else:
            print(
                f"\n{Fore.YELLOW}⚠️ Vérifiez la configuration avant de commencer le test technique{Style.RESET_ALL}"
            )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrompu par l'utilisateur{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Erreur inattendue: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
