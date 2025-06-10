# Test Technique - Système RAG

![Senior Data Scientist NLP/GenAI](https://img.shields.io/badge/Poste-Senior%20Data%20Scientist%20NLP%2FGenAI-blue)

![La Mètis](https://img.shields.io/badge/Entreprise-La%20M%C3%A8tis-green)

## 🎯 Contexte et Objectif

Ce test technique évalue votre capacité à concevoir et implémenter un **système de RAG (Retrieval-Augmented Generation)** capable de répondre à des questions complexes en se basant sur un corpus de documents spécialisés.

### Mission

Vous devez développer un système intelligent qui :
- **Indexe et structure** efficacement un corpus de documents
- **Recherche les informations pertinentes** pour répondre à une question donnée  
- **Génère des réponses fiables** en s'appuyant exclusivement sur le contenu du corpus
- **Garantit la traçabilité** des sources utilisées pour chaque réponse

Le système doit être capable de traiter des questions variées sur le **futur du secteur automobile** et de fournir des réponses documentées, précises et contextualisées.

---

## 📊 Corpus de Documents

### Description

Le corpus fourni contient des **documents en français** collectés sur Internet et traitant du **futur du secteur automobile**. Les thématiques couvertes incluent :

- 🔄 **Le ralentissement de la transition vers le tout-électrique** : face à une demande plus faible que prévu, les constructeurs freinent leurs objectifs 100% électriques et remettent en avant les motorisations hybrides comme solution de transition.
- ✂️ **Les restructurations massives pour réduire les coûts** : plusieurs constructeurs, notamment Nissan, annoncent des plans drastiques de fermetures d'usines et de suppressions d'emplois pour redevenir rentables et financer l'électrification.
- 🌍 **L'impact des tensions géopolitiques et de la concurrence chinoise** : les droits de douane américains pèsent sur la rentabilité, tandis que l'offensive des constructeurs chinois très compétitifs sur l'électrique force les acteurs traditionnels à réagir.
- 🔋 **L'avenir incertain mais prometteur des batteries** : la course aux batteries de nouvelle génération (solides, LFP) est lancée pour baisser les coûts et augmenter l'autonomie, mais les investissements sont massifs et parfois risqués.
- 🤖 **Le logiciel et la conduite autonome comme nouveaux champs de bataille** : la valeur d'une voiture se déplace vers le logiciel (SDV), menant à des partenariats stratégiques (parfois fragiles) et à une accélération des tests de conduite autonome.
- 🏭 **La délocalisation de la production automobile en Europe** : la carte industrielle se redessine, avec des sites de production délocalisés vers des pays à plus faibles coûts et un avenir incertain pour des sites historiques comme Poissy en France.
- ⚖️ **L'influence et la volatilité des politiques publiques** : les stratégies industrielles sont fortement dépendantes des décisions politiques, qu'il s'agisse de l'assouplissement des normes CO2 en Europe, des ZFE en France ou de l'incertitude des subventions aux États-Unis.

### Format des Données

Le corpus est fourni sous forme d'un fichier **CSV** (`corpus_automobile.csv`) avec la structure suivante :

| Colonne | Description |
|---------|-------------|
| `par_id` | Identifiant unique du paragraphe |
| `doc_id` | Identifiant unique du document source |
| `document_url` | URL du document original |
| `document_date` | Date de publication (peut être vide) |
| `document_netloc` | Nom de domaine source |
| `document_title` | Titre du document |
| `paragraph_text` | **Contenu textuel du paragraphe** |
| `paragraph_order` | Ordre d'apparition dans le document |
| `paragraph_score` | Score de pertinence du paragraphe |
| `document_score` | Score de pertinence du document |

> **💡 Note importante :** Les données sont déjà segmentées au niveau paragraphe, ce qui peut influencer votre stratégie d'indexation et de recherche.

### Statistiques du Corpus

- **Nombre total de paragraphes** : 1040
- **Nombre de documents sources** : 606
- **Période couverte** : articles récents (mai et juin 2025)
- **Sources** : Presse généraliste et spécialisée, Réseaux Sociaux, Sites commerciaux
- **Langue** : Français exclusivement

---

## 🎯 Livrable Attendu

### Objectif Principal

Modifier le script `ask_llm.py` fourni pour créer un **système RAG fonctionnel** qui :

1. **Charge et indexe** le corpus de documents
2. **Effectue une recherche sémantique** pour identifier les passages pertinents
3. **Génère une réponse contextuelle** en utilisant un LLM local
4. **Cite ses sources** de manière précise et vérifiable

### Script Final : `ask_llm_rag.py`

Le script modifié doit conserver l'interface en ligne de commande simple :

```bash
# Usage basique
python ask_llm_rag.py "Quels sont les défis de la voiture électrique ?"

# Avec uv
uv run ask_llm_rag.py "Comment évoluera la conduite autonome ?"

# Mode verbose pour voir les sources
python ask_llm_rag.py "Impact environnemental du véhicule électrique" --verbose
```

### Fonctionnalités Requises

#### ✅ **Core RAG Pipeline**
- Indexation du corpus avec embeddings sémantiques
- Recherche de passages pertinents pour chaque question
- Génération de réponse basée uniquement sur le contexte trouvé
- Gestion gracieuse des cas où aucune information n'est disponible

#### ✅ **Qualité et Fiabilité**
- **Fidélité aux sources** : réponses basées exclusivement sur le corpus
- **Citations précises** : références aux documents sources utilisés
- **Détection des limitations** : signaler quand l'information est insuffisante
- **Cohérence** : réponses structurées et bien articulées

#### ✅ **Performance et Robustesse**
- Temps de réponse raisonnable (< 30 secondes)
- Gestion d'erreurs appropriée
- Interface utilisateur claire et informative

---

## 🔧 Contraintes Techniques

### Modèles et Infrastructure

- **LLM** : Utilisation exclusive de modèles **open-source locaux** via Ollama
- **Embeddings** : Modèles de sentence transformers ou équivalents
- **Base vectorielle** : Au choix (ChromaDB, Faiss, Qdrant, etc.)
- **Déploiement** : Fonctionnement 100% local (pas d'API externe)

### Qualité du Code

- **Code propre** et bien structuré
- **Gestion d'erreurs** robuste
- **Documentation** des choix techniques principaux
- **Compatibilité** avec l'environnement `uv` fourni

---

## 📝 Exemples de Questions

Pour guider votre développement et vos tests, voici des questions représentatives :

### Questions Factuelles
- Combien d'usines et de postes Nissan prévoit-il de supprimer dans le cadre de son plan "Re:Nissan" ?
- Dans quelle usine espagnole Renault prévoit-il de produire trois de ses futurs SUV électriques et à partir de quand ?
- Quel est l'impact financier attendu des droits de douane américains sur le bénéfice d'exploitation de Honda pour l'exercice 2025-2026 ?

### Questions d'Analyse  
- Pour quelles raisons Honda a-t-il revu à la baisse ses investissements et ses objectifs dans les véhicules électriques ?
- Comment le changement de dirigeant chez Stellantis influence-t-il la stratégie d'électrification du groupe ?
- Pourquoi le site de Stellantis à Poissy est-il considéré comme un cas "en suspens" et quelles sont les pistes envisagées pour son avenir ?

### Questions de Synthèse
- En vous basant sur les exemples de Honda et Stellantis, expliquez les raisons qui poussent certains constructeurs à revoir leur stratégie "tout-électrique" et à se concentrer de nouveau sur les motorisations hybrides.
- Synthétisez la situation industrielle contrastée entre l'usine Stellantis de Poissy (France) et l'usine Renault de Palencia (Espagne) concernant la production de futurs véhicules.

### Questions Hors Sujet
- Quelles sont les caractéristiques techniques et le prix du dernier smartphone de Xiaomi ?
- Quelles sont les dernières avancées d'Airbus dans le domaine de l'aviation commerciale et quel est le carnet de commandes pour l'A350 ?

> **⚠️ Important :** Votre système doit être capable de répondre "Je ne trouve pas d'information sur ce sujet dans le corpus" si la question sort du périmètre du secteur automobile ou si les données sont insuffisantes.

---

## 🕒 Durée et Modalités

- **Durée estimée** : 6-8 heures
- **Format** : Travail à la maison
- **Livrable** : Script `ask_llm_rag.py` fonctionnel + documentation rapide des choix techniques

### Évaluation

Le test sera évalué selon ces critères :

1. **Fonctionnalité** : le système RAG fonctionne-t-il correctement ?
2. **Qualité des réponses** : pertinence, fidélité aux sources, citations
3. **Architecture technique** : choix techniques, optimisations, robustesse
4. **Clarté et documentation** : code lisible, choix expliqués

---

## 🚀 Premiers Pas

### 1. Vérification de l'environnement

```bash
# Tester que le LLM fonctionne
uv run test_llm.py

# Tester l'interface de base
uv run ask_llm.py "Bonjour, peux-tu me parler de voiture électrique ?"
```

### 2. Exploration du corpus

```bash
# Examiner la structure des données
head -5 data/corpus.csv
wc -l data/corpus.csv
```

### 3. Développement

À vous de jouer ! Vous avez toute la liberté pour :
- Choisir votre stratégie d'indexation
- Sélectionner vos outils de recherche sémantique  
- Optimiser la construction du contexte pour le LLM
- Implémenter la logique de citation des sources

> **💡 Conseil :** Commencez simple et itérez. Un RAG basique qui fonctionne est préférable à un système complexe incomplet.

---

## ⚖️ Licence et Usage

Ce test technique est fourni sous licence propriétaire restreinte. L'usage est limité à l'évaluation des candidats pour le poste de Senior Data Scientist NLP/GenAI chez La Mètis.

Voir le fichier [LICENCE](LICENCE) pour les conditions complètes.

---

**Bon développement !** 🚀