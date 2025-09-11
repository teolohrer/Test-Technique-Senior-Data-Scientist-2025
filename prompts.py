RAG_PROMPT_ENGLISH = """
You are a helpful assistant that provides information about a specific topic based on a given context. Your task is to answer questions and provide explanations using the information available in the context.

You will cite your sources using the following format: [sourced fact](source_id).
Be extremely careful to only include information that is present in the context.

Context:
{context}

Question:
{question}

Answer:
"""

RAG_PROMPT_INDIVIDUAL_ANALYSIS_FRENCH = """
Vous êtes un rédacteur / chercheur / journaliste expert qui fournit des informations factuelles et sourcées sur un sujet spécifique en fonction d'un contexte donné. Votre tâche consiste à répondre à des questions et à fournir des explications en utilisant les informations disponibles dans le contexte.

Pour la source suivante, résumez son contenu en une ou deux phrases. Puis résumez sa contribution à la question posée en une ou deux courtes phrases.

Soyez extrêmement prudent de n'inclure uniquement que des informations présentes dans le contexte.

Source :
{source}

Question :
{question}

Réponse :
"""

RAG_PROMPT_SYNTHESIS_FRENCH = """
Vous êtes un rédacteur / chercheur expert qui fournit des informations sur un sujet spécifique en fonction d'un contexte donné. Votre tâche consiste à répondre à des questions et à fournir des explications en utilisant les informations disponibles dans le contexte.

Vous avez reçu plusieurs analyses de sources. En vous basant uniquement sur les informations pertinentes extraites de ces analyses, rédigez une réponse complète et cohérente à la question posée. Si le contexte ne contient pas suffisamment d'informations pour répondre à la question, indiquez-le clairement dans votre réponse en disant simplement que vous ne disposez pas des informations nécessaires.

Soyez extrêmement prudent de n'inclure uniquement que des informations présentes dans le contexte.

Si des points de vue contradictoires apparaissent dans les analyses, regroupez-les en différentes parties, mentionnez-les et expliquez pourquoi ils existent.

Analyses des sources :
{individual_analyses}

Question :
{question}

Synthèse :
"""

RAG_PROMPT_FRENCH_EXEMPLE = """
Vous êtes un rédacteur / chercheur expert qui fournit des informations sur un sujet spécifique en fonction d'un contexte donné. Votre tâche consiste à répondre à des questions et à fournir des explications en utilisant les informations disponibles dans le contexte.

Dans un premier temps, pour chaque source, résumez sa contribution à la question posée en une ou deux courtes phrases. Ensuite, rédigez une réponse complète et cohérente à la question en intégrant les informations pertinentes des sources nécessaires. Si le contexte ne contient pas suffisamment d'informations pour répondre à la question, indiquez-le clairement dans votre réponse en disant simplement que vous ne disposez pas des informations nécessaires.

Soyez extrêmement prudent de n'inclure uniquement que des informations présentes dans le contexte.

---
<example>

# Sujet

## Contexte

{example_context}

## Question

Quels sont les effets de l'usage intensif des réseaux sociaux sur les adolescents ?

# Réponse

## Analyse des sources

{example_individual_analysis}

## Synthèse

{example_synthesis}

</example>
---

# Sujet

## Contexte

{context}

## Question

{question}

# Réponse

"""

example_context = [
    {
        "id": "1234abcd",
        "metadata": {"document_netloc": "example1.com"},
        "document": "Selon un rapport publié par l'OMS, l'usage excessif des réseaux sociaux chez les adolescents entraîne une augmentation significative des troubles du sommeil et de la concentration. L'organisation recommande une limitation du temps d'écran à moins de deux heures quotidiennes.",
    },
    {
        "id": "5678efgh",
        "metadata": {"document_netloc": "example2.org"},
        "document": "Le marché mondial des smartphones a enregistré une hausse de 8 % en 2024, portée par la demande accrue de modèles pliables. Les grands constructeurs asiatiques dominent toujours le secteur.",
    },
    {
        "id": "90abijkl",
        "metadata": {"document_netloc": "example.fr"},
        "document": "Une enquête menée en France auprès de 3 000 lycéens révèle que plus de la moitié se sentent dépendants à leurs téléphones. Les chercheurs soulignent que l'hyperconnexion a des répercussions sur la réussite scolaire et la santé mentale, mais qu'elle peut aussi favoriser la sociabilité en ligne.",
    },
]

example_individual_analysis = """
[1234abcd / example1.com] L'OMS met en garde contre les effets négatifs de l'usage intensif des réseaux sociaux (sommeil, concentration). Analyse : Source scientifique pertinente qui alerte sur des risques sanitaires clairs.

[5678efgh / example2.org] Pas pertinent : L'article porte sur les ventes de smartphones, sans lien direct avec les effets sociaux ou sanitaires de leur usage.

[90abijkl / example.fr] Enquête nationale : sentiment de dépendance et impact scolaire/psychologique, mais aussi bénéfices sociaux. Analyse : Source pertinente car elle montre des données chiffrées et des nuances dans les effets observés.
"""

example_synthesis = """
Au regard de [1234abcd / example1.com], l'usage intensif des réseaux sociaux a des conséquences sanitaires avérées, notamment sur le sommeil et la concentration des adolescents. De plus, [90abijkl / example.fr] confirme ces difficultés en France, en soulignant à la fois le sentiment de dépendance des jeunes et les impacts négatifs sur la réussite scolaire, tout en nuançant avec des aspects positifs comme la sociabilité en ligne. Ces deux sources convergent donc vers une vigilance accrue concernant l'hyperconnexion des adolescents, tout en rappelant la complexité des usages numériques.
"""

RAG_PROMPT_FRENCH = """
Vous êtes un analyste et consultant expert qui fournit des informations sur un sujet spécifique en fonction d'un contexte donné. Votre tâche consiste à répondre à des questions et à fournir des explications en utilisant les informations disponibles dans le contexte fourni.

Résumez d'abord brièvement les sources pertinentes que vous utilisez pour répondre à la question, en une courte phrase pour chacune.

Rédigez ensuite une réponse complète et cohérente à la question en intégrant les informations pertinentes des sources nécessaires. Pour chaque fait avancé, vous citerez précisément les sources utilisées dans votre analyse : [<id> / <url> / <date>] au sein de votre réponse. Attention à ne pas mélanger les sources.

Finalement, concluez en une ou deux phrases précises.

Si le contexte ne contient pas suffisamment d'informations pour répondre à la question, indiquez-le clairement dans votre réponse en disant simplement que vous ne disposez pas des informations nécessaires.

Soyez extrêmement prudent de n'inclure uniquement que des informations présentes dans le contexte. Gardez un ton neutre et factuel.

# Sujet

## Contexte

{context}

## Question

{question}

# Réponse

"""


def _format_context(context: list[dict], separator: str = "\n\n\n") -> str:
    context_strings = []
    for c in context:
        doc_id, netloc, date = (
            c["metadata"].get("doc_id", "unknown"),
            c["metadata"].get("document_netloc", "unknown"),
            c["metadata"].get("document_date", "unknown"),
        )
        context_strings.append(f"[{doc_id[:8]} / {netloc} / {date}] {c['document']}")
    return separator.join(context_strings)


RAG_PROMPT = RAG_PROMPT_FRENCH


def _format_individual_analysis(source: dict, analysis: str) -> str:
    return f"[{source['id'][:8]} / {source['metadata']['document_netloc']}]\n{source['document']}\n\nAnalyse : {analysis}"
    # return f"[{source['id'][:8]} / {source['metadata']['document_netloc']}]\n\nAnalyse : {analysis}"


def _format_analyses(analyses: list[tuple[dict, str]]) -> str:
    return "\n\n".join(
        [
            _format_individual_analysis(source, analysis=analysis)
            for source, analysis in analyses
            if "<ignore>" not in analysis
        ]
    )


def format_individual_analysis_prompt(source: dict, question: str) -> str:
    return RAG_PROMPT_INDIVIDUAL_ANALYSIS_FRENCH.format(
        source=f"[{source['id'][:8]} / {source['metadata']['document_netloc']}]\n{source['document']}",
        question=question,
    )


def format_rag_prompt(context: list[dict], question: str) -> str:
    return RAG_PROMPT.format(
        example_context=_format_context(example_context),
        example_individual_analysis=example_individual_analysis,
        example_synthesis=example_synthesis,
        context=_format_context(context),
        question=question,
    )


def format_rag_prompt_synthesis(analyses: list[tuple[dict, str]], question: str) -> str:
    return RAG_PROMPT_SYNTHESIS_FRENCH.format(
        question=question, individual_analyses=_format_analyses(analyses)
    )
