#!/bin/bash
echo "🚀 Installation environnement LLM..."

# Installation uv (si pas déjà installé)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Installation Ollama si pas déjà installé
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Téléchargement modèle
ollama pull phi3:mini  # Modèle léger 2.3GB

uv sync

echo "✅ Setup terminé ! Lancez : uv run test_llm.py"
