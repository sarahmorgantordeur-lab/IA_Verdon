# Chatbot entreprise — RAG local avec Ollama

Chatbot conversationnel basé sur vos documents internes. Utilise une architecture RAG (Retrieval-Augmented Generation) entièrement locale via Ollama — aucune donnée n'est envoyée à un service externe.

## Fonctionnement

1. Les documents de `docs.txt` sont découpés et indexés dans une base vectorielle FAISS
2. À chaque question, les passages pertinents sont récupérés
3. Le modèle génère une réponse basée uniquement sur ces passages

## Prérequis

- Python 3.9+
- [Ollama](https://ollama.com) installé et en cours d'exécution

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requierements.txt
```

Télécharger les modèles Ollama :

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Utilisation

Ajoutez vos documents dans `docs.txt`, puis lancez :

```bash
.venv/bin/python app.py
```

```
Chatbot entreprise (llama3.2) — tapez 'quitter' pour arrêter.

Vous : Quels sont les délais de remboursement ?
Bot : Les remboursements prennent 10 jours ouvrables.
```

Tapez `quitter` pour arrêter.

## Configuration

Dans `app.py`, deux constantes permettent de changer les modèles :

| Variable | Valeur par défaut | Description |
|---|---|---|
| `EMBED_MODEL` | `nomic-embed-text` | Modèle d'embeddings |
| `CHAT_MODEL` | `llama3.2` | Modèle de génération |

Autres modèles compatibles : `mistral`, `gemma3`, `llama3.1`, etc.

## Structure

```
.
├── app.py            # Code principal du chatbot
├── docs.txt          # Base de connaissances de l'entreprise
├── requierements.txt
├── .env              # Variables d'environnement (non versionné)
└── .gitignore
```

## Sécurité

- Le fichier `.env` est exclu de git
- Les modèles tournent localement — aucune donnée ne quitte la machine
