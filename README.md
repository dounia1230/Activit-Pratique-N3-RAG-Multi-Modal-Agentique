# RAG Multi Modal Agentique

Implémentation du cours de Pr. Mohamed YOUSSFI : un PDF est découpé en textes et
images, encodé avec CLIP, indexé dans Chroma et interrogé avec un modèle local
Ollama. Un agent LangChain utilise le RAG comme outil. Streamlit fournit
l’interface ; MCP permet de séparer l’agent du processus de recherche/génération.

Le notebook pédagogique est [rag_multimodal_agentic.ipynb](rag_multimodal_agentic.ipynb).
Chaque cellule de code est précédée d’une explication. Le notebook de référence
`rag_multimodal_agentique.ipynb` est conservé uniquement en local et exclu de Git.

## Installation

Python 3.11, [uv](https://docs.astral.sh/uv/) et [Ollama](https://ollama.com/) sont
nécessaires. Depuis la racine du projet :

```powershell
uv sync --frozen
ollama pull qwen3.5:4b
ollama list
```

Démarrer Ollama depuis son application ou avec `ollama serve` si aucun serveur
n’est déjà actif. Le modèle choisi doit accepter les images pour les réponses
visuelles, et les appels d’outils pour l’agent. Le modèle par défaut est
`qwen3.5:4b`. CLIP utilise le GPU CUDA si disponible, sinon le CPU. Ses poids
`ViT-B-32 / laion2b_s34b_b79k` sont téléchargés au premier usage.

Les dépendances sont verrouillées dans `uv.lock`. Les tests et le noyau du
notebook sont inclus dans `uv sync`. Dans VS Code/Jupyter, choisir le noyau
`.venv`. Pour le rendre disponible dans une autre installation Jupyter :

```powershell
uv run python -m ipykernel install --user --name rag-multimodal --display-name "RAG multimodal (.venv)"
```

## Documents et notebook

Placer votre exemplaire du papier **Attention Is All You Need** dans
`pdfs/attention is all you need.pdf`. Le notebook utilise aussi
`architecture_transformer.webp`, présent dans le dépôt, comme image de requête.
Pour un autre document, modifier `PDF_PATH` et adapter les questions.

Exécuter les cellules dans l’ordre. La récupération fonctionne sans Ollama.
Les appels de démonstration au LLM sont regroupés derrière
`EXECUTER_EXEMPLES = False` : passer à `True` pour les exécuter, ou appeler
`rag(...)` / `interroger_agent(...)` individuellement. Les exemples couvrent les
contributions, l’attention, les positions, le masque causal, les figures, les
scores BLEU, les paramètres et une information absente du papier.

Le notebook et les applications partagent la collection
`pdf_clip_vit_b32_laion2b`. Les extractions sont rangées sous
`resultats_pdf/<empreinte_du_pdf>/`. Réindexer le même document remplace ses
éléments ; indexer un autre document conserve les précédents. La recherche porte
sur l’ensemble de la collection, avec un filtre facultatif texte/image.

## Utilisation en terminal

```powershell
uv run python main.py index "pdfs/attention is all you need.pdf"
uv run python main.py ask "What is the main contribution of the Transformer?" --type texte -k 6
uv run python main.py ask "Explain the encoder and decoder" --image architecture_transformer.webp -k 6
uv run python main.py agent "Compare multi-head attention and positional encoding" --type texte
uv run python main.py agent --image architecture_transformer.webp --type image -k 3
```

En mode RAG direct, une image est la requête CLIP ; la question reste la consigne
de génération. En mode agent, le modèle observe l’image et formule une recherche
textuelle pour son outil documentaire. La réponse directe cite `[S1]`, `[S2]`,
etc. La synthèse de l’agent cite le PDF et la page pour éviter la confusion entre
les références locales de plusieurs recherches.

## Interface Streamlit

```powershell
uv run streamlit run app.py
```

Ouvrir l’adresse affichée dans le terminal. Ajouter un PDF puis cliquer sur
**Indexer le document**. Choisir **RAG direct** ou **Agent LangChain**, le filtre
et le nombre de sources, puis poser une question et/ou joindre une image.
La réponse, les sources, les figures et les recherches de l’agent sont affichées.
Les fichiers déposés restent dans `uploads/`, exclu de Git. Arrêter avec Ctrl+C.

## Agent séparé et serveur MCP

Après l’indexation, le client suivant démarre un serveur MCP local par stdio,
découvre l’outil, puis garde la session ouverte pendant toute la boucle d’agent :

```powershell
uv run python mcp_agent.py "Compare multi-head attention and positional encoding" --type texte -k 6
uv run python mcp_agent.py --image architecture_transformer.webp --type image -k 3
```

`mcp_agent.py` n’accède pas à CLIP ou Chroma : l’outil appelle `mcp_server.py`,
qui exécute le pipeline et retourne réponse, sources et avertissements. Les
citations et les identifiants traversent MCP ; le client ne lit pas les fichiers
images du serveur. Les figures restent visibles dans le notebook et Streamlit
en mode agent local. Le transport fourni est local par stdio, sans serveur HTTP.

Pour un autre hôte MCP, utiliser l’interpréteur Python de `.venv` comme `command`,
le chemin absolu de `mcp_server.py` comme premier argument et `RAG_DATA_DIR`
comme dossier absolu des données dans l’environnement. Le serveur réserve stdout
au protocole. Sa découverte ne charge aucun poids de modèle.

## Configuration

Les variables d’environnement sont lues au démarrage. `.env.example` sert de
modèle ; le projet ne charge pas automatiquement les fichiers `.env`.

| Variable | Valeur par défaut | Usage |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Serveur Ollama du RAG et de l’agent |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Génération documentaire, texte et images |
| `OLLAMA_AGENT_MODEL` | valeur de `OLLAMA_MODEL` | Modèle de décision de l’agent |
| `RAG_DATA_DIR` | dossier courant | Racine de Chroma, des extractions et des uploads |

Exemple PowerShell :

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:OLLAMA_MODEL = "qwen3.5:4b"
$env:RAG_DATA_DIR = (Get-Location).Path
```

Pour la CLI directe, `--data-dir` se place avant la sous-commande :
`uv run python main.py --data-dir ./donnees index ./pdfs/document.pdf`.
Le client MCP possède aussi une option `--data-dir`. Le notebook pédagogique
utilise la racine courante ; y conserver les données pour partager son index.

## Vérification et limites

```powershell
uv run pytest -q
```

Les tests couvrent l’indexation répétée et de plusieurs PDF, les recherches par
modalité, les entrées invalides, la préparation des images, les sources
effectivement transmises, les retours d’outils, les citations par appel, la
limite de trois outils, l’interface Streamlit et une vraie session MCP stdio.
Ils utilisent de petits PDF temporaires et des encodeurs/modèles simulés ; ils
ne téléchargent aucun poids et ne nécessitent pas de serveur Ollama.

Les exemples de génération doivent aussi être évalués manuellement : une citation
produite par le modèle ne garantit pas que sa conclusion soit correcte. CLIP a
un contexte court et peut tronquer les requêtes longues ; la recherche en anglais
est souvent préférable avec ces poids. La similarité cosinus n’est pas une
probabilité. Les scans nécessitent un OCR externe. L’extraction récupère les
images intégrées, sans rendu des dessins vectoriels ni reconstruction des tableaux.
Changer de modèle/poids CLIP nécessite une autre collection et une réindexation.

En cas de collection vide, indexer un PDF. En cas d’erreur Ollama, vérifier
`ollama list`, le serveur et les capacités du modèle. Un filtre image peut
légitimement ne rien retrouver si le PDF ne contient pas d’images intégrées.

Références techniques : [vision Ollama](https://docs.ollama.com/capabilities/vision),
[ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama),
[middleware LangChain](https://reference.langchain.com/python/langchain/agents/middleware/tool_call_limit/ToolCallLimitMiddleware)
et [SDK Python MCP](https://github.com/modelcontextprotocol/python-sdk).
