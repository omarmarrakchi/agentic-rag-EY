# Agentic RAG — Indexation et Recherche de TdRs (EY)

Système de RAG agentique pour l'indexation et la recherche intelligente dans un corpus de Termes de Référence (TdRs) issus des marchés publics.

---

## Objectif du projet

Construire un pipeline complet capable de :
1. **Filtrer** automatiquement un corpus de ~100 PDFs pour n'en garder que les TdRs réels
2. **Extraire et indexer** le contenu de ces TdRs dans une base vectorielle
3. **Répondre** à des questions en langage naturel via un agent RAG (architecture ReAct)
4. **Exposer** l'interface via une API FastAPI et une UI React avec mode Chat et mode Recherche

---

## Architecture globale

```
data/
  raw_tdrs/          ← PDFs bruts (100 fichiers, mix TdRs + non-TdRs)
  filtered_tdrs/     ← TdRs confirmés (sortie Step 1)
  rejected/          ← fichiers non-TdR (sortie Step 1)
  chunks/            ← chunks JSON par TdR (sortie Step 2)
  vector_db/         ← base vectorielle ChromaDB (sortie Step 3)

backend/
  config/
    settings.py      ← paramètres centralisés (seuils, chemins, modèles, provider)
  pipeline/
    step1_filter/    ← filtrage et classification des PDFs
    step2_chunking/  ← extraction, métadonnées et chunking des TdRs
    step3_indexing/  ← embeddings et indexation dans ChromaDB
    step4_agent/     ← agent RAG ReAct (LangGraph + Ollama/OpenAI)
  api/
    routes/          ← endpoints FastAPI (ask, search, provider, health)
  logs/              ← rapports JSON des runs

frontend/            ← UI React (Chat + Recherche)
.env                 ← clés API (ignoré par git)
```

---

## Étape 1 — Filtrage des PDFs

### Problème

Le corpus de 100 PDFs contient des fichiers très hétérogènes : vrais TdRs, manuels utilisateurs, rapports d'avancement, catalogues, présentations, fichiers scannés (image), documents en français et en anglais. Il faut séparer les vrais TdRs des autres avant toute indexation.

### Approche hybride : scoring + LLM

Le filtrage utilise 3 étapes en cascade :

```
PDF → [Lecture + OCR] → [Scoring mots-clés] → TdR confirmé ?
                                              ↓ non (ambigu)
                                         [Ollama LLM]
                                              ↓
                                     verdict final (TdR / rejeté)
```

**Pourquoi cette approche ?**
- Le scoring par mots-clés est rapide et déterministe — il traite ~80 % des cas sans LLM
- Ollama n'est sollicité que pour les cas ambigus (score entre 1 et 4), ce qui réduit la latence globale
- Le LLM local (pas d'API externe) garantit la confidentialité des documents EY

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/config/settings.py](backend/config/settings.py) | Tous les paramètres configurables (seuils, mots-clés, chemins, modèle LLM) |
| [backend/pipeline/step1_filter/pdf_reader.py](backend/pipeline/step1_filter/pdf_reader.py) | Lecture PDF natif ou OCR pour les scannés |
| [backend/pipeline/step1_filter/keyword_scorer.py](backend/pipeline/step1_filter/keyword_scorer.py) | Calcul du score TdR par mots-clés |
| [backend/pipeline/step1_filter/llm_classifier.py](backend/pipeline/step1_filter/llm_classifier.py) | Classification Ollama pour les cas ambigus |
| [backend/pipeline/step1_filter/filter_pipeline.py](backend/pipeline/step1_filter/filter_pipeline.py) | Orchestrateur principal |
| [backend/run_filter.py](backend/run_filter.py) | Point d'entrée CLI |

### Détail des sous-étapes

#### 1. Lecture PDF (`pdf_reader.py`)

- **PDF natif** (texte sélectionnable) : extraction directe via PyMuPDF (`fitz`)
- **PDF scanné** (image) : détection automatique si moins de 100 caractères natifs, puis OCR via Tesseract v5 + pytesseract
- L'OCR analyse les **3 premières pages** (`OCR_MAX_PAGES=3`) à 300 DPI pour éviter le cas où la page de couverture est pauvre en texte
- Les 5 000 premiers caractères sont extraits (`TEXT_EXTRACTION_CHARS=5000`)

#### 2. Scoring mots-clés (`keyword_scorer.py`)

Chaque texte extrait est comparé à trois listes de mots-clés :

| Catégorie | Exemples | Poids |
|---|---|---|
| **Fort** (`KEYWORDS_STRONG`) | "termes de référence", "terms of reference", "scope of work", "profil du consultant" | +3 |
| **Faible** (`KEYWORDS_WEAK`) | "consultant", "livrables", "méthodologie", "deliverables", "procurement" | +1 |
| **Exclusion** (`KEYWORDS_EXCLUSION`) | "manuel utilisateur", "catalogue", "rapport d'avancement", "règlement intérieur" | -3 |

**Seuils de décision** :
- Score ≥ 5 → **TdR confirmé** (pas besoin du LLM)
- Score ≤ 0 → **Rejeté** (pas besoin du LLM)
- Score 1–4 → **Ambigu** → passage à Ollama

#### 3. Classification LLM (`llm_classifier.py`)

Pour les cas ambigus, les 500 premiers caractères du texte sont envoyés au modèle `qwen2.5:14b-instruct-q3_K_M` via Ollama. Le modèle répond en JSON strict :

```json
{"is_tdr": true, "reason": "Le document décrit une mission de consultance avec livrables et profil requis"}
```

### Résultats (Run final)

Sur 100 PDFs bruts :
- **58 TdRs confirmés** → `data/filtered_tdrs/`
- **41 rejetés** → `data/rejected/`
- **1 erreur** (fichier corrompu)
- Ollama sollicité pour **28 cas ambigus**

---

## Étape 2 — Extraction et chunking des TdRs

### Problème

Pour indexer les TdRs dans une base vectorielle, on ne peut pas encoder un document entier en un seul vecteur. Il faut découper chaque TdR en morceaux intelligents, extraire les métadonnées structurées, et préserver la cohérence sémantique des sections.

### Approche : pymupdf4llm + chunking par sections + extraction LLM intelligente

```
PDF
 ↓ pymupdf4llm.to_markdown()     → Markdown structuré (titres ##, ###, tableaux)
 ↓ _extract_smart()              → sélectionne les sections clés (~7 000 chars)
 ↓ LLM (Ollama ou OpenAI)        → extrait 9 champs de métadonnées en JSON
 ↓ clean_text()                  → normalise les sauts de ligne
 ↓ _split_by_sections()          → découpe aux titres Markdown
 ↓ RecursiveCharacterTextSplitter → parents ~1 500 chars, children ~400 chars
 ↓ Sauvegarde JSON               → un fichier par TdR dans data/chunks/
```

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/pipeline/step2_chunking/text_extractor.py](backend/pipeline/step2_chunking/text_extractor.py) | Extraction 3 niveaux : pymupdf4llm → PyMuPDF → OCR Tesseract |
| [backend/pipeline/step2_chunking/metadata_extractor.py](backend/pipeline/step2_chunking/metadata_extractor.py) | Extraction intelligente des métadonnées via LLM (Ollama ou OpenAI) |
| [backend/pipeline/step2_chunking/text_cleaner.py](backend/pipeline/step2_chunking/text_cleaner.py) | Normalisation légère du Markdown |
| [backend/pipeline/step2_chunking/chunker.py](backend/pipeline/step2_chunking/chunker.py) | Chunking par sections Markdown + subdivision parent-child |
| [backend/pipeline/step2_chunking/chunking_pipeline.py](backend/pipeline/step2_chunking/chunking_pipeline.py) | Orchestrateur principal |
| [backend/run_chunking.py](backend/run_chunking.py) | Point d'entrée CLI |

### Détail des sous-étapes

#### 1. Extraction texte (`text_extractor.py`) — 3 niveaux

| Niveau | Outil | Cas d'usage |
|---|---|---|
| 1 | `pymupdf4llm.to_markdown()` | PDF natif → Markdown avec titres et sections |
| 2 | PyMuPDF `page.get_text()` | Fallback texte brut si pymupdf4llm échoue |
| 3 | Tesseract OCR | PDFs scannés sans couche texte (ex: `TDR-ERP.pdf`) |

**Avantage de pymupdf4llm** : produit un Markdown structuré avec `# ## ###` pour les titres — exploité par le chunker pour respecter les frontières sémantiques du document.

#### 2. Extraction métadonnées intelligente (`metadata_extractor.py`)

Au lieu d'envoyer le document entier, `_extract_smart()` sélectionne :
- Les **2 000 premiers caractères** → titre, organisation, contexte
- Les **sections dont le titre contient un mot-clé** métadonnée : "objectif", "livrable", "profil", "budget", "durée", "lieu", "date"...
- Cap total à **7 000 chars** — soit ~65% de réduction vs full-doc, pour ~95% de qualité

Le LLM retourne un JSON structuré avec 9 champs :

```json
{
  "titre": "Recrutement d'un consultant ERP",
  "organisation": "Ministère des Finances",
  "objectifs": ["Moderniser le système comptable", "Former les équipes"],
  "livrables": ["Rapport d'analyse", "Plan de déploiement"],
  "profil_consultant": "Expert ERP, 10 ans d'expérience, maîtrise SAP",
  "duree": "6 mois",
  "lieu": "Rabat, Maroc",
  "budget": "50 000 USD",
  "date_limite": "30 juin 2025"
}
```

**Provider configurable** : si `AGENT_PROVIDER = "openai"`, gpt-4o est utilisé pour l'extraction (plus rapide, meilleure qualité) ; sinon qwen2.5:14b local.

#### 3. Chunking par sections (`chunker.py`)

Contrairement au chunking à taille fixe, le découpage se fait d'abord aux **frontières sémantiques** du document (titres Markdown) :

```
## Objectifs de la mission        ← parent 0
  ├── child 0_0 (~400 chars)
  └── child 0_1 (~400 chars)

## Profil du consultant           ← parent 1
  ├── child 1_0 (~400 chars)
  └── child 1_1 (~400 chars)
```

| Paramètre | Valeur | Raison |
|---|---|---|
| `PARENT_CHUNK_SIZE` | 1 500 chars | Section logique complète pour le LLM |
| `PARENT_CHUNK_OVERLAP` | 200 chars | Continuité entre sections |
| `CHILD_CHUNK_SIZE` | 400 chars | Précis pour la recherche vectorielle |
| `CHILD_CHUNK_OVERLAP` | 80 chars | Légère continuité entre children |

### Décisions techniques et alternatives considérées

| Décision | Choix retenu | Alternative écartée | Raison |
|---|---|---|---|
| Extraction PDF | pymupdf4llm (Markdown) | Docling | Docling trop lent (~10-20x), pymupdf4llm donne 90% des bénéfices |
| Chunking | Par sections Markdown | Taille fixe | Respecte les frontières sémantiques, tableaux intacts |
| Métadonnées | Sections clés ~7 000 chars | Full document 20 000 chars | 65% moins de tokens, même qualité |
| Fallback scannés | Tesseract OCR | Ignorer le fichier | Récupère les TdRs scannés au lieu de les perdre |

---

## Étape 3 — Indexation vectorielle

### Problème

On dispose de milliers de child chunks. Pour trouver les plus pertinents en réponse à une question en moins d'une seconde, on combine deux approches complémentaires : recherche sémantique (vecteurs) et recherche lexicale (mots exacts).

### Approche : BGE-M3 + ChromaDB + BM25 (Recherche Hybride)

```
data/chunks/*.json
        │
        ▼
[BGE-M3] → vecteur 1 024 dims par child chunk
        │
        ▼
[ChromaDB]
   tdr_children ← children + embeddings + métadonnées (HNSW ef_search=64)
   tdr_parents  ← parents + texte (récupérés par ID)
        │
        ▼
[Index BM25] ← construit en RAM au démarrage depuis ChromaDB

── Requête ──────────────────────────────────────────────────────────────────
Question utilisateur
        │
   ┌────┴────────────────────────┐
   │ BGE-M3 (sémantique)         │  → top K résultats
   │ BM25 (mots exacts)          │  → top K résultats
   └────┬────────────────────────┘
        │
   RRF (Reciprocal Rank Fusion)  → fusionne les deux classements
        │
   Cross-encoder reranker        → re-trie par pertinence réelle
        │
   Top résultats → agent
```

**Avantage de la recherche hybride** :
- BGE-M3 trouve les résultats sémantiquement proches ("consultant informatique" ~ "expert IT")
- BM25 trouve les termes exacts ("USAID", "50 000 USD", noms propres)
- RRF favorise les documents bien classés dans les DEUX recherches

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/pipeline/step3_indexing/embedder.py](backend/pipeline/step3_indexing/embedder.py) | BGE-M3 singleton, encodage par batch, gestion GPU/CPU |
| [backend/pipeline/step3_indexing/vector_store.py](backend/pipeline/step3_indexing/vector_store.py) | Interface ChromaDB + index BM25 + hybrid_search() + fonctions metadata |
| [backend/pipeline/step3_indexing/indexing_pipeline.py](backend/pipeline/step3_indexing/indexing_pipeline.py) | Orchestrateur — lit chunks, encode, insère dans ChromaDB |
| [backend/run_indexing.py](backend/run_indexing.py) | Point d'entrée CLI |

### Modèle d'embedding : `BAAI/bge-m3`

| Critère | Détail |
|---|---|
| **Performance** | Meilleur modèle open-source multilingue (2024-2025) |
| **Langues** | 100+ langues dont français et anglais |
| **Dimensions** | 1 024 |
| **Taille** | ~2.3 Go |

### Décisions techniques et alternatives considérées

| Décision | Choix retenu | Alternative écartée | Raison |
|---|---|---|---|
| Recherche | Hybride BM25 + vectorielle | Vectorielle seule | BM25 couvre les termes exacts manqués par la similarité sémantique |
| Fusion | RRF (Reciprocal Rank Fusion) | Somme pondérée des scores | RRF est robuste et ne nécessite pas de calibration des poids |
| Base vectorielle | ChromaDB (local) | FAISS, Qdrant | Simple, local, supporte filtres sur métadonnées |
| HNSW ef_search | 64 (vs défaut 10) | Valeur par défaut | Meilleur rappel au prix d'une légère latence supplémentaire |

---

## Étape 4 — Agent RAG (ReAct + LangGraph)

### Problème

Avec la base vectorielle prête, il faut un agent capable de raisonner, de choisir le bon outil selon la question, et de synthétiser les résultats pour répondre en langage naturel.

### Approche : pattern ReAct avec LangGraph + 6 outils spécialisés

L'agent suit une boucle **Thought → Action → Observation** :

```
Question : "Quel profil de consultant pour une mission ERP ?"
│
▼ [Agent choisit l'outil approprié]
├── Action   : search_child_chunks("profil consultant ERP qualifications")
├── Observation : chunks pertinents trouvés (hybride BM25 + vectoriel + reranking)
├── Action   : retrieve_parent_chunks("TDR_ERP_p3, TDR_ERP_p5")
├── Observation : contexte complet (1 500 chars par parent)
└── Réponse finale : synthèse structurée avec sources citées
```

### Les 6 outils de l'agent

| Outil | Description | Cas d'usage |
|---|---|---|
| `search_child_chunks(query)` | Recherche hybride BM25+vectorielle + reranking cross-encoder | Questions thématiques, qualifications, livrables |
| `retrieve_parent_chunks(ids)` | Récupère le contexte complet par ID | Après search, pour enrichir le contexte LLM |
| `count_documents()` | Nombre total de TdRs dans la base | "Combien de TdRs ?" |
| `list_all_documents()` | Liste tous les TdRs avec métadonnées | "Liste tous les TdRs disponibles" |
| `filter_documents(filters)` | Filtre par organisation, lieu, budget, durée | "TdRs de l'UNICEF au Maroc" |
| `get_document_details(source)` | Fiche complète d'un TdR spécifique | "Détails du TdR-ERP" |

### Provider LLM configurable

L'agent supporte deux providers sans redémarrer le serveur :

| Provider | Modèle | Avantages |
|---|---|---|
| **Ollama** (local) | qwen2.5:14b-instruct-q3_K_M | Gratuit, confidentiel, hors ligne |
| **OpenAI** (API) | gpt-4o | Meilleure qualité, plus rapide, analyses plus riches |

Le switch se fait via l'API (`POST /api/provider`) ou le bouton toggle dans l'interface React.

### Prompts adaptés par modèle

Chaque provider dispose d'un prompt système optimisé pour ses capacités :

- **Prompt Ollama** : instructions courtes et explicites, ordre strict des outils, règles anti-hallucination renforcées
- **Prompt GPT-4o** : instructions riches, analyses comparatives, tableaux Markdown, minimum 4 recherches, extraction de chiffres exacts (budgets, durées, dates)

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/pipeline/step4_agent/tools.py](backend/pipeline/step4_agent/tools.py) | Les 6 outils LangChain de l'agent |
| [backend/pipeline/step4_agent/agent.py](backend/pipeline/step4_agent/agent.py) | Graph LangGraph + prompts duaux + provider factory + mémoire |
| [backend/pipeline/step4_agent/reranker.py](backend/pipeline/step4_agent/reranker.py) | Cross-encoder BAAI/bge-reranker-base pour le reranking |
| [backend/pipeline/step4_agent/agent_pipeline.py](backend/pipeline/step4_agent/agent_pipeline.py) | Interface `ask()` pour l'API FastAPI |
| [backend/run_agent.py](backend/run_agent.py) | CLI interactive avec streaming token par token |

### Décisions techniques et alternatives considérées

| Décision | Choix retenu | Alternative écartée | Raison |
|---|---|---|---|
| LLM local | qwen2.5:14b-instruct-q3_K_M | qwen2.5:7b | Meilleur raisonnement, tient dans 6 Go VRAM avec quantization |
| Reranking | BAAI/bge-reranker-base (transformers) | CrossEncoder sentence-transformers | Évite le bug meta tensor de PyTorch 2.6 |
| Tool calling | SequentialChatOllama (force séquentiel) | Parallèle natif | qwen2.5 tente des appels parallèles qui cassent LangGraph |
| Provider switch | `_settings.AGENT_PROVIDER` dynamique | Redémarrage serveur | Bascule à chaud sans interruption de service |

---

## Étape 5 — API FastAPI + Interface React

### Problème

Le pipeline RAG est opérationnel en CLI, mais il faut une interface web accessible aux analystes EY sans compétences techniques. L'interface doit proposer deux modes : conversation intelligente et recherche directe avec filtres.

### Architecture

```
React Frontend (Vite)                FastAPI Backend
─────────────────────                ───────────────
Mode Chat                            POST /api/ask
  └── Question utilisateur    ──SSE→ Streaming token par token
  └── Réponse en temps réel         (LangGraph ReAct agent)

Mode Recherche                       POST /api/search
  └── Mots-clés + filtres    ──────→ BGE-M3 + BM25 + reranking
  └── Grille de résultats            Retourne TdRResult avec métadonnées

Toggle Ollama/GPT-4o                 GET/POST /api/provider
  └── Bouton switch header   ──────→ Switch provider à chaud
```

### Fonctionnalités de l'interface

#### Mode Chat
- Conversation en langage naturel avec l'agent RAG
- **Streaming SSE** : les tokens s'affichent en temps réel (pas d'attente)
- Indicateur de recherche en cours ("Recherche dans la base TdR...")
- Mémoire de conversation par session (questions de suivi possibles)
- Support français et anglais

#### Mode Recherche
- Barre de recherche par mots-clés (optionnelle si filtre rempli)
- Filtres : **Organisation / Bailleur** et **Lieu / Pays / Région**
- Grille de résultats avec cartes : score de pertinence, titre, organisation, lieu, durée, budget, date limite
- Score coloré : vert (≥70%), orange (≥50%), gris (<50%)
- Modal détail : profil consultant, objectifs, livrables, missions similaires

#### Toggle Provider
- Bouton switch dans le header : **Ollama ⟵⟶ GPT-4o**
- Bascule instantanée sans rechargement de page
- État synchronisé avec le backend au chargement

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/api/main.py](backend/api/main.py) | Application FastAPI + middleware CORS |
| [backend/api/routes/ask.py](backend/api/routes/ask.py) | `POST /api/ask` — streaming SSE de l'agent |
| [backend/api/routes/search.py](backend/api/routes/search.py) | `POST /api/search` — recherche directe avec filtres |
| [backend/api/routes/provider.py](backend/api/routes/provider.py) | `GET/POST /api/provider` — switch Ollama/OpenAI |
| [backend/api/routes/health.py](backend/api/routes/health.py) | `GET /api/health` — liveness check |
| [backend/run_api.py](backend/run_api.py) | Point d'entrée — lance uvicorn |
| [frontend/src/App.jsx](frontend/src/App.jsx) | Composant React principal (Chat + Recherche + Provider toggle) |
| [frontend/src/App.css](frontend/src/App.css) | Styles de l'interface |

### Détail des endpoints

#### `POST /api/ask` — Agent conversationnel (SSE)

```json
// Requête
{ "question": "Quel profil pour une mission ERP ?", "session_id": "uuid" }

// Réponse (stream SSE)
data: {"type": "tool", "content": "Recherche dans la base TdR..."}
data: {"type": "token", "content": "D'après"}
data: {"type": "token", "content": " le TdR..."}
data: {"type": "done"}
```

#### `POST /api/search` — Recherche directe

```json
// Requête
{ "query": "consultant formation", "organisation": "UNICEF", "lieu": "Maroc", "k": 12 }

// Réponse
{
  "results": [
    {
      "source": "TDR-formation.pdf",
      "score": 0.82,
      "titre": "Recrutement consultant formation",
      "organisation": "UNICEF",
      "lieu": "Rabat, Maroc",
      "budget": "30 000 USD",
      "duree": "3 mois",
      "date_limite": "15 juillet 2025",
      "profil_consultant": "Expert formation, 5 ans exp.",
      "objectifs": ["..."],
      "livrables": ["..."]
    }
  ]
}
```

#### `GET/POST /api/provider` — Switch provider

```json
// GET → provider actif
{ "provider": "openai", "model": "gpt-4o", "status": "active" }

// POST → switch
{ "provider": "ollama" }
// Réponse : { "provider": "ollama", "model": "qwen2.5:14b-instruct-q3_K_M", "status": "switched" }
```

### Décisions techniques et alternatives considérées

| Décision | Choix retenu | Alternative écartée | Raison |
|---|---|---|---|
| Streaming | SSE (Server-Sent Events) | WebSockets | SSE est unidirectionnel, plus simple, suffisant pour le streaming de réponse |
| Frontend | React + Vite (no TypeScript) | Next.js | Simplicité, pas de SSR nécessaire pour un outil interne |
| CUDA conflict | `CUDA_VISIBLE_DEVICES=""` au démarrage | Désactiver le reranker GPU | Évite le conflit PyTorch + Ollama sur 6 Go VRAM sans perdre le reranking |
| Clé API | Fichier `.env` (ignoré git) | Variable d'environnement manuelle | Simple et sécurisé pour usage local |

---

## Étape 6 — Tests & Benchmarking

### Problème

Un pipeline RAG sans tests est fragile : une modification dans le chunking, la recherche hybride ou les outils de l'agent peut dégrader silencieusement la qualité des réponses. Il faut mesurer et valider chaque composant de manière isolée et end-to-end.

### Approche : tests unitaires pytest + benchmarks + évaluation RAGAS

```
backend/tests/
  ├── conftest.py               ← fixtures partagées (SAMPLE_MARKDOWN, sample_chunks)
  ├── test_chunker.py           ← tests unitaires chunking (15+ assertions)
  ├── test_metadata_extractor.py ← tests unitaires _extract_smart()
  ├── test_hybrid_search.py     ← tests BM25 + algorithme RRF
  ├── test_tools.py             ← tests des 6 outils agent (monkeypatch ChromaDB)
  ├── dataset_eval.json         ← 10 questions de référence avec réponses attendues
  ├── evaluate_rag.py           ← évaluation RAGAS (4 métriques de qualité)
  ├── benchmark_search.py       ← vectoriel seul vs hybride BM25+vectoriel
  └── benchmark_providers.py   ← Ollama vs OpenAI (latence, qualité, outils)
```

### Tests unitaires (pytest)

Les tests unitaires vérifient chaque fonction de manière isolée — sans lancer l'API, sans ChromaDB, sans appel LLM.

| Fichier | Ce qui est testé | Approche |
|---|---|---|
| `test_chunker.py` | `_split_by_sections()`, `create_chunks()`, tailles parent/child, sanitize | Markdown synthétique en entrée |
| `test_metadata_extractor.py` | `_extract_smart()` : tête 2 000 chars, sections clés, cap 7 000 chars | Texte long avec sections métadonnées |
| `test_hybrid_search.py` | `_tokenize()`, logique RRF manuelle, fusion BM25+vectoriel | Listes de documents synthétiques |
| `test_tools.py` | 6 outils agent : `count_documents`, `filter_documents`, `get_document_details`... | `monkeypatch` des fonctions ChromaDB |

**Lancer les tests :**

```bash
cd backend
python -m pytest tests/ -v
# ou un seul fichier
python -m pytest tests/test_chunker.py -v
```

**Résultat attendu :**

```
52 passed in ~84s
```

### Benchmark : Vectoriel vs Hybride BM25

Compare les résultats de recherche vectorielle seule vs hybride (BM25 + vectorielle + RRF) sur 9 requêtes types.

**Métriques mesurées :**
- Nombre de résultats retournés
- Score de pertinence du top-1
- Latence (ms)
- Différences dans le top-3 (BM25 a-t-il changé les résultats ?)

```bash
cd backend
python tests/benchmark_search.py
```

**Résultats obtenus :**

| Métrique | Résultat |
|---|---|
| Latence vectoriel | ~60ms |
| Latence hybride | ~55ms (overhead BM25 négligeable) |
| Requêtes où BM25 a changé le top-3 | **6/9 (67%)** |
| Impact sur requêtes lexicales ("UNICEF", "50 000 USD") | 3/4 améliorées |
| Impact sur requêtes sémantiques | 2/3 améliorées |

> La recherche hybride est strictement meilleure que la vectorielle seule sans coût de latence supplémentaire.

### Benchmark : Ollama vs OpenAI

Compare les deux providers sur 8 questions types en mesurant latence, longueur de réponse et nombre d'outils appelés.

```bash
# L'API doit tourner sur http://localhost:8000
cd backend
python tests/benchmark_providers.py
```

**Métriques comparées :**

| Métrique | Ollama (qwen2.5:14b) | GPT-4o |
|---|---|---|
| Latence moyenne | Plus lente (local) | Plus rapide (API) |
| Longueur réponse | Concise | Plus détaillée |
| Outils appelés | 1-2 par question | 3-4 par question |
| Qualité analyse | Correcte | Supérieure (tableaux, chiffres) |

### Évaluation RAGAS

Évalue automatiquement la qualité du pipeline RAG avec 4 métriques standardisées :

| Métrique | Question posée | Score idéal |
|---|---|---|
| **Faithfulness** | La réponse est-elle fidèle aux chunks récupérés ? | 1.0 |
| **Answer Relevancy** | La réponse répond-elle bien à la question ? | 1.0 |
| **Context Recall** | Les bons contextes ont-ils été récupérés ? | 1.0 |
| **Context Precision** | Les contextes récupérés sont-ils tous pertinents ? | 1.0 |

**Prérequis :**
1. Remplir `dataset_eval.json` avec les vraies réponses tirées de tes TdRs (remplacer les `"ADAPTER"`)
2. API active sur `http://localhost:8000`
3. Provider OpenAI recommandé (GPT-4o comme juge RAGAS)

```bash
cd backend
python tests/evaluate_rag.py
```

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/tests/conftest.py](backend/tests/conftest.py) | Fixtures partagées pytest |
| [backend/tests/test_chunker.py](backend/tests/test_chunker.py) | Tests unitaires chunking |
| [backend/tests/test_metadata_extractor.py](backend/tests/test_metadata_extractor.py) | Tests unitaires extraction intelligente |
| [backend/tests/test_hybrid_search.py](backend/tests/test_hybrid_search.py) | Tests BM25 + RRF |
| [backend/tests/test_tools.py](backend/tests/test_tools.py) | Tests des 6 outils agent |
| [backend/tests/dataset_eval.json](backend/tests/dataset_eval.json) | 10 questions de référence pour RAGAS |
| [backend/tests/evaluate_rag.py](backend/tests/evaluate_rag.py) | Évaluation RAGAS automatique |
| [backend/tests/benchmark_search.py](backend/tests/benchmark_search.py) | Benchmark vectoriel vs hybride |
| [backend/tests/benchmark_providers.py](backend/tests/benchmark_providers.py) | Benchmark Ollama vs OpenAI |

---

## Étape 7 — Containerisation Docker Compose

### Problème

Le projet nécessite Python 3.11, Node.js 20, des modèles ML (~2.3 Go), ChromaDB et Ollama. L'installation manuelle prend ~30 minutes et varie selon les OS. Docker permet de livrer un environnement identique en une seule commande.

### Architecture des conteneurs

```
docker compose up
        │
        ├── [frontend]  node:20-alpine → Vite build → nginx:alpine
        │               Port 5173:80
        │               Proxy /api/* → backend:8000
        │
        └── [backend]   python:3.11-slim
                        Port 8000:8000
                        Volumes :
                          ./data/          → /app/data/        (ChromaDB + chunks)
                          hf_cache volume  → /root/.cache/huggingface (BGE-M3 + reranker)
                        Env :
                          OLLAMA_BASE_URL=http://host.docker.internal:11434
                          OPENAI_API_KEY (via .env)
```

**Ollama reste sur la machine hôte** — le conteneur backend le joint via `host.docker.internal:11434`.

### Fichiers créés

| Fichier | Rôle |
|---|---|
| [docker-compose.yml](docker-compose.yml) | Orchestrateur : backend + frontend + volume HuggingFace |
| [backend/Dockerfile](backend/Dockerfile) | Image Python : torch CPU + deps + code FastAPI |
| [backend/.dockerignore](backend/.dockerignore) | Exclut venv, cache, logs, résultats benchmarks |
| [frontend/Dockerfile](frontend/Dockerfile) | Build Vite (node:20) → Nginx (2 stages) |
| [frontend/nginx.conf](frontend/nginx.conf) | Sert le build React + proxy SSE `/api/*` → backend |
| [frontend/.dockerignore](frontend/.dockerignore) | Exclut node_modules, dist |

### Décisions techniques

| Décision | Choix retenu | Raison |
|---|---|---|
| Torch | CPU-only (`--index-url .../cpu`) | GPU réservé à Ollama hôte, réduit l'image de ~2 Go |
| Frontend | Build multi-stage node:20 → nginx | Image finale ~25 Mo au lieu de ~500 Mo avec node |
| Ollama | Sur hôte, pas dans Docker | Évite de dupliquer les modèles (~10 Go) |
| Modèles HF | Volume nommé `hf_cache` | Téléchargés une seule fois, persistés entre rebuilds |
| API calls | `API_BASE = ''` (URLs relatives) | Nginx proxy gère `/api/*` → pas de CORS ni hardcode |

### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installé et démarré
- Ollama en cours d'exécution sur la machine hôte
- Fichier `.env` à la racine (contenant `OPENAI_API_KEY`)

### Lancer avec Docker

```bash
# Premier lancement (build des images ~10-15 min)
docker compose up --build

# Lancements suivants (images déjà buildées)
docker compose up

# En arrière-plan
docker compose up -d

# Arrêter
docker compose down
```

- **Interface** → [http://localhost:5173](http://localhost:5173)
- **API** → [http://localhost:8000](http://localhost:8000)

> **Note** : le premier démarrage télécharge BGE-M3 (~2.3 Go) et bge-reranker-base (~1 Go) dans le volume `hf_cache`. Les démarrages suivants sont instantanés.

---

## Installation

### Prérequis

- Python 3.11+
- Node.js 18+ (pour le frontend)
- [Tesseract OCR v5](https://github.com/UB-Mannheim/tesseract/wiki) installé dans `C:\Program Files\Tesseract-OCR\` (Windows) avec les langues `fra` et `eng`
- [Ollama](https://ollama.com/) installé et en cours d'exécution (optionnel si OpenAI utilisé)

### 1. Cloner le dépôt

```bash
git clone https://github.com/omarmarrakchi/agentic-rag-EY.git
cd agentic-rag-EY
```

### 2. Créer et activer le virtualenv Python

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé OpenAI (optionnel)

Créer un fichier `.env` à la racine du projet (copier `.env.example`) :

```
OPENAI_API_KEY=sk-...
```

### 5. Télécharger le modèle Ollama (si provider local)

```bash
ollama pull qwen2.5:14b-instruct-q3_K_M
```

### 6. Installer les dépendances frontend

```bash
cd frontend
npm install
```

### 7. Placer les PDFs bruts

Copier les PDFs à analyser dans `data/raw_tdrs/`.

---

## Exécution

### Étape 1 — Filtrage des TdRs

```bash
cd backend
python run_filter.py
```

### Étape 2 — Chunking des TdRs

```bash
python run_chunking.py
```

### Étape 3 — Indexation vectorielle

```bash
python run_indexing.py
```

### Étape 4 — Agent RAG (mode CLI interactif)

```bash
python run_agent.py
```

### Étape 5 — Lancer l'API + l'interface web

**Backend** (depuis `backend/`) :
```bash
python run_api.py
# API disponible sur http://localhost:8000
```

**Frontend** (depuis `frontend/`) :
```bash
npm run dev
# Interface disponible sur http://localhost:5173
```

### Étape 7 — Docker Compose (alternative à l'installation manuelle)

```bash
# À la racine du projet
docker compose up --build
# Interface → http://localhost:5173
# API       → http://localhost:8000
```

### Étape 6 — Tests & Benchmarking

**Tests unitaires** (sans API) :
```bash
cd backend
python -m pytest tests/ -v
```

**Benchmark vectoriel vs hybride** (sans API) :
```bash
python tests/benchmark_search.py
```

**Benchmark Ollama vs OpenAI** (API requise) :
```bash
python tests/benchmark_providers.py
```

**Évaluation RAGAS** (API requise + dataset_eval.json rempli) :
```bash
python tests/evaluate_rag.py
```

**Changer de provider à chaud** :
```bash
# Basculer vers OpenAI
curl -X POST http://localhost:8000/api/provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai"}'

# Revenir en local
curl -X POST http://localhost:8000/api/provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama"}'
```

---

## Configuration

Tous les paramètres sont centralisés dans [backend/config/settings.py](backend/config/settings.py) :

```python
# Étape 2 — Chunking
PARENT_CHUNK_SIZE     = 1500  # caractères par parent chunk
CHILD_CHUNK_SIZE      = 400   # caractères par child chunk

# Étape 3 — Indexation vectorielle
EMBEDDING_MODEL       = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE  = 64

# Étape 4 — Agent RAG
AGENT_MODEL           = "qwen2.5:14b-instruct-q3_K_M"
AGENT_TOP_K           = 10     # children récupérés par recherche hybride
AGENT_SCORE_THRESHOLD = 0.45   # score minimum avant reranking
RERANKER_MODEL        = "BAAI/bge-reranker-base"
RERANKER_TOP_K        = 6      # résultats après reranking

# Provider LLM
AGENT_PROVIDER        = "ollama"   # "ollama" | "openai"
OPENAI_MODEL          = "gpt-4o"
OPENAI_API_KEY        = ""         # ou via .env
```

---

## Roadmap

- [x] **Étape 1** — Filtrage des PDFs (scoring + LLM)
- [x] **Étape 2** — Extraction pymupdf4llm + chunking par sections Markdown + métadonnées LLM intelligentes
- [x] **Étape 3** — Indexation vectorielle BGE-M3 + ChromaDB + recherche hybride BM25
- [x] **Étape 4** — Agent RAG LangGraph ReAct + 6 outils + reranking cross-encoder + prompts duaux
- [x] **Étape 5** — API FastAPI SSE + UI React (Chat + Recherche + Provider toggle Ollama/OpenAI)
- [x] **Étape 6** — Tests unitaires pytest + benchmarks vectoriel/hybride + évaluation RAGAS
- [x] **Étape 7** — Containerisation Docker Compose (backend Python + frontend Nginx)

---

## Structure du projet

```
agentic-rag-ey/
├── docker-compose.yml             ← orchestrateur Docker (backend + frontend)
├── .env                           ← clés API (ignoré par git)
├── .env.example                   ← template de configuration
├── data/
│   ├── raw_tdrs/                  ← PDFs bruts (non versionnés)
│   ├── filtered_tdrs/             ← TdRs validés (non versionnés)
│   ├── rejected/                  ← fichiers non-TdR (non versionnés)
│   ├── chunks/                    ← JSON par TdR (non versionnés)
│   └── vector_db/                 ← ChromaDB (non versionné)
├── backend/
│   ├── config/
│   │   └── settings.py            ← configuration centralisée + chargement .env
│   ├── pipeline/
│   │   ├── step1_filter/
│   │   │   ├── pdf_reader.py      ← lecture + OCR
│   │   │   ├── keyword_scorer.py  ← scoring mots-clés
│   │   │   ├── llm_classifier.py  ← classification Ollama
│   │   │   └── filter_pipeline.py ← orchestrateur
│   │   ├── step2_chunking/
│   │   │   ├── text_extractor.py  ← pymupdf4llm + OCR fallback
│   │   │   ├── metadata_extractor.py ← extraction intelligente Ollama/OpenAI
│   │   │   ├── text_cleaner.py    ← nettoyage Markdown
│   │   │   ├── chunker.py         ← chunking par sections Markdown
│   │   │   └── chunking_pipeline.py ← orchestrateur
│   │   ├── step3_indexing/
│   │   │   ├── embedder.py        ← BGE-M3, batch, GPU/CPU
│   │   │   ├── vector_store.py    ← ChromaDB + BM25 + hybrid_search + filtres
│   │   │   └── indexing_pipeline.py ← orchestrateur
│   │   └── step4_agent/
│   │       ├── tools.py           ← 6 outils LangChain
│   │       ├── agent.py           ← LangGraph + prompts duaux + provider factory
│   │       ├── reranker.py        ← cross-encoder BAAI/bge-reranker-base
│   │       └── agent_pipeline.py  ← interface ask()
│   ├── api/
│   │   ├── main.py                ← FastAPI app + CORS
│   │   └── routes/
│   │       ├── ask.py             ← POST /api/ask (SSE streaming)
│   │       ├── search.py          ← POST /api/search (recherche directe)
│   │       ├── provider.py        ← GET/POST /api/provider (switch LLM)
│   │       └── health.py          ← GET /api/health
│   ├── tests/
│   │   ├── conftest.py            ← fixtures partagées pytest
│   │   ├── test_chunker.py        ← tests unitaires chunking
│   │   ├── test_metadata_extractor.py ← tests _extract_smart()
│   │   ├── test_hybrid_search.py  ← tests BM25 + RRF
│   │   ├── test_tools.py          ← tests 6 outils agent
│   │   ├── dataset_eval.json      ← questions de référence RAGAS
│   │   ├── evaluate_rag.py        ← évaluation RAGAS
│   │   ├── benchmark_search.py    ← vectoriel vs hybride
│   │   └── benchmark_providers.py ← Ollama vs OpenAI
│   ├── logs/                      ← rapports JSON (non versionnés)
│   ├── Dockerfile                 ← image Python backend
│   ├── .dockerignore
│   ├── requirements.txt
│   ├── run_filter.py
│   ├── run_chunking.py
│   ├── run_indexing.py
│   ├── run_agent.py
│   └── run_api.py                 ← point d'entrée API
└── frontend/
    ├── src/
    │   ├── App.jsx                ← composant principal (Chat + Recherche + Toggle)
    │   └── App.css                ← styles
    ├── Dockerfile                 ← build Vite → Nginx (multi-stage)
    ├── nginx.conf                 ← proxy /api/* → backend + SPA routing
    ├── .dockerignore
    ├── package.json
    └── vite.config.js
```
