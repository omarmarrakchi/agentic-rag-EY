# Agentic RAG — Indexation et Recherche de TdRs (EY)

Système de RAG agentique pour l'indexation et la recherche intelligente dans un corpus de Termes de Référence (TdRs) issus des marchés publics.

---

## Objectif du projet

Construire un pipeline complet capable de :
1. **Filtrer** automatiquement un corpus de ~100 PDFs pour n'en garder que les TdRs réels
2. **Extraire et indexer** le contenu de ces TdRs dans une base vectorielle
3. **Répondre** à des questions en langage naturel via un agent RAG (architecture ReAct)
4. **Exposer** l'interface via une API FastAPI et une UI React

---

## Architecture globale

```
data/
  raw_tdrs/          ← PDFs bruts (100 fichiers, mix TdRs + non-TdRs)
  filtered_tdrs/     ← TdRs confirmés (sortie Step 1)
  rejected/          ← fichiers non-TdR (sortie Step 1)
  chunks/            ← chunks JSON par TdR (sortie Step 2)

backend/
  config/
    settings.py      ← paramètres centralisés (seuils, chemins, modèles)
  pipeline/
    step1_filter/    ← filtrage et classification des PDFs
    step2_chunking/  ← extraction, métadonnées et chunking des TdRs
  logs/              ← rapports JSON des runs

frontend/            ← UI React (à venir)
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

**Normalisation des accents** : le texte et les mots-clés sont normalisés avec `unicodedata` avant comparaison, ce qui permet de matcher les sorties OCR imparfaites (ex : "TERMES DE REFERENCES" → "termes de reference" matche "termes de référence").

**Seuils de décision** :
- Score ≥ 5 → **TdR confirmé** (pas besoin du LLM)
- Score ≤ 0 → **Rejeté** (pas besoin du LLM)
- Score 1–4 → **Ambigu** → passage à Ollama

#### 3. Classification LLM (`llm_classifier.py`)

Pour les cas ambigus, les 500 premiers caractères du texte sont envoyés au modèle `qwen2.5:7b` via Ollama (serveur local). Le modèle répond en JSON strict :

```json
{"is_tdr": true, "reason": "Le document décrit une mission de consultance avec livrables et profil requis"}
```

**Cas particulier — PDF scanné sans texte** : si l'OCR a échoué (texte vide), le nom du fichier est transmis au LLM comme contexte de secours.

### Décisions techniques et alternatives considérées

| Décision | Choix retenu | Alternative écartée | Raison |
|---|---|---|---|
| LLM local vs API | Ollama (`qwen2.5:7b`) en local | OpenAI GPT-4 | Confidentialité des données, pas de coût API, fonctionne hors ligne |
| LLM pour tous les fichiers | Non — seulement les ambigus | Passer tous les PDFs au LLM | Latence : ~2 s/fichier × 100 = 3 min vs ~10 s pour les cas certains |
| OCR | Tesseract v5 + pytesseract | PyMuPDF OCR intégré | Tesseract donne de meilleurs résultats sur les PDFs scannés en fra+eng |
| Seuil "TdR confirmé" | Score ≥ 5 (soit 2 mots forts) | Seuil plus bas | Évite les faux positifs sur des documents qui contiennent "consultant" par hasard |
| Mots d'exclusion | Termes très spécifiques ("règlement intérieur") | Termes génériques ("règlement") | "règlement" apparaît dans les clauses de marchés publics légitimes et pénalisait des vrais TdRs |

### Résultats (Run final)

Sur 100 PDFs bruts :
- **58 TdRs confirmés** → `data/filtered_tdrs/`
- **41 rejetés** → `data/rejected/`
- **1 erreur** (fichier corrompu `TR-2012-006_TDR_WP73.pdf`)
- Ollama sollicité pour **28 cas ambigus**

Rapport détaillé : `backend/logs/filter_report_20260512_161530.json`

---

## Étape 2 — Extraction et chunking des TdRs

### Problème

Pour indexer les TdRs dans une base vectorielle, on ne peut pas encoder un document entier en un seul vecteur — les modèles d'embeddings ont une limite de ~512 tokens. Il faut découper chaque TdR en morceaux, mais de façon intelligente pour ne pas perdre le contexte lors des réponses.

### Approche : chunking hiérarchique parent-child

Chaque TdR est découpé en deux niveaux :

```
Document complet (ex: TDR-ERP.pdf, 15 000 chars)
│
├── Parent chunk 0  (chars 0–800)     ← contexte riche pour le LLM
│     ├── Child 0_0  (chars 0–200)    ← petit chunk précis pour la recherche
│     ├── Child 0_1  (chars 180–380)
│     └── Child 0_2  (chars 360–560)
│
├── Parent chunk 1  (chars 700–1500)  ← overlap de 100 chars avec parent 0
│     ├── Child 1_0  ...
│     └── ...
└── ...
```

**Pourquoi deux niveaux ?**
- Les **children** (200 chars) sont précis → utilisés pour la **recherche vectorielle**
- Les **parents** (800 chars) sont riches → fournis au **LLM pour formuler la réponse**
- L'overlap entre chunks évite de couper une phrase entre deux morceaux

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| [backend/pipeline/step2_chunking/text_extractor.py](backend/pipeline/step2_chunking/text_extractor.py) | Extrait le texte complet du PDF (sans limite de caractères) |
| [backend/pipeline/step2_chunking/metadata_extractor.py](backend/pipeline/step2_chunking/metadata_extractor.py) | Extrait les attributs structurés du TdR via Ollama (3 500 premiers chars) |
| [backend/pipeline/step2_chunking/text_cleaner.py](backend/pipeline/step2_chunking/text_cleaner.py) | Nettoie le texte (espaces, artefacts OCR, mots coupés) |
| [backend/pipeline/step2_chunking/chunker.py](backend/pipeline/step2_chunking/chunker.py) | Découpe en parents (800 chars) puis en children (200 chars) avec overlap |
| [backend/pipeline/step2_chunking/chunking_pipeline.py](backend/pipeline/step2_chunking/chunking_pipeline.py) | Orchestrateur — lit `filtered_tdrs/`, écrit dans `data/chunks/` |
| [backend/run_chunking.py](backend/run_chunking.py) | Point d'entrée CLI |

### Détail des sous-étapes

#### 1. Extraction texte complet (`text_extractor.py`)

Même logique que l'étape 1 (natif vs OCR), mais sans limite de caractères — le texte entier du document est extrait pour ne rien perdre lors de l'indexation.

#### 2. Extraction des métadonnées (`metadata_extractor.py`)

Les 3 500 premiers caractères sont envoyés à Ollama une seule fois par document. Le LLM retourne un JSON structuré :

```json
{
  "titre": "Recrutement d'un consultant ERP",
  "organisation": "Ministère des Finances",
  "objectifs": ["Moderniser le système comptable", "Former les équipes"],
  "livrables": ["Rapport d'analyse", "Plan de déploiement"],
  "profil_consultant": "Expert ERP, 10 ans d'expérience, maîtrise SAP",
  "duree": "6 mois",
  "lieu": "Rabat, Maroc",
  "budget": null,
  "date_limite": null
}
```

Ces métadonnées sont attachées à chaque chunk et serviront de **filtres** dans la base vectorielle (ex : *"montre-moi les TdRs de l'UNICEF de moins de 3 mois"*).

#### 3. Nettoyage (`text_cleaner.py`)

- Supprime les caractères de contrôle et tirets mous
- Normalise les espaces et tabulations
- Pour les PDFs scannés : recolle les mots coupés en fin de ligne (`consul-\ntant` → `consultant`)
- Limite les sauts de ligne consécutifs à 2

#### 4. Chunking parent-child (`chunker.py`)

| Paramètre | Valeur | Raison |
|---|---|---|
| `PARENT_CHUNK_SIZE` | 800 chars | ~150 tokens — contexte suffisant pour le LLM |
| `PARENT_CHUNK_OVERLAP` | 100 chars | Évite de couper une phrase entre deux parents |
| `CHILD_CHUNK_SIZE` | 200 chars | ~40 tokens — précis pour la recherche vectorielle |
| `CHILD_CHUNK_OVERLAP` | 20 chars | Légère continuité entre children |

### Format de sortie (un JSON par TdR)

```json
{
  "source": "TDR-ERP.pdf",
  "is_scanned": false,
  "page_count": 8,
  "total_chars": 15240,
  "metadata": { "titre": "...", "organisation": "...", ... },
  "parents": [
    { "chunk_id": "TDR_ERP_p0", "type": "parent", "text": "...", "children": ["TDR_ERP_c0_0", ...] }
  ],
  "children": [
    { "chunk_id": "TDR_ERP_c0_0", "type": "child", "text": "...", "parent_id": "TDR_ERP_p0" }
  ]
}
```

### Décisions techniques et alternatives considérées

| Décision | Choix retenu | Alternative écartée | Raison |
|---|---|---|---|
| Stratégie de chunking | Parent-child hiérarchique | Chunks fixes taille unique | Permet recherche précise (child) + réponse contextualisée (parent) |
| Taille parent | 800 chars | 1 500 chars | Reste dans la fenêtre du LLM même pour des modèles légers |
| Extraction métadonnées | LLM sur 3 500 premiers chars | Regex/heuristiques | Les TdRs ont des structures trop variées pour des patterns fixes |
| Stockage intermédiaire | JSON par document dans `data/chunks/` | Directement en base vectorielle | Permet de déboguer et rejouer l'étape 3 sans refaire l'OCR |

### Résultats (Run final)

Sur 58 TdRs filtrés :
- **116 fichiers traités** (PDFs + variantes .PDF)
- **0 erreur**
- **3 970 parent chunks** créés
- **19 586 child chunks** créés

Rapport détaillé : `backend/logs/chunking_report_20260512_172549.json`

---

## Installation

### Prérequis

- Python 3.11+
- [Tesseract OCR v5](https://github.com/UB-Mannheim/tesseract/wiki) installé dans `C:\Program Files\Tesseract-OCR\` (Windows) avec les langues `fra` et `eng`
- [Ollama](https://ollama.com/) installé et en cours d'exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/omarmarrakchi/agentic-rag-EY.git
cd agentic-rag-EY
```

### 2. Créer et activer le virtualenv

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

### 4. Télécharger le modèle Ollama

```bash
ollama pull qwen2.5:7b
```

Vérifier qu'Ollama tourne :

```bash
ollama list
# qwen2.5:7b doit apparaître dans la liste
```

### 5. Placer les PDFs bruts

Copier les PDFs à analyser dans `data/raw_tdrs/`.

---

## Exécution

### Étape 1 — Filtrage des TdRs

Depuis le dossier `backend/` (avec le venv activé) :

```bash
python run_filter.py
```

Le pipeline :
1. Lit tous les PDFs de `data/raw_tdrs/`
2. Extrait le texte (natif ou OCR)
3. Calcule le score par mots-clés
4. Passe les cas ambigus à Ollama
5. Copie les fichiers dans `data/filtered_tdrs/` ou `data/rejected/`
6. Sauvegarde un rapport JSON dans `backend/logs/`

**Sortie attendue :**
```
INFO     99 PDFs trouvés dans data/raw_tdrs
Filtrage TdRs: 100%|████████████| 99/99 [02:34<00:00]
INFO     Terminé — TdRs: 58 | Rejetés: 41 | Erreurs: 1 | Ollama utilisé: 28 fois
INFO     Rapport sauvegardé : backend/logs/filter_report_YYYYMMDD_HHMMSS.json
```

### Étape 2 — Chunking des TdRs

Depuis le dossier `backend/` (avec le venv activé) :

```bash
python run_chunking.py
```

Le pipeline :
1. Lit tous les PDFs de `data/filtered_tdrs/`
2. Extrait le texte complet (natif ou OCR toutes pages)
3. Appelle Ollama pour extraire les métadonnées structurées
4. Nettoie le texte
5. Découpe en chunks parent-child
6. Sauvegarde un JSON par TdR dans `data/chunks/`
7. Sauvegarde un rapport JSON dans `backend/logs/`

**Sortie attendue :**
```
INFO     58 TdRs trouvés dans data/filtered_tdrs
Chunking TdRs: 100%|████████████| 58/58
INFO     Terminé — Succès: 58 | Erreurs: 0 | Parents: 3970 | Children: 19586
INFO     Rapport sauvegardé : backend/logs/chunking_report_YYYYMMDD_HHMMSS.json
```

### Configuration

Tous les paramètres sont centralisés dans [backend/config/settings.py](backend/config/settings.py) :

```python
# Étape 1 — Filtrage
SCORE_TDR_CONFIRMED   = 5     # seuil minimum pour confirmer un TdR sans LLM
SCORE_REJECTED        = 0     # seuil maximum pour rejeter sans LLM
TEXT_EXTRACTION_CHARS = 5000  # caractères extraits par PDF pour la classification
OCR_MAX_PAGES         = 3     # pages analysées par OCR
OLLAMA_MODEL          = "qwen2.5:7b"
TESSERACT_CMD         = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Étape 2 — Chunking
PARENT_CHUNK_SIZE     = 800   # caractères par parent chunk
PARENT_CHUNK_OVERLAP  = 100   # chevauchement entre parents
CHILD_CHUNK_SIZE      = 200   # caractères par child chunk
CHILD_CHUNK_OVERLAP   = 20    # chevauchement entre children
```

---

## Roadmap

- [x] **Étape 1** — Filtrage des PDFs (scoring + LLM)
- [x] **Étape 2** — Extraction et chunking des TdRs filtrés
- [ ] **Étape 3** — Indexation vectorielle (embeddings + base vectorielle)
- [ ] **Étape 4** — Agent RAG (LangGraph, pattern ReAct)
- [ ] **Étape 5** — API FastAPI + UI React
- [ ] **Étape 6** — Containerisation Docker Compose

---

## Structure du projet

```
agentic-rag-ey/
├── data/
│   ├── raw_tdrs/          ← PDFs bruts à analyser (non versionnés)
│   ├── filtered_tdrs/     ← TdRs validés (non versionnés)
│   └── rejected/          ← fichiers non-TdR (non versionnés)
├── backend/
│   ├── config/
│   │   └── settings.py    ← configuration centralisée
│   ├── pipeline/
│   │   ├── step1_filter/
│   │   │   ├── pdf_reader.py          ← lecture + OCR
│   │   │   ├── keyword_scorer.py      ← scoring mots-clés
│   │   │   ├── llm_classifier.py      ← classification Ollama
│   │   │   └── filter_pipeline.py     ← orchestrateur
│   │   └── step2_chunking/
│   │       ├── text_extractor.py      ← extraction texte complet
│   │       ├── metadata_extractor.py  ← attributs TdR via Ollama
│   │       ├── text_cleaner.py        ← nettoyage texte
│   │       ├── chunker.py             ← découpage parent-child
│   │       └── chunking_pipeline.py   ← orchestrateur
│   ├── logs/              ← rapports JSON (non versionnés)
│   ├── requirements.txt
│   ├── run_filter.py      ← point d'entrée étape 1
│   └── run_chunking.py    ← point d'entrée étape 2
└── frontend/              ← UI React (à venir)
```
