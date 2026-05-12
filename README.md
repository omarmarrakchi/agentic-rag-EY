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

backend/
  config/
    settings.py      ← paramètres centralisés (seuils, chemins, modèles)
  pipeline/
    step1_filter/    ← filtrage et classification des PDFs
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

### Configuration

Tous les paramètres sont centralisés dans [backend/config/settings.py](backend/config/settings.py) :

```python
SCORE_TDR_CONFIRMED   = 5     # seuil minimum pour confirmer un TdR sans LLM
SCORE_REJECTED        = 0     # seuil maximum pour rejeter sans LLM
TEXT_EXTRACTION_CHARS = 5000  # caractères extraits par PDF
OCR_MAX_PAGES         = 3     # pages analysées par OCR
OLLAMA_MODEL          = "qwen2.5:7b"
TESSERACT_CMD         = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Roadmap

- [x] **Étape 1** — Filtrage des PDFs (scoring + LLM)
- [ ] **Étape 2** — Extraction et chunking des TdRs filtrés
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
│   │   └── step1_filter/
│   │       ├── pdf_reader.py       ← lecture + OCR
│   │       ├── keyword_scorer.py   ← scoring mots-clés
│   │       ├── llm_classifier.py   ← classification Ollama
│   │       └── filter_pipeline.py  ← orchestrateur
│   ├── logs/              ← rapports JSON (non versionnés)
│   ├── requirements.txt
│   └── run_filter.py      ← point d'entrée CLI
└── frontend/              ← UI React (à venir)
```
