from pathlib import Path

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Dossiers de données
RAW_TDRS_DIR     = ROOT_DIR / "data" / "raw_tdrs"
FILTERED_DIR     = ROOT_DIR / "data" / "filtered_tdrs"
REJECTED_DIR     = ROOT_DIR / "data" / "rejected"

# Logs
LOGS_DIR = ROOT_DIR / "backend" / "logs"

# ── Filtrage TdR ────────────────────────────────────────────────────────────
# Seuil au-dessus duquel le fichier est considéré TdR sans passer par Ollama
SCORE_TDR_CONFIRMED   = 5
# Seuil en-dessous duquel le fichier est rejeté sans passer par Ollama
SCORE_REJECTED        = 0
# Entre les deux → Ollama tranche

# Nombre de caractères extraits pour l'analyse
TEXT_EXTRACTION_CHARS = 5000

# ── OCR ─────────────────────────────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_LANGUAGES = "fra+eng"  # français + anglais
OCR_MAX_PAGES = 3          # pages analysées par OCR (couvre les couvertures vides)

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"    # modèle performant pour la classification

# ── Mots-clés de scoring ────────────────────────────────────────────────────
KEYWORDS_STRONG = [
    # Français
    "termes de référence",
    "appel à candidatures",
    "appel d'offres",
    "profil du consultant",
    "profil recherché",
    "livrables attendus",
    "bailleur de fonds",
    "cahier des charges",
    "recrutement d'un consultant",
    "recrutement d'un expert",
    "mission de consultance",
    "avis de recrutement",
    "avis de manifestation d'intérêt",
    # Anglais
    "terms of reference",
    "call for consultants",
    "request for proposals",
    "scope of work",
    "statement of work",
    "consultant profile",
    "expected deliverables",
    "recruitment of a consultant",
    "recruitment of an expert",
    "expression of interest",
    "individual consultant",
    "consulting firm",
]

KEYWORDS_WEAK = [
    # Français
    "consultant",
    "expertise",
    "mission",
    "objectifs",
    "livrables",
    "compétences requises",
    "qualifications",
    "expérience requise",
    "méthodologie",
    "rapport final",
    "durée de la mission",
    "prestataire",
    "soumissionnaire",
    # Anglais
    "deliverables",
    "qualifications required",
    "required experience",
    "methodology",
    "final report",
    "duration of the assignment",
    "bidder",
    "contractor",
    "assignment",
    "procurement",
]

KEYWORDS_EXCLUSION = [
    # Français
    "manuel utilisateur",
    "catalogue",
    "rapport d'avancement",
    "certificat",
    "fiche technique",
    "plan de montage",
    "règlement intérieur",      # plus spécifique que "règlement" seul
    # Anglais
    "user manual",
    "progress report",
    "datasheet",
    "technical specification",
    "product manual",
]

# Poids des catégories
WEIGHT_STRONG    =  3
WEIGHT_WEAK      =  1
WEIGHT_EXCLUSION = -3

# ── Étape 2 — Chunking ──────────────────────────────────────────────────────
CHUNKS_DIR            = ROOT_DIR / "data" / "chunks"
PARENT_CHUNK_SIZE     = 800   # caractères par parent chunk
PARENT_CHUNK_OVERLAP  = 100   # chevauchement entre parents
CHILD_CHUNK_SIZE      = 200   # caractères par child chunk
CHILD_CHUNK_OVERLAP   = 20    # chevauchement entre children

# ── Étape 3 — Indexation vectorielle ────────────────────────────────────────
VECTOR_DB_DIR         = ROOT_DIR / "data" / "vector_db"
EMBEDDING_MODEL       = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE  = 32    # chunks encodés en parallèle
COLLECTION_CHILDREN   = "tdr_children"   # collection ChromaDB des children
COLLECTION_PARENTS    = "tdr_parents"    # collection ChromaDB des parents
