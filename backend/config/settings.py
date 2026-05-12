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

# Nombre de caractères extraits pour l'analyse (première page suffit)
TEXT_EXTRACTION_CHARS = 2000

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"    # modèle performant pour la classification

# ── Mots-clés de scoring ────────────────────────────────────────────────────
KEYWORDS_STRONG = [
    "termes de référence",
    "terms of reference",
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
]

KEYWORDS_WEAK = [
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
]

KEYWORDS_EXCLUSION = [
    "manuel utilisateur",
    "user manual",
    "catalogue",
    "rapport d'avancement",
    "progress report",
    "certificat",
    "datasheet",
    "fiche technique",
    "plan de montage",
    "règlement",
    "regulation",
]

# Poids des catégories
WEIGHT_STRONG    =  3
WEIGHT_WEAK      =  1
WEIGHT_EXCLUSION = -3
