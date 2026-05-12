"""Point d'entrée pour lancer le pipeline de filtrage des TdRs."""

import sys
from pathlib import Path

# Permet d'importer les modules backend depuis n'importe où
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.step1_filter.filter_pipeline import run

if __name__ == "__main__":
    run()
