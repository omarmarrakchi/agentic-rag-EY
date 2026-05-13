"""Point d'entrée pour lancer le pipeline d'indexation vectorielle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.step3_indexing.indexing_pipeline import run

if __name__ == "__main__":
    run()
