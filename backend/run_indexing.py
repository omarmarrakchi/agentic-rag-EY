"""Point d'entrée pour lancer le pipeline d'indexation vectorielle."""

import os
import sys
from pathlib import Path

# Active le GPU pour l'encodage en batch (indexation uniquement)
os.environ["EMBEDDER_DEVICE"] = "cuda"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.step3_indexing.indexing_pipeline import run

if __name__ == "__main__":
    run()
