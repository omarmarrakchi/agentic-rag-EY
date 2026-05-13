"""Point d'entrée pour lancer le pipeline de chunking des TdRs filtrés."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.step2_chunking.chunking_pipeline import run

if __name__ == "__main__":
    run()
