"""
Orchestre l'indexation de tous les chunks dans ChromaDB :
  1. Lit les JSON de data/chunks/
  2. Encode les children par batch avec BGE-M3
  3. Insère les children (avec embeddings) dans la collection tdr_children
  4. Insère les parents (sans embeddings) dans la collection tdr_parents
  5. Sauvegarde un rapport JSON dans backend/logs/
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import colorlog
from tqdm import tqdm

from config.settings import CHUNKS_DIR, LOGS_DIR, EMBEDDING_BATCH_SIZE
from pipeline.step3_indexing.embedder import encode
from pipeline.step3_indexing.vector_store import (
    insert_children,
    insert_parents,
    collection_stats,
)


def _setup_logger() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red"},
    ))
    logger = logging.getLogger("indexing_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def _build_child_metadata(child: dict, doc_meta: dict, source: str) -> dict:
    """Construit les métadonnées d'un child à stocker dans ChromaDB."""
    return {
        "parent_id":    child["parent_id"],
        "source":       source,
        "titre":        doc_meta.get("titre") or "",
        "organisation": doc_meta.get("organisation") or "",
        "duree":        doc_meta.get("duree") or "",
        "lieu":         doc_meta.get("lieu") or "",
        "objectifs":    doc_meta.get("objectifs") or [],
        "livrables":    doc_meta.get("livrables") or [],
    }


def _build_parent_metadata(parent: dict, doc_meta: dict, source: str) -> dict:
    """Construit les métadonnées d'un parent à stocker dans ChromaDB."""
    return {
        "source":       source,
        "titre":        doc_meta.get("titre") or "",
        "organisation": doc_meta.get("organisation") or "",
        "duree":        doc_meta.get("duree") or "",
        "lieu":         doc_meta.get("lieu") or "",
    }


def run() -> dict:
    """
    Indexe tous les chunks JSON dans ChromaDB.
    Retourne les statistiques finales.
    """
    logger = _setup_logger()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))
    logger.info(f"{len(chunk_files)} fichiers JSON trouvés dans {CHUNKS_DIR}")

    # ── Collecte de tous les children et parents ──────────────────────────
    all_child_ids, all_child_texts, all_child_metas = [], [], []
    all_parent_ids, all_parent_texts, all_parent_metas = [], [], []

    for json_path in tqdm(chunk_files, desc="Chargement chunks", unit="fichier"):
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)

        source   = doc["source"]
        doc_meta = doc.get("metadata", {})

        for child in doc["children"]:
            all_child_ids.append(child["chunk_id"])
            all_child_texts.append(child["text"])
            all_child_metas.append(_build_child_metadata(child, doc_meta, source))

        for parent in doc["parents"]:
            all_parent_ids.append(parent["chunk_id"])
            all_parent_texts.append(parent["text"])
            all_parent_metas.append(_build_parent_metadata(parent, doc_meta, source))

    logger.info(f"{len(all_child_ids)} children | {len(all_parent_ids)} parents à indexer")

    # ── Encodage des children par batch ───────────────────────────────────
    logger.info("Encodage des children avec BGE-M3 (peut prendre plusieurs minutes)...")
    all_embeddings = []

    for i in tqdm(
        range(0, len(all_child_texts), EMBEDDING_BATCH_SIZE),
        desc="Encodage",
        unit="batch",
    ):
        batch = all_child_texts[i : i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(encode(batch))

    logger.info(f"Encodage terminé — {len(all_embeddings)} vecteurs de 1024 dimensions")

    # ── Insertion dans ChromaDB ────────────────────────────────────────────
    logger.info("Insertion des children dans ChromaDB...")
    insert_children(all_child_ids, all_embeddings, all_child_texts, all_child_metas)

    logger.info("Insertion des parents dans ChromaDB...")
    insert_parents(all_parent_ids, all_parent_texts, all_parent_metas)

    # ── Statistiques finales ───────────────────────────────────────────────
    stats = collection_stats()
    stats["fichiers_traites"] = len(chunk_files)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = LOGS_DIR / f"indexing_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\nTerminé — Children indexés : {stats['children']} | "
        f"Parents stockés : {stats['parents']}"
    )
    logger.info(f"Rapport sauvegardé : {report_path}")

    return stats
