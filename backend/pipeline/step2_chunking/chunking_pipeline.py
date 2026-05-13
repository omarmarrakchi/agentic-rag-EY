"""
Orchestre les 4 étapes du chunking pour tous les TdRs filtrés :
  1. Extraction texte complet (text_extractor)
  2. Extraction métadonnées via Ollama (metadata_extractor)
  3. Nettoyage du texte (text_cleaner)
  4. Chunking parent-child (chunker)

Résultats sauvegardés dans data/chunks/ (un JSON par TdR).
Rapport JSON sauvegardé dans backend/logs/.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import colorlog
from tqdm import tqdm

from config.settings import FILTERED_DIR, CHUNKS_DIR, LOGS_DIR
from pipeline.step2_chunking.text_extractor import extract_full_text
from pipeline.step2_chunking.metadata_extractor import extract_metadata
from pipeline.step2_chunking.text_cleaner import clean_text
from pipeline.step2_chunking.chunker import create_chunks


def _setup_logger() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red"},
    ))
    logger = logging.getLogger("chunking_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def run() -> list[dict]:
    """
    Lance le pipeline de chunking sur tous les PDFs de filtered_tdrs/.
    Retourne la liste des rapports par fichier.
    """
    logger = _setup_logger()

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(FILTERED_DIR.glob("*.pdf")) + sorted(FILTERED_DIR.glob("*.PDF"))
    logger.info(f"{len(pdfs)} TdRs trouvés dans {FILTERED_DIR}")

    reports = []
    stats = {
        "total": len(pdfs),
        "success": 0,
        "error": 0,
        "total_parents": 0,
        "total_children": 0,
    }

    for pdf_path in tqdm(pdfs, desc="Chunking TdRs", unit="fichier"):
        report = {
            "file": pdf_path.name,
            "is_scanned": False,
            "total_chars": 0,
            "nb_parents": 0,
            "nb_children": 0,
            "metadata_error": None,
            "error": None,
        }

        # ── Étape 1 : extraction texte complet ────────────────────────────
        extract_result = extract_full_text(pdf_path)
        if extract_result["error"]:
            report["error"] = extract_result["error"]
            stats["error"] += 1
            logger.error(f"[ERREUR] {pdf_path.name} — {extract_result['error']}")
            reports.append(report)
            continue

        report["is_scanned"] = extract_result["is_scanned"]
        raw_text = extract_result["text"]

        if not raw_text:
            report["error"] = "Texte vide après extraction"
            stats["error"] += 1
            logger.warning(f"[VIDE]   {pdf_path.name}")
            reports.append(report)
            continue

        # ── Étape 2 : extraction métadonnées via Ollama ───────────────────
        meta = extract_metadata(raw_text)
        report["metadata_error"] = meta.pop("error", None)

        # ── Étape 3 : nettoyage du texte ──────────────────────────────────
        cleaned = clean_text(raw_text, is_scanned=extract_result["is_scanned"])
        report["total_chars"] = len(cleaned)

        # ── Étape 4 : chunking parent-child ───────────────────────────────
        chunks = create_chunks(cleaned, pdf_path.stem)
        report["nb_parents"]  = len(chunks["parents"])
        report["nb_children"] = len(chunks["children"])

        # ── Sauvegarde JSON ───────────────────────────────────────────────
        output = {
            "source":     pdf_path.name,
            "is_scanned": extract_result["is_scanned"],
            "page_count": extract_result["page_count"],
            "total_chars": len(cleaned),
            "metadata":   meta,
            "parents":    chunks["parents"],
            "children":   chunks["children"],
        }

        out_path = CHUNKS_DIR / f"{pdf_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        stats["success"]        += 1
        stats["total_parents"]  += report["nb_parents"]
        stats["total_children"] += report["nb_children"]

        logger.info(
            f"[OK] {pdf_path.name} — "
            f"{report['total_chars']} chars | "
            f"{report['nb_parents']} parents | "
            f"{report['nb_children']} children"
        )
        reports.append(report)

    # ── Rapport JSON ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = LOGS_DIR / f"chunking_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "files": reports}, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\nTerminé — Succès: {stats['success']} | Erreurs: {stats['error']} | "
        f"Parents: {stats['total_parents']} | Children: {stats['total_children']}"
    )
    logger.info(f"Rapport sauvegardé : {report_path}")

    return reports
