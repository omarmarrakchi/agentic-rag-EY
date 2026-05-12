"""
Orchestre les 3 étapes du filtrage pour tous les PDFs du dossier raw_tdrs :
  1. Lecture + détection natif/scanné (pdf_reader)
  2. Scoring par mots-clés (keyword_scorer)
  3. Classification Ollama si ambigus (llm_classifier)

Résultats copiés dans filtered_tdrs/ ou rejected/.
Rapport JSON sauvegardé dans logs/.
"""

import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

import colorlog
from tqdm import tqdm

from config.settings import RAW_TDRS_DIR, FILTERED_DIR, REJECTED_DIR, LOGS_DIR
from pipeline.step1_filter.pdf_reader import read_pdf
from pipeline.step1_filter.keyword_scorer import compute_score
from pipeline.step1_filter.llm_classifier import classify_with_llm


def _setup_logger() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red"},
    ))
    logger = logging.getLogger("filter_pipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def run() -> list[dict]:
    """
    Lance le pipeline de filtrage sur tous les PDFs de raw_tdrs.
    Retourne la liste des rapports par fichier.
    """
    logger = _setup_logger()

    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_TDRS_DIR.glob("*.pdf"))
    logger.info(f"{len(pdfs)} PDFs trouvés dans {RAW_TDRS_DIR}")

    reports = []
    stats = {"tdr": 0, "rejected": 0, "error": 0, "llm_used": 0}

    for pdf_path in tqdm(pdfs, desc="Filtrage TdRs", unit="fichier"):
        report = {
            "file": pdf_path.name,
            "is_scanned": False,
            "score": None,
            "matched_strong": [],
            "matched_weak": [],
            "matched_excl": [],
            "scorer_verdict": None,
            "llm_used": False,
            "llm_verdict": None,
            "llm_reason": None,
            "final_verdict": None,
            "error": None,
        }

        # ── Étape 1 : lecture ──────────────────────────────────────────────
        read_result = read_pdf(pdf_path)
        if read_result["error"]:
            report["error"] = read_result["error"]
            report["final_verdict"] = "error"
            stats["error"] += 1
            logger.error(f"[ERREUR] {pdf_path.name} — {read_result['error']}")
            reports.append(report)
            continue

        report["is_scanned"] = read_result["is_scanned"]
        text = read_result["text"]

        # ── Étape 2 : scoring ──────────────────────────────────────────────
        score_result = compute_score(text)
        report.update({
            "score": score_result["score"],
            "matched_strong": score_result["matched_strong"],
            "matched_weak": score_result["matched_weak"],
            "matched_excl": score_result["matched_excl"],
            "scorer_verdict": score_result["verdict"],
        })

        # PDF scanné sans texte → forcer ambiguous pour qu'Ollama utilise
        # le nom du fichier comme contexte de secours
        if read_result["is_scanned"] and not text:
            score_result["verdict"] = "ambiguous"

        if score_result["verdict"] != "ambiguous":
            report["final_verdict"] = score_result["verdict"]
        else:
            # ── Étape 3 : Ollama pour les ambigus ─────────────────────────
            # Si le texte est vide (OCR a échoué), on donne au moins le nom
            # du fichier comme indice à Ollama
            context = text if text else f"[Fichier PDF scanné, texte non extrait]\nNom du fichier : {pdf_path.name}"
            report["llm_used"] = True
            stats["llm_used"] += 1
            llm_result = classify_with_llm(context)
            report["llm_verdict"] = llm_result["verdict"]
            report["llm_reason"]  = llm_result["reason"]

            if llm_result["error"]:
                report["error"] = llm_result["error"]
                # En cas d'erreur Ollama, on rejette par prudence
                report["final_verdict"] = "rejected"
                logger.warning(f"[LLM ERREUR] {pdf_path.name} — {llm_result['error']}")
            else:
                report["final_verdict"] = llm_result["verdict"]

        # ── Copie dans le bon dossier ──────────────────────────────────────
        dest_dir = FILTERED_DIR if report["final_verdict"] == "tdr" else REJECTED_DIR
        shutil.copy2(pdf_path, dest_dir / pdf_path.name)

        if report["final_verdict"] == "tdr":
            stats["tdr"] += 1
            logger.info(f"[TDR]      {pdf_path.name}  (score={report['score']})")
        else:
            stats["rejected"] += 1
            logger.info(f"[REJETÉ]   {pdf_path.name}  (score={report['score']})")

        reports.append(report)

    # ── Sauvegarde du rapport JSON ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = LOGS_DIR / f"filter_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "files": reports}, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\nTerminé — TdRs: {stats['tdr']} | Rejetés: {stats['rejected']} | "
        f"Erreurs: {stats['error']} | Ollama utilisé: {stats['llm_used']} fois"
    )
    logger.info(f"Rapport sauvegardé : {report_path}")

    return reports
