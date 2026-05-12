"""
Détecte si un PDF est natif (texte sélectionnable) ou scanné (image),
puis extrait les N premiers caractères pour l'analyse.
"""

import fitz  # pymupdf
from pathlib import Path

from config.settings import TEXT_EXTRACTION_CHARS


def _extract_native_text(doc: fitz.Document) -> str:
    """Extrait le texte des premières pages d'un PDF natif."""
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= TEXT_EXTRACTION_CHARS:
            break
    return text.strip()


def _has_enough_text(doc: fitz.Document, min_chars: int = 100) -> bool:
    """Retourne True si le PDF contient suffisamment de texte natif."""
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= min_chars:
            return True
    return False


def read_pdf(pdf_path: Path) -> dict:
    """
    Analyse un PDF et retourne un dict avec :
      - text      : texte extrait (str)
      - is_scanned: True si le PDF est une image scannée
      - page_count: nombre de pages
      - error     : message d'erreur si échec (None sinon)
    """
    result = {
        "text": "",
        "is_scanned": False,
        "page_count": 0,
        "error": None,
    }

    try:
        doc = fitz.open(str(pdf_path))
        result["page_count"] = len(doc)

        if _has_enough_text(doc):
            result["text"] = _extract_native_text(doc)[:TEXT_EXTRACTION_CHARS]
            result["is_scanned"] = False
        else:
            # PDF scanné : on tente un OCR sur la première page via pymupdf
            result["is_scanned"] = True
            page = doc[0]
            # get_textpage_ocr est disponible dans pymupdf >= 1.21
            try:
                tp = page.get_textpage_ocr(language="fra+eng", dpi=200, full=False)
                result["text"] = page.get_text(textpage=tp)[:TEXT_EXTRACTION_CHARS]
            except Exception:
                # Si l'OCR intégré échoue, on retourne le texte vide
                # (le scorer donnera un score bas → Ollama tranchera)
                result["text"] = ""

        doc.close()

    except Exception as exc:
        result["error"] = str(exc)

    return result
