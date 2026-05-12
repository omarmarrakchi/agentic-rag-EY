"""
Détecte si un PDF est natif (texte sélectionnable) ou scanné (image),
puis extrait les N premiers caractères pour l'analyse.

Stratégie OCR (PDFs scannés) :
  1. pytesseract sur les OCR_MAX_PAGES premières pages (Tesseract v5)
     → évite le cas où la page 1 est une couverture pauvre en texte
  2. Si pytesseract échoue → texte vide (le pipeline enverra le nom du
     fichier à Ollama comme contexte de secours)
"""

import fitz  # pymupdf
import pytesseract
from PIL import Image
from pathlib import Path

from config.settings import TEXT_EXTRACTION_CHARS, TESSERACT_CMD, OCR_LANGUAGES, OCR_MAX_PAGES

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _extract_native_text(doc: fitz.Document) -> str:
    """Extrait le texte des pages d'un PDF natif jusqu'à TEXT_EXTRACTION_CHARS."""
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


def _ocr_pages(doc: fitz.Document) -> str:
    """
    Applique pytesseract sur les OCR_MAX_PAGES premières pages.
    S'arrête dès que TEXT_EXTRACTION_CHARS est atteint.
    Retourne le texte reconnu ou une chaîne vide si l'OCR échoue.
    """
    mat = fitz.Matrix(300 / 72, 300 / 72)
    text = ""
    try:
        pages_to_scan = min(OCR_MAX_PAGES, len(doc))
        for i in range(pages_to_scan):
            pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
            if len(text) >= TEXT_EXTRACTION_CHARS:
                break
    except Exception:
        pass
    return text.strip()


def read_pdf(pdf_path: Path) -> dict:
    """
    Analyse un PDF et retourne un dict avec :
      - text      : texte extrait (str), peut être vide si OCR échoue
      - is_scanned: True si le PDF est une image scannée
      - page_count: nombre de pages
      - error     : message d'erreur si échec total (None sinon)
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
            result["is_scanned"] = True
            result["text"] = _ocr_pages(doc)[:TEXT_EXTRACTION_CHARS]

        doc.close()

    except Exception as exc:
        result["error"] = str(exc)

    return result
