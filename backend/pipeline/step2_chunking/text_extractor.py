"""
Extrait le texte complet d'un PDF.

Stratégie en 3 niveaux :
  1. pymupdf4llm — Markdown structuré (titres, paragraphes) — PDFs natifs
  2. PyMuPDF     — texte brut natif — fallback si pymupdf4llm échoue
  3. OCR (pytesseract + Pillow) — PDFs scannés sans couche texte
"""

import fitz
import pymupdf4llm
from pathlib import Path

from config.settings import TESSERACT_CMD, OCR_LANGUAGES, OCR_MAX_PAGES


def _page_count(pdf_path: Path) -> int:
    try:
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _ocr_pdf(pdf_path: Path) -> str:
    """Extrait le texte d'un PDF scanné via Tesseract OCR."""
    import pytesseract
    from PIL import Image
    import io

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    doc = fitz.open(str(pdf_path))
    pages_to_scan = min(len(doc), OCR_MAX_PAGES)
    texts = []

    for page_num in range(pages_to_scan):
        page = doc[page_num]
        # Rendu haute résolution (2x) pour meilleure qualité OCR
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
        texts.append(text)

    doc.close()
    return "\n\n".join(texts)


def extract_full_text(pdf_path: Path) -> dict:
    """
    Lit un PDF et retourne :
      - text       : texte extrait (Markdown si possible, texte brut sinon)
      - is_scanned : True si OCR utilisé
      - page_count : nombre de pages
      - error      : message d'erreur si échec total (None sinon)
    """
    result = {
        "text": "",
        "is_scanned": False,
        "page_count": _page_count(pdf_path),
        "error": None,
    }

    # ── Niveau 1 : pymupdf4llm → Markdown structuré ──────────────────────────
    try:
        markdown = pymupdf4llm.to_markdown(str(pdf_path))
        if len(markdown.strip()) >= 100:
            result["text"] = markdown
            return result
    except Exception:
        pass

    # ── Niveau 2 : PyMuPDF fallback (texte brut natif) ───────────────────────
    try:
        doc = fitz.open(str(pdf_path))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) >= 100:
            result["text"] = text.strip()
            return result
    except Exception:
        pass

    # ── Niveau 3 : OCR Tesseract (PDF scanné sans couche texte) ──────────────
    try:
        text = _ocr_pdf(pdf_path)
        if len(text.strip()) >= 100:
            result["text"] = text.strip()
            result["is_scanned"] = True
            return result
    except Exception as exc:
        result["error"] = str(exc)

    return result
