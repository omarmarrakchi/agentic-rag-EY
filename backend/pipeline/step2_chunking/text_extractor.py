"""
Extrait le texte COMPLET d'un PDF avec support des tableaux.

Stratégie en 3 niveaux :
  1. pdfplumber  — texte natif + tableaux structurés (pipe-separated)
  2. PyMuPDF     — fallback texte natif sans structure de tableau
  3. Tesseract   — fallback OCR pour les PDFs scannés

Les tableaux sont encadrés par [TABLEAU]...[/TABLEAU] pour que
le chunker puisse les traiter comme des blocs indivisibles.
"""

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path

from config.settings import TESSERACT_CMD, OCR_LANGUAGES

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

_MIN_NATIVE_CHARS = 100


def _format_table(table: list) -> str:
    """Convertit un tableau pdfplumber (liste de listes) en texte lisible."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [
            str(cell).strip().replace("\n", " ") if cell is not None else ""
            for cell in row
        ]
        if any(c for c in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_with_pdfplumber(pdf_path: Path) -> tuple[str, int]:
    """
    Extrait texte + tableaux structurés avec pdfplumber.
    Retourne (texte_complet, nombre_de_pages).
    Les tableaux sont injectés sous forme [TABLEAU]...[/TABLEAU].
    """
    pages_content = []

    with pdfplumber.open(str(pdf_path)) as doc:
        page_count = len(doc.pages)
        for page in doc.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            # Essai avec les lignes du PDF, puis fallback spacing
            tables = None
            try:
                tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                })
            except Exception:
                pass

            if not tables:
                try:
                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    })
                except Exception:
                    pass

            for table in (tables or []):
                formatted = _format_table(table)
                if formatted and len(formatted) > 15:
                    page_text += f"\n\n[TABLEAU]\n{formatted}\n[/TABLEAU]"

            pages_content.append(page_text)

    return "\n\n".join(pages_content).strip(), page_count


def extract_full_text(pdf_path: Path) -> dict:
    """
    Lit un PDF et retourne :
      - text       : texte complet extrait (str)
      - is_scanned : True si le PDF est une image scannée
      - page_count : nombre de pages
      - error      : message d'erreur si échec (None sinon)
    """
    result = {
        "text": "",
        "is_scanned": False,
        "page_count": 0,
        "error": None,
    }

    # ── Niveau 1 : pdfplumber (texte + tableaux structurés) ──────────────────
    try:
        text, page_count = _extract_with_pdfplumber(pdf_path)
        result["page_count"] = page_count
        if len(text) >= _MIN_NATIVE_CHARS:
            result["text"] = text
            return result
    except Exception:
        pass

    # ── Niveau 2 : PyMuPDF (fallback texte natif) ─────────────────────────────
    try:
        doc = fitz.open(str(pdf_path))
        result["page_count"] = len(doc)
        native_text = "".join(page.get_text() for page in doc)
        doc.close()
        if len(native_text.strip()) >= _MIN_NATIVE_CHARS:
            result["text"] = native_text.strip()
            return result
    except Exception:
        pass

    # ── Niveau 3 : OCR Tesseract (PDFs scannés) ───────────────────────────────
    try:
        result["is_scanned"] = True
        doc = fitz.open(str(pdf_path))
        result["page_count"] = len(doc)
        mat = fitz.Matrix(300 / 72, 300 / 72)
        ocr_text = ""
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text += pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
        doc.close()
        result["text"] = ocr_text.strip()
    except Exception as exc:
        result["error"] = str(exc)

    return result
