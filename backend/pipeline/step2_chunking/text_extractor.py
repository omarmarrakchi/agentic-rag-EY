"""
Extrait le texte COMPLET d'un PDF (sans limite de caractères).
Réutilise la même logique que l'étape 1 (natif vs OCR),
mais sans tronquer pour avoir tout le contenu du TdR.
"""

import fitz
import pytesseract
from PIL import Image
from pathlib import Path

from config.settings import TESSERACT_CMD, OCR_LANGUAGES

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

_MIN_NATIVE_CHARS = 100


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

    try:
        doc = fitz.open(str(pdf_path))
        result["page_count"] = len(doc)

        # Tente l'extraction native sur toutes les pages
        native_text = ""
        for page in doc:
            native_text += page.get_text()

        if len(native_text.strip()) >= _MIN_NATIVE_CHARS:
            result["text"] = native_text.strip()
            result["is_scanned"] = False
        else:
            # PDF scanné : OCR sur toutes les pages
            result["is_scanned"] = True
            mat = fitz.Matrix(300 / 72, 300 / 72)
            ocr_text = ""
            try:
                for i in range(len(doc)):
                    pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text += pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
            except Exception:
                pass
            result["text"] = ocr_text.strip()

        doc.close()

    except Exception as exc:
        result["error"] = str(exc)

    return result
