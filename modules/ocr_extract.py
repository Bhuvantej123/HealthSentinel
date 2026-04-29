"""
Module 1: OCR Extraction Pipeline
Extracts text from uploaded medical report images or PDFs.
Optimized for speed: PyMuPDF direct extraction -> pytesseract -> EasyOCR
"""

import numpy as np
from pathlib import Path

# Global cache to prevent reloading the model on every function call
_easyocr_reader = None

def extract_text(file_path: str) -> str:
    """
    Main entry point. Accepts image or PDF path, returns extracted text.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        # SPEED OPTIMIZATION 1: Try direct PDF text extraction first
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text_parts = [page.get_text() for page in doc]
            full_text = "\n".join(text_parts).strip()
            doc.close()
            # If the PDF is digitally generated (not scanned), this returns the text instantly
            if len(full_text) > 50: 
                return full_text
        except Exception:
            pass
            
        # If direct extraction fails or is empty (scanned PDF), fallback to converting to images
        images = _pdf_to_images(path)
    else:
        images = [_load_image(path)]

    parts = []
    for img in images:
        text = _ocr_image(img)
        if text:
            parts.append(text)

    return "\n".join(parts).strip()


# ── PDF → images ──────────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: Path) -> list:
    """Convert each PDF page to a numpy RGB array using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA → RGB
                import cv2
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            images.append(arr)
        doc.close()
        return images
    except Exception as e:
        raise RuntimeError(f"PDF page conversion failed: {e}")


# ── Image loading ──────────────────────────────────────────────────────────────

def _load_image(image_path: Path) -> np.ndarray:
    """Load an image file as a numpy RGB array."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    from PIL import Image
    return np.array(Image.open(str(image_path)).convert("RGB"))


# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr_image(img_array: np.ndarray) -> str:
    """Run OCR on a numpy image. Optimized to try pytesseract first for speed."""

    # SPEED OPTIMIZATION 2: Primary is pytesseract (extremely fast)
    try:
        import pytesseract
        from PIL import Image
        pil_img = Image.fromarray(img_array)
        text = pytesseract.image_to_string(pil_img)
        if text.strip() and len(text.strip()) > 20:
            return text
    except Exception:
        pass

    # SPEED OPTIMIZATION 3: Fallback is EasyOCR, but cached so it only loads once
    try:
        global _easyocr_reader
        import easyocr
        if _easyocr_reader is None:
            # Only load the model if pytesseract failed or wasn't installed
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            
        results = _easyocr_reader.readtext(img_array, detail=0, paragraph=True)
        text = "\n".join(results)
        if text.strip():
            return text
    except Exception as e:
        return f"[OCR failed: {e}]"
        
    return ""
