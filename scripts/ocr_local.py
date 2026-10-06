"""OCR fallback: returns page dicts like scraper.extract_pages(), for scanned PDFs.
Higher DPI + preprocessing + PSM tuning for better digit/Arabic accuracy."""
from pathlib import Path

def _preprocess(img):
    """Grayscale + contrast boost + upscale — improves OCR accuracy on scanned docs,
    especially small stamp-box digits and Arabic diacritics."""
    from PIL import ImageOps, ImageEnhance
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img

def ocr_pages(pdf_path: Path, dpi: int = 400) -> list[dict]:
    from pdf2image import pdfinfo_from_path, convert_from_path
    import pytesseract

    info = pdfinfo_from_path(str(pdf_path), poppler_path=r"C:\poppler-26.02.0\Library\bin")
    total_pages = info["Pages"]

    pages = []
    for page_num in range(1, total_pages + 1):
        # process one page at a time -- keeps memory flat regardless of document size,
        # instead of loading every page's full-res image into RAM upfront
        imgs = convert_from_path(
            str(pdf_path), dpi=dpi, poppler_path=r"C:\poppler-26.02.0\Library\bin",
            first_page=page_num, last_page=page_num,
        )
        if not imgs:
            continue
        img = _preprocess(imgs[0])
        text = pytesseract.image_to_string(img, lang="eng+ara", config="--psm 6").strip()
        if len(text) < 30:
            text = pytesseract.image_to_string(img, lang="eng+ara", config="--psm 3").strip()
        if len(text) >= 30:
            pages.append({"page": page_num, "text": text, "total_pages": total_pages})
        del imgs, img  # explicit release before next page
    return pages

def ocr_digits_region(img, config: str = "--psm 7 -c tessedit_char_whitelist=0123456789/"):
    """Specialized OCR pass for a digit-heavy region (e.g. stamp box), if needed later."""
    import pytesseract
    return pytesseract.image_to_string(img, config=config).strip()
