"""PDF Type Auto-Detection & Inspection Module with Per-Page Error Isolation and Logging."""

from __future__ import annotations
import logging
import os
import re
from typing import List, Tuple
import pypdf
from src.utils.models import PDFType, PageInspection

logger = logging.getLogger(__name__)


class PDFDetector:
    """Detects PDF structure, encryption, and per-page text vs image characteristics."""

    MIN_CHARS_FOR_TEXT_PAGE = 60
    MIN_WORDS_FOR_TEXT_PAGE = 10

    def __init__(self, min_chars: int = 60, min_words: int = 10):
        self.min_chars = min_chars
        self.min_words = min_words

    def inspect_pdf(self, pdf_path: str, password: str = "") -> Tuple[PDFType, List[PageInspection], List[str]]:
        """Inspects a PDF file and returns overall PDFType, per-page details, and warnings."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if os.path.getsize(pdf_path) == 0:
            raise ValueError(f"PDF file is empty (0 bytes): {pdf_path}")

        warnings: List[str] = []

        ext = os.path.splitext(pdf_path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
            msg = f"Detected image bank statement ({ext.upper()}); routing directly through OCR pipeline."
            warnings.append(msg)
            logger.info(msg)
            inspections = [
                PageInspection(
                    page_number=1,
                    char_count=0,
                    word_count=0,
                    image_count=1,
                    has_tables=False,
                    page_type=PDFType.IMAGE_BASED,
                )
            ]
            return PDFType.IMAGE_BASED, inspections, warnings

        # 1. Check encryption & basic structure via pypdf
        try:
            reader = pypdf.PdfReader(pdf_path)
            if reader.is_encrypted:
                if password:
                    decrypt_status = reader.decrypt(password)
                    if decrypt_status == 0:
                        raise PermissionError("Invalid password for encrypted PDF statement.")
                    msg = "Encrypted PDF successfully unlocked with provided password."
                    warnings.append(msg)
                    logger.info(msg)
                else:
                    if reader.decrypt("") == 0:
                        raise PermissionError("PDF statement is password-protected. Please provide password.")
        except PermissionError:
            raise
        except Exception as exc:
            raise ValueError(f"Corrupted or invalid PDF file: {exc}") from exc

        page_inspections: List[PageInspection] = []
        text_pages = 0
        image_pages = 0

        # 2. Inspect page-by-page directly via loaded pypdf reader (instantaneous inspection)
        try:
            if len(reader.pages) == 0:
                raise ValueError("PDF contains 0 pages.")

            for idx, page in enumerate(reader.pages, start=1):
                try:
                    raw_text = (page.extract_text() or "").strip()
                    char_count = len(raw_text)
                    word_count = len(raw_text.split())
                    try:
                        image_count = len(page.images) if hasattr(page, "images") else 0
                    except Exception:
                        image_count = 0

                    has_tables = bool(
                        re.search(
                            r"\b(?:date|datum|txn|buchung)\b.*?(?:balance|saldo|debit|credit|amount)",
                            raw_text,
                            re.IGNORECASE,
                        )
                    )

                    if char_count >= self.min_chars and word_count >= self.min_words:
                        p_type = PDFType.TEXT_BASED
                        text_pages += 1
                    else:
                        p_type = PDFType.IMAGE_BASED
                        image_pages += 1

                    page_inspections.append(
                        PageInspection(
                            page_number=idx,
                            char_count=char_count,
                            word_count=word_count,
                            image_count=image_count,
                            has_tables=has_tables,
                            page_type=p_type,
                        )
                    )
                    logger.debug(
                        "Page %d inspected: type=%s, chars=%d, words=%d, tables=%s",
                        idx, p_type.value, char_count, word_count, has_tables,
                    )
                except Exception as page_exc:
                    warn_msg = f"Page {idx} inspection encountered an error ({page_exc}); falling back to IMAGE_BASED OCR."
                    logger.warning(warn_msg)
                    warnings.append(warn_msg)
                    image_pages += 1
                    page_inspections.append(
                        PageInspection(
                            page_number=idx,
                            char_count=0,
                            word_count=0,
                            image_count=1,
                            has_tables=False,
                            page_type=PDFType.IMAGE_BASED,
                            error=str(page_exc),
                        )
                    )
        except Exception as exc:
            raise ValueError(f"Failed to inspect PDF pages: {exc}") from exc

        if text_pages > 0 and image_pages == 0:
            overall_type = PDFType.TEXT_BASED
        elif image_pages > 0 and text_pages == 0:
            overall_type = PDFType.IMAGE_BASED
            msg = f"Detected scanned/image-based PDF ({image_pages} page(s)); routing through OpenCV + Tesseract OCR pipeline."
            warnings.append(msg)
            logger.info(msg)
        else:
            overall_type = PDFType.HYBRID
            msg = f"Detected hybrid PDF ({text_pages} text page(s), {image_pages} scanned page(s)); combining native extraction and OCR."
            warnings.append(msg)
            logger.info(msg)

        return overall_type, page_inspections, warnings
