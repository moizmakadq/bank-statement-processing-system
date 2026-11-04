"""Image Preprocessing & OCR Engine for Scanned Bank Statements and Image Files.

Supports:
1. RapidOCR (ONNX Runtime) for pure Python, cross-platform OCR without external binaries
2. Tesseract OCR (via pytesseract) as secondary fallback if installed
3. OpenCV 300 DPI deskewing, Non-Local Means denoising, CLAHE contrast enhancement
4. Layout-preserving horizontal word-clustering into row-aligned tabular lines
5. Both PDF statement pages and direct image statement files (.png, .jpg, .jpeg, .tiff, .bmp, .webp)
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPID_OCR = True
except ImportError:
    HAS_RAPID_OCR = False

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)


@dataclass
class OCRPageResult:
    page_number: int
    text: str
    lines: List[str]
    confidence: float
    skew_angle_deg: float
    word_boxes: List[Dict[str, object]]


class OCREngine:
    """High-accuracy OpenCV + RapidOCR / Tesseract pipeline for scanned bank statement pages and image files."""

    IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"})

    def __init__(self, dpi: int = 300, lang: str = "eng"):
        self.dpi = dpi
        self.lang = lang
        self.scale = dpi / 72.0
        self.rapid_ocr = RapidOCR() if HAS_RAPID_OCR else None

    def render_pdf_page_to_image(self, pdf_path: str, page_index: int) -> np.ndarray:
        """Renders a 0-indexed PDF page to an OpenCV BGR numpy array at target DPI."""
        pdf = pdfium.PdfDocument(pdf_path)
        if page_index < 0 or page_index >= len(pdf):
            raise IndexError(f"Page index {page_index} out of range (0..{len(pdf)-1})")
        page = pdf[page_index]
        bitmap = page.render(scale=self.scale)
        pil_img = bitmap.to_pil()
        rgb = np.array(pil_img)
        if rgb.ndim == 2:
            return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @classmethod
    def load_image_to_bgr(cls, file_path: str) -> np.ndarray:
        """Loads an image file (PNG, JPG, TIFF, etc.) into an OpenCV BGR numpy array."""
        # Use PIL to safely handle Unicode paths on Windows
        pil_img = Image.open(file_path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def estimate_and_deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimates skew angle of scanned text lines and rotates image to horizontal."""
        inv = cv2.bitwise_not(gray)
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles: List[float] = []
        h, w = gray.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < w * 0.08 or ch > h * 0.15 or cw < ch * 2:
                continue
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            if abs(angle) <= 15.0:
                angles.append(angle)

        if not angles:
            return gray, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.15:
            return gray, 0.0

        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            gray,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return deskewed, round(median_angle, 2)

    @staticmethod
    def preprocess_image(bgr_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Applies grayscale conversion, deskewing, contrast stretching, denoising, and binarization."""
        if bgr_image.ndim == 3:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_image.copy()

        deskewed, skew_angle = OCREngine.estimate_and_deskew(gray)
        denoised = cv2.fastNlMeansDenoising(deskewed, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary, skew_angle

    def ocr_page(self, file_path: str, page_number: int = 1) -> OCRPageResult:
        """Runs full image preprocessing and layout-preserving OCR on a PDF page or image file."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.IMAGE_EXTENSIONS:
            logger.info("Running OCR on image file: %s", os.path.basename(file_path))
            bgr = self.load_image_to_bgr(file_path)
        else:
            logger.info("Running 300 DPI OCR on page %d of %s", page_number, os.path.basename(file_path))
            bgr = self.render_pdf_page_to_image(file_path, page_number - 1)

        processed_img, skew_angle = self.preprocess_image(bgr)

        # 1. Primary Engine: RapidOCR (ONNX Runtime)
        if self.rapid_ocr is not None:
            return self._ocr_with_rapidocr(processed_img, page_number, skew_angle)

        # 2. Secondary Engine: Tesseract OCR (if available)
        if pytesseract is not None:
            return self._ocr_with_pytesseract(processed_img, page_number, skew_angle)

        raise RuntimeError(
            "No OCR engine is available. Please install 'rapidocr_onnxruntime' or Tesseract-OCR."
        )

    def _ocr_with_rapidocr(self, processed_img: np.ndarray, page_number: int, skew_angle: float) -> OCRPageResult:
        """Extracts text, bounding boxes, and aligned lines using RapidOCR."""
        ocr_results, _ = self.rapid_ocr(processed_img)
        if not ocr_results:
            return OCRPageResult(
                page_number=page_number, text="", lines=[], confidence=0.0, skew_angle_deg=skew_angle, word_boxes=[]
            )

        words_info: List[Dict[str, object]] = []
        confidences: List[float] = []

        for item in ocr_results:
            box, txt, score = item[0], str(item[1]).strip(), float(item[2])
            if not txt:
                continue
            pts = np.array(box)
            left = float(np.min(pts[:, 0]))
            top = float(np.min(pts[:, 1]))
            right = float(np.max(pts[:, 0]))
            bottom = float(np.max(pts[:, 1]))
            conf_val = round(score * 100.0, 1) if score <= 1.0 else round(score, 1)
            confidences.append(conf_val)

            words_info.append({
                "text": txt,
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "width": int(right - left),
                "height": int(bottom - top),
                "conf": conf_val,
            })

        # Cluster words into horizontal row lines
        lines_list = self._cluster_words_into_lines(words_info)
        full_text = "\n".join(lines_list)
        avg_conf = round(float(np.mean(confidences)), 2) if confidences else 0.0

        logger.debug(
            "Page %d RapidOCR complete: %d lines, avg_conf=%.2f%%, skew=%.2f deg",
            page_number, len(lines_list), avg_conf, skew_angle
        )

        return OCRPageResult(
            page_number=page_number,
            text=full_text,
            lines=lines_list,
            confidence=avg_conf,
            skew_angle_deg=skew_angle,
            word_boxes=words_info,
        )

    def _ocr_with_pytesseract(self, processed_img: np.ndarray, page_number: int, skew_angle: float) -> OCRPageResult:
        """Fallback OCR implementation using Tesseract via pytesseract."""
        pil_img = Image.fromarray(processed_img)
        custom_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
        ocr_data = pytesseract.image_to_data(
            pil_img, lang=self.lang, config=custom_config, output_type=pytesseract.Output.DICT
        )

        words_info: List[Dict[str, object]] = []
        confidences: List[float] = []

        n_boxes = len(ocr_data.get("text", []))
        for i in range(n_boxes):
            txt = (ocr_data["text"][i] or "").strip()
            try:
                conf = float(ocr_data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0

            if not txt:
                continue
            if conf >= 0:
                confidences.append(conf)

            left = int(ocr_data["left"][i])
            width = int(ocr_data["width"][i])
            words_info.append({
                "text": txt,
                "left": left,
                "top": int(ocr_data["top"][i]),
                "right": left + width,
                "width": width,
                "height": int(ocr_data["height"][i]),
                "conf": conf,
            })

        lines_list = self._cluster_words_into_lines(words_info)
        full_text = "\n".join(lines_list)
        avg_conf = round(float(np.mean(confidences)), 2) if confidences else 0.0

        return OCRPageResult(
            page_number=page_number,
            text=full_text,
            lines=lines_list,
            confidence=avg_conf,
            skew_angle_deg=skew_angle,
            word_boxes=words_info,
        )

    @staticmethod
    def _cluster_words_into_lines(words_info: List[Dict[str, object]]) -> List[str]:
        """Clusters bounding boxes by vertical proximity and orders horizontally to form text rows."""
        if not words_info:
            return []

        words_sorted_y = sorted(words_info, key=lambda w: int(w["top"]))
        row_clusters: List[List[Dict[str, object]]] = []
        current_cluster: List[Dict[str, object]] = [words_sorted_y[0]]
        current_y = float(words_sorted_y[0]["top"])

        for w in words_sorted_y[1:]:
            wy = float(w["top"])
            wh = float(w["height"])
            tolerance = max(16.0, wh * 0.65)
            if abs(wy - current_y) <= tolerance:
                current_cluster.append(w)
                current_y = float(np.mean([float(item["top"]) for item in current_cluster]))
            else:
                row_clusters.append(current_cluster)
                current_cluster = [w]
                current_y = wy
        if current_cluster:
            row_clusters.append(current_cluster)

        lines_list: List[str] = []
        for cluster in row_clusters:
            cluster_sorted_x = sorted(cluster, key=lambda w: int(w["left"]))
            line_parts: List[str] = []
            prev_right: Optional[int] = None
            for w in cluster_sorted_x:
                left = int(w["left"])
                right = int(w.get("right", left + int(w["width"])))
                txt = str(w["text"])
                if prev_right is not None:
                    gap = left - prev_right
                    if gap > 60:
                        line_parts.append("    ")
                    elif gap > 20:
                        line_parts.append("  ")
                    else:
                        line_parts.append(" ")
                line_parts.append(txt)
                prev_right = right
            line_str = "".join(line_parts).strip()
            if line_str:
                lines_list.append(line_str)

        return lines_list
