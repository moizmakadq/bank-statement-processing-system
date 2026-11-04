"""End-to-End Bank Statement Processing & Classification Pipeline with Python Logging."""

from __future__ import annotations
import logging
import os
import time
from typing import List, Optional
from src.classification.hybrid_engine import HybridClassificationEngine
from src.export.csv_exporter import CSVExporter
from src.export.excel_exporter import ExcelExporter
from src.extraction.statement_extractor import StatementExtractor
from src.utils.models import PDFType, StatementProcessingResult
from src.validation.cleaner_validator import CleanerAndValidator

logger = logging.getLogger(__name__)


class BankStatementPipeline:
    """Orchestrates PDF detection, OCR/text extraction, balance validation, non-LLM classification, and exports."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        corpus_csv_path: Optional[str] = None,
        ocr_dpi: int = 300,
    ):
        self.extractor = StatementExtractor(ocr_dpi=ocr_dpi)
        self.validator = CleanerAndValidator()
        self.classifier = HybridClassificationEngine(model_path=model_path, corpus_csv_path=corpus_csv_path)

    def process_pdf(self, pdf_path: str, password: str = "") -> StatementProcessingResult:
        """Runs the full pipeline on a PDF bank statement and returns StatementProcessingResult."""
        t0 = time.perf_counter()
        file_size = os.path.getsize(pdf_path)
        logger.info("Starting pipeline on %s (%d bytes)", os.path.basename(pdf_path), file_size)

        pdf_type, metadata, raw_txns, page_inspections, warnings = self.extractor.extract_from_pdf(
            pdf_path, password=password
        )

        cleaned_txns, reconciliation = self.validator.clean_and_reconcile(metadata, raw_txns)
        classified_txns, cat_summary, month_summary = self.classifier.classify_all(cleaned_txns)

        ocr_confs = [p.ocr_confidence for p in page_inspections if p.ocr_confidence is not None]
        avg_ocr_conf = round(sum(ocr_confs) / len(ocr_confs), 2) if ocr_confs else None
        ocr_applied = pdf_type in (PDFType.IMAGE_BASED, PDFType.HYBRID)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info(
            "Completed %s in %.1f ms (type=%s, txns=%d, reconciled=%.1f%%)",
            os.path.basename(pdf_path),
            elapsed_ms,
            pdf_type.value,
            len(classified_txns),
            reconciliation.reconciliation_rate,
        )

        return StatementProcessingResult(
            file_name=os.path.basename(pdf_path),
            file_size_bytes=file_size,
            pdf_type=pdf_type,
            total_pages=len(page_inspections),
            processing_time_ms=elapsed_ms,
            ocr_applied=ocr_applied,
            average_ocr_confidence=avg_ocr_conf,
            metadata=metadata,
            transactions=classified_txns,
            reconciliation=reconciliation,
            category_summary=cat_summary,
            monthly_summary=month_summary,
            warnings=warnings,
        )

    @staticmethod
    def export_to_csv(result: StatementProcessingResult, csv_path: str) -> str:
        logger.info("Exporting CSV to %s", csv_path)
        return CSVExporter.export_transactions_csv(result, csv_path)

    @staticmethod
    def export_to_excel(results: List[StatementProcessingResult], excel_path: str) -> str:
        logger.info("Exporting multi-sheet Excel workbook to %s", excel_path)
        return ExcelExporter.export_workbook(results, excel_path)
