# Project Architecture (`PROJECT_ARCHITECTURE.md`)

## 1. High-Level Architectural Overview (v2.0)

The **Bank Statement Processing & Classification System** is architected as a five-stage, decoupled, deterministic + traditional machine learning pipeline designed to handle heterogeneous domestic and international PDF bank statements without relying on Large Language Models (LLMs).

```text
+-----------------------------------------------------------------------------------+
|                        INPUT LAYER (CLI / FastAPI / Web UI)                       |
|   Text-Based PDFs   |   Scanned / Image-Based PDFs   |   Hybrid Multi-Page PDFs   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 1: INGESTION, AUTO-DETECTION & ERROR ISOLATION (`src/ingestion/`)           |
|  • PDFDetector: Encryption check (pypdf), character/word density, per-page        |
|    try/except error isolation with fallback to IMAGE_BASED OCR                    |
|  • OCREngine: 300 DPI rasterization (pypdfium2) -> OpenCV Deskew (minAreaRect)    |
|    -> Non-Local Means Denoising -> CLAHE Contrast -> Otsu Binarization            |
|    -> Tesseract 5.3.0 (`--oem 3 --psm 6`) -> Y-Coordinate Row Clustering          |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 2: MULTI-BANK & MULTI-CURRENCY EXTRACTION (`src/extraction/`)               |
|  • AccountMetadata Extractor: 16 Bank Signatures, IFSC/MICR, 7 Currencies         |
|    (INR, USD, EUR, GBP, AED, SGD, JPY), Regional Date Hints (DD/MM vs MM/DD)      |
|  • Dual-Strategy Transaction Parser:                                              |
|    1. Structured Grid Table Parser (dynamic column mapping for Debit/Credit/Amt)  |
|    2. Coordinate Line & Regex Fallback Parser with European Decimal Support       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 3: CLEANING, NORMALIZATION & RECONCILIATION (`src/validation/`)             |
|  • ISO Date Normalization (22+ date formats + regional preference + OCR `O->0`)   |
|  • Domestic & International Rail Detection (UPI, NEFT, IMPS, RTGS, WIRE, SEPA,    |
|    DIRECT_DEBIT, ZELLE, VENMO, CASHAPP, POS_CARD, ATM, ACH_NACH, CHEQUE, etc.)    |
|  • Mathematical Running Balance Audit: Opening + Credit - Debit == Closing        |
|  • Auto-Polarity Swap Repair, Duplicate Fingerprint Detection, Reversal Flagging  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 4: NON-LLM HYBRID CLASSIFICATION ENGINE (`src/classification/`)             |
|  • Tier 1: HeuristicRuleClassifier (32 regex & merchant rules, 19 categories,     |
|            RULE_INTEREST_DIVIDEND prioritized above RULE_SALARY_PAYROLL)          |
|  • Tier 2: Dual-Head TraditionalMLClassifier trained on `data/training_corpus.csv`|
|            (2,618 samples, Sparse CSR TF-IDF [Word 1,2-g + Char 4-g] +            |
|            Multinomial NB + L2 Softmax LogReg predicting Category & Sub-Category) |
|  • Tier 3: Optional TransformerMLClassifier (`models/distilbert_classifier/`)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 5: EXPORT & PRESENTATION LAYER (`src/export/` & Web UI)                     |
|  • CSVExporter: UTF-8 tabular ledger with 24 enriched columns                     |
|  • ExcelExporter: 3-sheet openpyxl workbook with SUM/SUMIF/COUNTIF & BarCharts    |
|  • FastAPI Server (`api.py`) with BackgroundTask temp file cleanup                |
+-----------------------------------------------------------------------------------+
```

## 2. Core Modules & Responsibilities

| Module Path | Primary Class | Responsibility |
| :--- | :--- | :--- |
| `src/utils/models.py` | `AccountMetadata`, `TransactionRecord`, `ReconciliationSummary`, `StatementProcessingResult`, `SUPPORTED_CATEGORIES` | Strongly typed dataclasses and 19 category definitions enforcing schema consistency across all pipeline stages. |
| `src/utils/generate_training_corpus.py` | `build_training_corpus` | Generates `data/training_corpus.csv` containing 2,618 labeled domestic, international, P2P, and OCR-corrupted banking transactions. |
| `src/ingestion/pdf_detector.py` | `PDFDetector` | Inspects PDF encryption status and per-page character/word density with per-page `try/except` error isolation. |
| `src/ingestion/ocr_engine.py` | `OCREngine` | Renders image-based pages at 300 DPI, performs OpenCV deskewing and CLAHE/Otsu binarization, runs Tesseract 5.3 OCR, and reconstructs horizontal table rows. |
| `src/extraction/statement_extractor.py` | `StatementExtractor` | Identifies 16 bank templates, resolves `DD/MM/YYYY` vs `MM/DD/YYYY` date ambiguity, detects 7 currencies, parses Indian/US/European amounts, and isolates per-page parsing errors. |
| `src/validation/cleaner_validator.py` | `CleanerAndValidator` | Extracts domestic and international payment channels and counterparties, reconciles running balances row-by-row, auto-corrects swapped Debit/Credit columns, and flags duplicates/reversals. |
| `src/classification/rule_classifier.py` | `HeuristicRuleClassifier` | Applies 32 deterministic regex rules across 19 financial categories with strict Debit/Credit polarity filtering and dividend priority over salary. |
| `src/classification/ml_classifier.py` | `TraditionalMLClassifier` | Implements sparse L2-normalized sublinear TF-IDF and a Dual-Head Soft-Voting Ensemble (Multinomial NB + L2 Logistic Regression) predicting both `category` (19 classes) and `sub_category` (36 classes), with 80/20 split and 5-fold CV evaluation. |
| `src/classification/transformer_classifier.py` | `TransformerMLClassifier` | Optional Phase 5 backend loading `all-MiniLM-L6-v2` / `DistilBERT` embeddings and linear head weights from `models/distilbert_classifier/`. |
| `src/classification/hybrid_engine.py` | `HybridClassificationEngine` | Combines Rule Engine, Dual-Head Traditional ML (using ML sub-categories on fallback), and optional Transformer predictions. |
| `src/export/csv_exporter.py` | `CSVExporter` | Exports 24-column UTF-8 CSV transaction files. |
| `src/export/excel_exporter.py` | `ExcelExporter` | Builds a 3-sheet formula-driven Excel (`.xlsx`) report with embedded `openpyxl` BarCharts. |
| `api.py` | `FastAPI` app | REST API with deterministic `BackgroundTask` temporary file cleanup and structured logging. |
