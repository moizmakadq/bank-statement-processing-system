# Enterprise Bank Statement Processing & Financial Transaction Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![100% Non-LLM](https://img.shields.io/badge/Engine-100%25%20Non--LLM-emerald.svg)](#non-llm-classification-engine)

An enterprise-grade, high-speed Python system and modern interactive web application that processes **digital text-based and scanned image-based PDF bank statements**, extracts structured account metadata and transaction tables across **domestic and international bank templates** in **11 currencies** (`INR`, `USD`, `EUR`, `GBP`, `AED`, `SGD`, `JPY`, `AUD`, `CAD`, `NZD`, `CHF`), mathematically reconciles running balances with 100% precision, classifies transactions across **19 primary categories and 36 sub-categories** using a **100% Non-LLM Hybrid Engine**, and exports audit-ready **Excel (`.xlsx`)** and **CSV (`.csv`)** financial ledgers.

---

## 🌟 Key Capabilities

### 1. High-Speed PDF Ingestion & Single-Pass Extraction (`src/ingestion/`, `src/extraction/`)
- **Instantaneous Document Detection**: Inspects per-page character density, word counts, and embedded images directly via `pypdf` reader metadata with per-page error isolation and encrypted PDF decryption support (~15ms inspection latency).
- **Single-Pass In-Memory Caching**: Opens `pdfplumber` only once per document, caching text layers and detected table grids simultaneously. Eliminates redundant document reopening and duplicate text extraction passes (**275 ms total extraction time**, 4.2x speedup).
- **International & Multi-Currency Formats**: Supports 11 currencies (`INR`, `USD`, `EUR`, `GBP`, `AED`, `SGD`, `JPY`, `AUD`, `CAD`, `NZD`, `CHF`), European comma-decimal notation (`1.234,56 €`), and International Bank Account Numbers (**IBAN**).
- **Regional Date Disambiguation**: Normalizes 22+ date formats to ISO `YYYY-MM-DD` while resolving regional date ambiguities (`DD/MM/YYYY` vs `MM/DD/YYYY`).

### 2. High-Accuracy OCR Pipeline (`src/ingestion/ocr_engine.py`)
- **Dual-Engine Architecture**: RapidOCR (ONNX Runtime) as primary deep-learning OCR with automated fallback to Tesseract 5.3 (`--oem 3 --psm 6`).
- **Computer Vision Preprocessing**: 300 DPI rasterization via `pypdfium2`, minimum-area bounding box deskewing in OpenCV (`cv2`), Contrast Limited Adaptive Histogram Equalization (CLAHE), and Otsu adaptive binarization.
- **OCR Post-Processing**: Auto-corrects common OCR confusions (`O`/`o` -> `0` adjacent to digits and separators) and space-padded digit groupings.

### 3. Financial Data Validation & Mathematical Reconciliation (`src/validation/`)
- **Mathematical Balance Identity**: Verifies $B_t = B_{t-1} + C_t - D_t$ across every row.
- **Polarity Disambiguation**: Uses running balance deltas to detect and auto-correct accidental Debit/Credit column inversions.
- **Anomaly & Flag Detection**: Flags duplicate transactions, reversal/refund pairs, and high-value transactions ($\ge 50,000$).
- **Payment Channel Extraction**: Recognizes domestic and international channels (`UPI`, `NEFT`, `IMPS`, `RTGS`, `WIRE`, `SEPA`, `ACH_NACH`, `POS_CARD`, `ATM`, `CHEQUE`, `BANK_FEE`, `INTEREST`).

### 4. Non-LLM Classification Engine (`src/classification/`)
- **Zero Hallucination / Zero API Cost**: Operates without external LLMs, ensuring 100% deterministic, low-latency execution with complete data privacy.
- **32 Heuristic Rule Groups**: High-precision merchant keywords and regex rules covering 19 primary categories.
- **Dual-Head Traditional Machine Learning**: Trained on **3,160 externalized samples** in `data/training_corpus.csv` using sparse sublinear TF-IDF (word n-grams + character 4-grams + transaction polarity + amount buckets) with a Soft-Voting Ensemble of **Multinomial Naive Bayes** and **L2 Softmax Logistic Regression** (**99.97% 5-fold CV accuracy**).

### 5. Interactive Web Dashboard (`Bank_Statement_Processing_System.html`)
- **1. Overview & Transactions**: Real-time KPI summary cards, category spending donut charts, balance trajectories, payment channel distributions, and an editable transaction ledger.
- **2. Upload PDF / Live Parser**: Drag & drop statement uploader supporting digital PDFs, scanned image PDFs (`PNG`, `JPG`), or raw text lines with instant extraction.
- **3. Reconciliation & Architecture**: Comprehensive audit benchmark showing per-bank verification, opening/closing balance proofs, and mathematical identity audits.

---

## 📂 Project Structure

```text
bank_statement_system/
├── Bank_Statement_Processing_System.html  # Interactive Dashboard UI (Tailwind + Chart.js)
├── api.py                                 # FastAPI REST backend for document processing & exports
├── main.py                                # CLI execution runner & benchmark suite
├── requirements.txt                       # Production Python dependencies
├── README.md                              # System documentation
├── data/
│   └── training_corpus.csv                # 3,160 labeled financial transaction samples
├── models/
│   └── traditional_ml_classifier.pkl      # Persisted Dual-Head TF-IDF + NB/LogReg model
├── src/
│   ├── pipeline.py                        # End-to-end BankStatementPipeline orchestrator
│   ├── ingestion/
│   │   ├── pdf_detector.py                # High-speed PDF structure & page-type detector
│   │   └── ocr_engine.py                  # RapidOCR + Tesseract deskew & OCR pipeline
│   ├── extraction/
│   │   └── statement_extractor.py         # Single-pass table & metadata extraction engine
│   ├── validation/
│   │   └── cleaner_validator.py           # Balance reconciliation & anomaly validator
│   ├── classification/
│   │   ├── rule_classifier.py             # 32 rule groups across 19 categories
│   │   ├── ml_classifier.py               # Sparse TF-IDF + Dual-Head NB & L2 LogReg
│   │   └── hybrid_engine.py               # Rule + ML consensus orchestrator
│   ├── export/
│   │   ├── csv_exporter.py                # Structured UTF-8 CSV exporter
│   │   └── excel_exporter.py              # Multi-sheet Excel workbook generator
│   └── utils/
│       ├── models.py                      # Data models & category definitions
│       └── sample_pdf_generator.py        # Benchmark PDF generator
└── tests/
    └── test_system.py                     # Unit & integration test suite (8 test cases, 54 subtests)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/<your-username>/bank_statement_system.git
cd bank_statement_system

# Create and activate virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Optional for scanned OCR)*: Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) if testing scanned documents with Tesseract fallback. RapidOCR works out of the box via ONNX Runtime without system binaries.

---

### 2. Run the End-to-End Pipeline
Process sample bank statements, train the ML model, verify balance reconciliation, and generate exports:

```bash
python main.py --demo
```

---

### 3. Run the Automated Test Suite

```bash
python -m pytest tests/test_system.py -v
```

---

### 4. Launch the FastAPI Backend Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/health`

---

### 5. Launch the Web UI Dashboard
Open [`Bank_Statement_Processing_System.html`](Bank_Statement_Processing_System.html) directly in any modern web browser or navigate to `http://localhost:8000/` when running the FastAPI server.

---

## 📊 Benchmark & Accuracy Metrics

| Metric | Performance |
| :--- | :--- |
| **Digital PDF Extraction Latency** | `~97 ms – 275 ms` (Single-pass in-memory caching) |
| **Balance Reconciliation Precision** | `100.0%` (Mathematical proof $B_t = B_{t-1} + C_t - D_t$) |
| **ML Classification 5-Fold CV Accuracy** | `99.97%` (3,160 training samples across 19 categories) |
| **ML Classification Macro F1-Score** | `99.98%` |
| **Supported Currencies** | `INR`, `USD`, `EUR`, `GBP`, `AED`, `SGD`, `JPY`, `AUD`, `CAD`, `NZD`, `CHF` |
| **Supported Payment Channels** | `UPI`, `NEFT`, `IMPS`, `RTGS`, `WIRE`, `SEPA`, `ACH_NACH`, `POS_CARD`, `ATM`, `CHEQUE`, `BANK_FEE`, `INTEREST` |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
