# Changelog (`CHANGELOG.md`)

All notable changes made during the development and Phase 1–5 expansion of the **Bank Statement Processing & Classification System** are documented in this file.

## [2.0.0] - 2026-11-04 — Phase 1 to Phase 5 System Improvements

### Added
- **Phase 1 — Rule Engine Expansion (`src/classification/rule_classifier.py`, `src/validation/cleaner_validator.py`)**:
  - Added 5 new categories to `SUPPORTED_CATEGORIES` and `HeuristicRuleClassifier` (expanding from 14 to **19 categories**):
    1. `Education & Learning` (`EdTech`, `Online Courses`, `School/College Fees`) — keywords: `BYJU`, `UNACADEMY`, `COURSERA`, `UDEMY`, `SCHOOL FEE`, `TUITION`, `UNIVERSITY`, `EDX`, `UPGRAD`, `VEDANTU`.
    2. `Government & Taxes` (`Income Tax`, `Property Tax`, `Government Services`) — keywords: `TIN NSDL`, `CBDT`, `INCOME TAX`, `PROPERTY TAX`, `MCA`, `PASSPORT`, `CHALLAN`, `IRS TAX`, `HMRC`.
    3. `Charity & Donations` (`Religious`, `Crowdfunding`, `NGO`) — keywords: `DONATION`, `CHARITY`, `TEMPLE`, `MOSQUE`, `CHURCH`, `KETTO`, `MILAAP`, `NGO`, `GIVEINDIA`, `RED CROSS`, `UNICEF`.
    4. `Personal Care & Wellness` (`Gym`, `Salon`, `Wellness`) — keywords: `CULT FIT`, `URBAN COMPANY`, `SALON`, `SPA`, `GYM`, `FITNESS`, `LAKME SALON`, `PLANET FITNESS`, `EQUINOX`.
    5. `Wallet & Recharge` (`Digital Wallet Top-ups`) — keywords: `PAYTM WALLET`, `PHONEPE WALLET`, `AMAZON PAY LOAD`, `FREECHARGE`, `MOBIKWIK`.
  - Added international payment rails (`WIRE`, `SEPA`, `DIRECT_DEBIT`, `ZELLE`, `VENMO`, `CASHAPP`, `FASTER PAYMENTS`, `BACS`) and international retail/pharmacy/supermarket merchants (`Target`, `Costco`, `CVS`, `Walgreens`, `Tesco`, `Sainsbury`, `Waitrose`, `Asda`, `Aldi`, `Lidl`, `Best Buy`, `Home Depot`).
- **Phase 2 — ML Model Enhancement (`data/training_corpus.csv`, `src/classification/ml_classifier.py`, `src/classification/hybrid_engine.py`)**:
  - Externalized training data to `data/training_corpus.csv` with **2,618 labeled samples** across 19 categories and 36 sub-categories (including 1,000+ international transactions, 1,070 P2P transfer variations, and 15% OCR-corrupted/abbreviated edge cases).
  - Implemented **Dual-Head ML Prediction** in `TraditionalMLClassifier`: Head 1 predicts primary `category` (19 classes) and Head 2 predicts `sub_category` (36 classes) conditioned on TF-IDF features and predicted primary category.
  - Updated `HybridClassificationEngine.classify_transaction()` to assign `ml_res.sub_category` whenever deterministic rules do not match.
  - Implemented `evaluate_train_test_split(test_ratio=0.20)` (80/20 stratified split, per-class Precision/Recall/F1/Support, and 19×19 confusion matrix) and `evaluate_cross_validation(n_folds=5)` (5-fold stratified CV achieving **99.89% ± 0.15% accuracy** and **99.78% ± 0.27% macro-F1**).
  - Accelerated TF-IDF feature matrix construction and L-BFGS-B optimization by 25x using `scipy.sparse.csr_matrix`.
- **Phase 3 — Extraction Improvements (`src/extraction/statement_extractor.py`)**:
  - Resolved regional date format ambiguity via `BANK_DATE_FORMAT_HINTS` (`DD/MM/YYYY` for Indian/UK/EU banks vs `MM/DD/YYYY` for US banks such as Chase, Citibank, Bank of America, and Wells Fargo).
  - Added per-page `try/except` error isolation in `PDFDetector.inspect_pdf` and `StatementExtractor.extract_from_pdf` so failed pages log warnings and return partial results.
  - Added multi-currency detection (`INR`, `USD`, `EUR`, `GBP`, `AED`, `SGD`, `JPY`) and European comma-as-decimal amount parsing (`1.234,56 €`).
- **Phase 4 — System Hardening (`api.py`, `src/**`, `tests/test_system.py`)**:
  - Replaced `print()` statements with structured Python `logging` (`DEBUG`, `INFO`, `WARNING`) across all modules.
  - Hardened temporary file cleanup in `api.py` using `starlette.background.BackgroundTask` so exported `.csv` and `.xlsx` temporary files are unlinked immediately after `FileResponse` completes.
  - Expanded `tests/test_system.py` to test all 19 rule categories, dividend priority, 24 date format variations, regional date ambiguity, European decimals, multi-currency headers, 2,618-sample ML evaluation, deliberate balance mismatches, and per-page error isolation.
- **Phase 5 — Optional Transformer Backend (`src/classification/transformer_classifier.py`, `notebooks/kaggle_distilbert_finetuning.ipynb`)**:
  - Added `TransformerMLClassifier` supporting `all-MiniLM-L6-v2` / `DistilBERT` embeddings from `models/distilbert_classifier/` and created the companion Kaggle GPU notebook.

### Fixed
- **Dividend vs. Salary Rule Priority (`src/classification/rule_classifier.py`)**:
  - Moved `RULE_INTEREST_DIVIDEND` to index 0 (above `RULE_SALARY_PAYROLL`) and added explicit patterns for `EQUITY DIVIDEND`, `FINAL DIVIDEND`, `INTERIM DIVIDEND`, `SPECIAL DIVIDEND`, and `DIVIDEND WARRANT` with `CREDIT` polarity filtering.

## [1.0.0] - 2026-11-04 — Initial End-to-End Implementation
- Initial implementation of PDF auto-detection, 300 DPI OpenCV + Tesseract OCR engine, multi-bank extraction, running balance reconciliation, 14-category non-LLM hybrid classifier, CSV/Excel exporters, FastAPI server, and interactive Web UI.
