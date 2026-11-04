# Implementation Plan & Requirement Checklist (`IMPLEMENTATION_PLAN.md`)

## 1. Execution Phases & Status Summary (v2.0)

| Phase | Milestone Description | Status | Verification Method |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Rule Engine Expansion (19 Categories, Dividend Priority Fix, International Rails & Merchants) | **COMPLETED** | `test_04_rule_classifier_all_19_categories_and_dividend_priority` (PASS) |
| **Phase 2** | ML Model Enhancement (`data/training_corpus.csv` 2,618 samples, Dual-Head Sub-Category ML, 80/20 Split & 5-Fold CV) | **COMPLETED** | `test_05` — 99.89% 5-Fold CV Accuracy, 99.78% Macro-F1, 100.0% Sub-Category Test Acc |
| **Phase 3** | Extraction Improvements (Regional Date Disambiguation, Per-Page Error Isolation, 7 Currencies & European Decimals) | **COMPLETED** | `test_02`, `test_03`, `test_07` (PASS across 24 date formats & 7 currencies) |
| **Phase 4** | System Hardening (Python `logging` Across All Modules, `BackgroundTask` Temp File Cleanup, Expanded Unit Tests) | **COMPLETED** | All 8 test suites in `tests/test_system.py` passed in 46.4s |
| **Phase 5** | Optional Transformer Backend (`src/classification/transformer_classifier.py` & `notebooks/kaggle_distilbert_finetuning.ipynb`) | **COMPLETED** | Integrated into `HybridClassificationEngine` |

## 2. Detailed Phase 1–5 Requirements Checklist

- [x] **Item 1**: Added 5 missing categories (`Education & Learning`, `Government & Taxes`, `Charity & Donations`, `Personal Care & Wellness`, `Wallet & Recharge`) to `src/classification/rule_classifier.py` and `src/utils/models.py`.
- [x] **Item 2**: Moved `RULE_INTEREST_DIVIDEND` to highest priority (index 0) above `RULE_SALARY_PAYROLL` with `EQUITY DIVIDEND`, `FINAL DIVIDEND`, `INTERIM DIVIDEND` and `CREDIT` polarity filtering.
- [x] **Item 3**: Added international payment rails (`Wire Transfer`, `ACH`, `Direct Debit`, `SEPA`, `Zelle`, `Venmo`, `CashApp`) and international merchants (`Target`, `Costco`, `CVS`, `Walgreens`, `Tesco`, `Sainsbury`).
- [x] **Item 4**: Expanded training corpus from 114 to **2,618 samples** (1,000+ international patterns, 1,070 P2P transfer variations, OCR-corrupted edge cases, and all 19 categories).
- [x] **Item 5**: Implemented 80/20 stratified train/test split, 5-fold stratified cross-validation with macro-F1, per-class Precision/Recall/F1 table, and 19×19 confusion matrix.
- [x] **Item 6**: Externalized training corpus to `data/training_corpus.csv` and loaded dynamically at training time.
- [x] **Item 7**: Implemented Dual-Head Sub-Category ML prediction in `TraditionalMLClassifier` and updated `hybrid_engine.py` to use `ml_res.sub_category` when deterministic rules do not match.
- [x] **Item 8**: Fixed date format ambiguity in `src/extraction/statement_extractor.py` using `BANK_DATE_FORMAT_HINTS` (`DD/MM/YYYY` for Indian/UK/EU banks vs `MM/DD/YYYY` for US banks).
- [x] **Item 9**: Added per-page `try/except` error isolation in `pdf_detector.py` and `statement_extractor.py` to log warnings and return partial results.
- [x] **Item 10**: Added multi-currency detection (`EUR`/`€`, `GBP`/`£`, `AED`, `SGD`, `JPY`/`¥`, `USD`/`$`, `INR`/`₹`) and European comma-as-decimal parsing (`1.234,56`).
- [x] **Item 11**: Replaced `print()` calls with structured `logging.getLogger(__name__)` (`DEBUG`, `INFO`, `WARNING`) across all modules.
- [x] **Item 12**: Added comprehensive unit tests in `tests/test_system.py` covering all 19 categories, positive/negative polarity, 24 date formats, Indian/US/European amounts, deliberate balance mismatches, and per-page error isolation.
- [x] **Item 14**: Cleaned up temporary file management in `api.py` using `BackgroundTask(_cleanup_files, ...)` and context managers.
- [x] **Item 15**: Created `src/classification/transformer_classifier.py` (`TransformerMLClassifier`) and `notebooks/kaggle_distilbert_finetuning.ipynb`, wired into `hybrid_engine.py`.
