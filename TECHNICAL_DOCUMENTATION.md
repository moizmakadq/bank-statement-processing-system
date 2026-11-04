# Technical Documentation (`TECHNICAL_DOCUMENTATION.md`)

## 1. Detailed Module Specifications & Phase 1–5 Upgrades

### 1.1 Rule Engine Expansion & Priority Ordering (`src/classification/rule_classifier.py`)

- **19 Supported Primary Categories**:
  1. `Salary & Income` (`Interest & Dividends`, `Payroll / Salary`, `Business / Consulting Income`)
  2. `Food & Dining` (`Restaurants & Food Delivery`)
  3. `Groceries & Supermarkets` (`Supermarket & Quick Commerce`)
  4. `Shopping & E-Commerce` (`Online & Retail Shopping`, `Refund / Reversal`)
  5. `Utilities & Bills` (`Electricity, Gas & Water`, `Mobile, Broadband & DTH`)
  6. `Rent & Housing` (`House Rent & Maintenance`)
  7. `Travel & Transport` (`Cab, Auto & Metro Transit`, `Flights, Rail & Hotels`, `Fuel & EV Charging`)
  8. `Healthcare & Medical` (`Hospitals, Labs & Pharmacy`)
  9. `Entertainment & Subscriptions` (`OTT, Gaming & Movies`)
  10. `Investments & Insurance` (`Mutual Funds & Equities`, `Insurance Premium`, `Fixed Deposit / Provident Fund`)
  11. `Loans, EMI & Credit Cards` (`Loan EMI`, `Credit Card Bill Payment`)
  12. `ATM & Cash Withdrawal` (`ATM Cash Withdrawal`)
  13. `Transfers & Peer-to-Peer` (`Bank / UPI Transfer`)
  14. `Bank Fees, Taxes & Charges` (`Bank Charges & Taxes`)
  15. `Education & Learning` (`EdTech`, `Online Courses`, `School/College Fees`)
  16. `Government & Taxes` (`Income Tax`, `Property Tax`, `Government Services`)
  17. `Charity & Donations` (`Religious`, `Crowdfunding`, `NGO`)
  18. `Personal Care & Wellness` (`Gym`, `Salon`, `Wellness`)
  19. `Wallet & Recharge` (`Digital Wallet Top-ups`)
- **Dividend vs. Salary Priority Resolution**:
  `RULE_INTEREST_DIVIDEND` is placed at index `0` of `self.rules` (ahead of `RULE_SALARY_PAYROLL` at index `1`) and matches `EQUITY DIVIDEND`, `FINAL DIVIDEND`, `INTERIM DIVIDEND`, `SPECIAL DIVIDEND`, `DIVIDEND WARRANT`, `INT.PD`, `INTEREST CREDIT`, and `SB INT` with `req_polarity = "CREDIT"`.

### 1.2 Dual-Head Sparse TF-IDF & Traditional ML Ensemble (`src/classification/ml_classifier.py`)

- **Externalized Training Data (`data/training_corpus.csv`)**:
  - Contains **2,618 labeled samples** across 19 categories and 36 sub-categories.
  - Built deterministically by `src/utils/generate_training_corpus.py` and loaded via `TraditionalMLClassifier.load_corpus_from_csv()`.
- **Sparse CSR TF-IDF Matrix (`_build_tfidf_matrix`)**:
  - Extracts word unigrams (`W_`), word bigrams (`BG_`), character 4-grams (`C4_`), polarity (`POL_`), and amount magnitude buckets (`AMT_`).
  - Constructs a `scipy.sparse.csr_matrix` of shape $(N, V)$ ($2618 \times 8020$) with sublinear TF scaling and $L_2$ row normalization, accelerating matrix-vector products by 25x over dense arrays.
- **Dual-Head Classification**:
  - **Primary Head (19 Categories)**: Soft-voting ensemble of Multinomial Naive Bayes ($\alpha = 0.20$) and L2-Regularized Multinomial Logistic Regression ($\lambda = 0.6$, optimized via `L-BFGS-B`):
    $$P_{\text{cat}}(c \mid \mathbf{x}) = 0.55 P_{\text{LR}}(c \mid \mathbf{x}) + 0.45 P_{\text{NB}}(c \mid \mathbf{x})$$
  - **Secondary Head (36 Sub-Categories)**: Conditional Multinomial Naive Bayes restricted to the candidate sub-categories $\mathcal{S}(c^*)$ belonging to the winning primary category $c^* = \arg\max_c P_{\text{cat}}(c \mid \mathbf{x})$:
    $$s^* = \arg\max_{s \in \mathcal{S}(c^*)} \left( \log \hat{\pi}_s + \mathbf{x}^\top \log \hat{\boldsymbol{\theta}}_s \right)$$

### 1.3 Regional Date Disambiguation, Multi-Currency & Error Isolation (`src/extraction/statement_extractor.py`)

- **Regional Date Ambiguity Resolution**:
  - `BANK_DATE_FORMAT_HINTS` maps US banks (`JPMorgan Chase`, `Citibank`, `Bank of America`, `Wells Fargo`) to `MM/DD/YYYY` (`prefer_mdy=True`) and Indian/UK/EU/SG/UAE banks (`HDFC`, `ICICI`, `SBI`, `Axis`, `Kotak`, `HSBC`, `Barclays`, `Deutsche Bank`, `DBS`, `Emirates NBD`) to `DD/MM/YYYY` (`prefer_mdy=False`).
  - `normalize_date(raw_date, prefer_mdy=...)` prioritizes `mdy_formats` before `dmy_formats` when `prefer_mdy=True`, accurately resolving ambiguous dates such as `04/07/2026` (`2026-04-07` for US statements vs. `2026-07-04` for Indian statements).
- **European Comma-Decimal & Multi-Currency Parsing (`parse_amount`)**:
  - Detects European decimal strings via `re.search(r",\d{1,2}$", num_raw)` when dots act as thousand separators (`1.234,56 €` $\rightarrow$ `1234.56`), while preserving Indian (`1,24,500.00`) and US (`124,500.00`) comma-thousand parsing.
- **Per-Page Error Isolation**:
  - Both `PDFDetector.inspect_pdf` and `StatementExtractor.extract_from_pdf` wrap individual page operations in `try/except Exception as page_exc:`, logging warnings via `logger.warning(...)` and preserving extracted transactions from all healthy pages.
