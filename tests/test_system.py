"""Comprehensive Unit & Integration Test Suite for Bank Statement Processing & Classification System (Phases 1-5).

Covers:
1. PDF Auto-Detection (Text-based vs Scanned Image-based)
2. Date Normalization across 22+ format variations + Regional Ambiguity (Indian DD/MM/YYYY vs US MM/DD/YYYY)
3. Amount Parsing across Indian (1,24,500.00), International (124,500.00), and European comma-decimal (1.234,56 €) formats
4. Multi-Currency Header Detection (INR, USD, EUR, GBP, AED, SGD, JPY)
5. All 19 Categories in HeuristicRuleClassifier (Positive & Negative Polarity Cases + Dividend vs Salary Priority)
6. International Payment Rails (Wire Transfer, ACH, Direct Debit, SEPA, Zelle, Venmo, CashApp) & Merchants (Target, Costco, CVS, Walgreens, Tesco, Sainsbury)
7. Dual-Head Traditional ML Classifier (2,300+ External CSV Corpus, Sub-Category Prediction, 80/20 Split, 5-Fold CV, Per-Class Metrics & Confusion Matrix)
8. Balance Reconciliation with Deliberate Mismatch, Polarity Swap Auto-Correction & Per-Page Error Isolation
9. End-to-End Text & Scanned OCR Statement Processing and CSV/Excel Export Integrity
"""

from __future__ import annotations
import os
import tempfile
import unittest
from unittest.mock import patch

from src.classification.hybrid_engine import HybridClassificationEngine
from src.classification.ml_classifier import TraditionalMLClassifier
from src.classification.rule_classifier import HeuristicRuleClassifier
from src.classification.transformer_classifier import TransformerMLClassifier
from src.export.csv_exporter import CSVExporter
from src.export.excel_exporter import ExcelExporter
from src.extraction.statement_extractor import StatementExtractor
from src.ingestion.pdf_detector import PDFDetector
from src.pipeline import BankStatementPipeline
from src.utils.generate_training_corpus import build_training_corpus
from src.utils.models import AccountMetadata, PDFType, SUPPORTED_CATEGORIES, TransactionRecord, ValidationStatus
from src.utils.sample_pdf_generator import generate_all_sample_pdfs
from src.validation.cleaner_validator import CleanerAndValidator


class TestBankStatementSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="bank_stmt_test_v2_")
        cls.corpus_csv = os.path.join(cls.tmp_dir, "training_corpus.csv")
        build_training_corpus(cls.corpus_csv)
        cls.pdf_paths = generate_all_sample_pdfs(cls.tmp_dir)
        cls.pipeline = BankStatementPipeline(
            model_path=os.path.join(cls.tmp_dir, "model.pkl"),
            corpus_csv_path=cls.corpus_csv,
            ocr_dpi=300,
        )

    def test_01_pdf_auto_detection_text_vs_image(self):
        """Verifies PDFDetector accurately distinguishes TEXT_BASED from IMAGE_BASED PDFs."""
        detector = PDFDetector()
        hdfc_type, hdfc_pages, _ = detector.inspect_pdf(self.pdf_paths["hdfc_text"])
        self.assertEqual(hdfc_type, PDFType.TEXT_BASED)
        self.assertGreater(hdfc_pages[0].char_count, 100)

        sbi_type, sbi_pages, warnings = detector.inspect_pdf(self.pdf_paths["sbi_scanned"])
        self.assertEqual(sbi_type, PDFType.IMAGE_BASED)
        self.assertEqual(sbi_pages[0].char_count, 0)
        self.assertTrue(any("OCR" in w for w in warnings))

    def test_02_date_normalization_22_formats_and_regional_ambiguity(self):
        """Tests 22+ date format variations and DD/MM/YYYY (Indian/UK) vs MM/DD/YYYY (US) disambiguation."""
        ext = StatementExtractor()
        # 20+ format variations
        date_cases = [
            ("05/10/2026", False, "2026-10-05"),
            ("05-10-2026", False, "2026-10-05"),
            ("05.10.2026", False, "2026-10-05"),
            ("05/10/26", False, "2026-10-05"),
            ("05-10-26", False, "2026-10-05"),
            ("05.10.26", False, "2026-10-05"),
            ("2026-10-05", False, "2026-10-05"),
            ("2026/10/05", False, "2026-10-05"),
            ("2026.10.05", False, "2026-10-05"),
            ("05 Oct 2026", False, "2026-10-05"),
            ("5-Oct-2026", False, "2026-10-05"),
            ("05/Oct/2026", False, "2026-10-05"),
            ("05 October 2026", False, "2026-10-05"),
            ("05-October-2026", False, "2026-10-05"),
            ("5th Oct 2026", False, "2026-10-05"),
            ("Oct 05, 2026", False, "2026-10-05"),
            ("October 5, 2026", False, "2026-10-05"),
            ("Oct. 5, 2026", False, "2026-10-05"),
            ("05 Sept 2026", False, "2026-09-05"),
            ("O5/1O/2026", False, "2026-10-05"),  # OCR O->0
            # Regional ambiguity: "04/07/2026" -> July 4 in India (DD/MM/YYYY), April 7 in US (MM/DD/YYYY)
            ("04/07/2026", False, "2026-07-04"),
            ("04/07/2026", True, "2026-04-07"),
            ("10-12-2026", False, "2026-12-10"),
            ("10-12-2026", True, "2026-10-12"),
        ]
        for raw, prefer_mdy, expected in date_cases:
            with self.subTest(raw=raw, prefer_mdy=prefer_mdy):
                self.assertEqual(ext.normalize_date(raw, prefer_mdy=prefer_mdy), expected)

        # Verify US vs Indian bank metadata date_format_hint
        chase_meta = ext.extract_metadata("JPMORGAN CHASE BANK NA\nAccount Number: 449102938\nPeriod: 04/01/2026 to 04/30/2026")
        self.assertEqual(chase_meta.bank_name, "JPMorgan Chase")
        self.assertEqual(chase_meta.date_format_hint, "MM/DD/YYYY")
        self.assertEqual(chase_meta.statement_period_start, "2026-04-01")

        hdfc_meta = ext.extract_metadata("HDFC BANK LTD\nIFSC Code: HDFC0000240\nPeriod: 01/04/2026 to 30/04/2026")
        self.assertEqual(hdfc_meta.bank_name, "HDFC Bank")
        self.assertEqual(hdfc_meta.date_format_hint, "DD/MM/YYYY")
        self.assertEqual(hdfc_meta.statement_period_start, "2026-04-01")

    def test_03_amount_parsing_indian_international_european_and_multicurrency(self):
        """Tests Indian (1,24,500.00), International (124,500.00), European (1.234,56), and multi-currency detection."""
        ext = StatementExtractor()
        # Indian format
        self.assertEqual(ext.parse_amount("₹ 1,24,500.50 Cr"), (124500.50, "CR"))
        # US/International format
        self.assertEqual(ext.parse_amount("$124,500.75 DR"), (124500.75, "DR"))
        self.assertEqual(ext.parse_amount("(3,400.00)"), (-3400.00, "DR"))
        # European comma-as-decimal format
        self.assertEqual(ext.parse_amount("1.234,56 €"), (1234.56, None))
        self.assertEqual(ext.parse_amount("-450,80 EUR"), (-450.80, "DR"))
        self.assertEqual(ext.parse_amount("12.345.678,90"), (12345678.90, None))
        # OCR O->0
        self.assertEqual(ext.parse_amount("1,5O0.OO"), (1500.00, None))

        # Multi-currency detection in headers
        for header_text, expected_ccy in [
            ("DEUTSCHE BANK AG\nBalance: 2.450,00 €\nCurrency: EUR", "EUR"),
            ("BARCLAYS BANK UK\nOpening Balance: £4,200.00", "GBP"),
            ("EMIRATES NBD DUBAI\nCurrency: AED\nOpening Balance: AED 15,000.00", "AED"),
            ("DBS BANK SINGAPORE\nCurrency: SGD\nBalance: S$ 8,500.00", "SGD"),
            ("BANK STATEMENT TOKYO\nCurrency: JPY\nBalance: ¥150,000.00", "JPY"),
            ("CITIBANK NA USA\nOpening Balance: $5,400.00", "USD"),
            ("STATE BANK OF INDIA\nIFSC: SBIN0004120\nBalance: ₹64,500.00", "INR"),
        ]:
            meta = ext.extract_metadata(header_text)
            self.assertEqual(meta.currency, expected_ccy)

    def test_04_rule_classifier_all_19_categories_and_dividend_priority(self):
        """Tests positive & negative polarity cases across all 19 categories plus dividend priority over salary."""
        clf = HeuristicRuleClassifier()
        self.assertEqual(len(clf.CATEGORIES), 19)

        def make_txn(desc: str, debit: float = 500.0, credit: float = 0.0) -> TransactionRecord:
            return TransactionRecord(
                txn_id="T1", row_index=1, date="2026-10-01", raw_date="01/10/2026",
                description=desc, clean_description=desc.upper(),
                debit=debit, credit=credit,
                transaction_type="CREDIT" if credit > 0 else "DEBIT",
            )

        # 1. Verify Dividend Priority Fix: Even if text mentions company/salary-like tokens, Equity/Final/Interim Dividend matches Interest & Dividends
        div_txn = make_txn("ACH CR INFOSYS LIMITED FINAL DIVIDEND WARRANT", debit=0.0, credit=4500.0)
        div_res = clf.classify(div_txn)
        self.assertIsNotNone(div_res)
        self.assertEqual(div_res.category, "Salary & Income")
        self.assertEqual(div_res.sub_category, "Interest & Dividends")
        self.assertEqual(div_res.rule_id, "RULE_INTEREST_DIVIDEND")

        eq_div_txn = make_txn("NEFT CR RELIANCE INDUSTRIES EQUITY DIVIDEND PAYROLL ACCT", debit=0.0, credit=3250.0)
        eq_res = clf.classify(eq_div_txn)
        self.assertEqual(eq_res.sub_category, "Interest & Dividends")

        # 2. Verify Polarity Negative Case: A DEBIT containing "SALARY" must NOT trigger RULE_SALARY_PAYROLL
        sal_debit_txn = make_txn("UPI DR SALARY ADVANCE REPAYMENT TO OFFICE", debit=5000.0, credit=0.0)
        sal_debit_res = clf.classify(sal_debit_txn)
        self.assertIsNone(sal_debit_res)

        # 3. Verify all 19 categories match positive cases
        positive_cases = [
            ("NEFT CR TCS LTD MONTHLY SALARY PAYROLL", 0.0, 85000.0, "Salary & Income", "Payroll / Salary"),
            ("UPI DR SWIGGY FOOD ORDER DINING", 450.0, 0.0, "Food & Dining", "Restaurants & Food Delivery"),
            ("POS DEBIT COSTCO WHOLESALE GROCERY", 3200.0, 0.0, "Groceries & Supermarkets", "Supermarket & Quick Commerce"),
            ("POS DEBIT TESCO EXTRA SUPERMARKET UK", 85.0, 0.0, "Groceries & Supermarkets", "Supermarket & Quick Commerce"),
            ("POS DEBIT SAINSBURYS SUPERMARKET LONDON", 64.0, 0.0, "Groceries & Supermarkets", "Supermarket & Quick Commerce"),
            ("POS DEBIT TARGET STORE T-2184 RETAIL", 120.0, 0.0, "Shopping & E-Commerce", "Online & Retail Shopping"),
            ("BBPS DR BESCOM ELECTRICITY BILL", 2100.0, 0.0, "Utilities & Bills", "Electricity, Gas & Water"),
            ("NEFT DR LANDLORD HOUSE RENT OCTOBER", 25000.0, 0.0, "Rent & Housing", "House Rent & Maintenance"),
            ("UPI DR IRCTC RAIL TICKET BOOKING", 1450.0, 0.0, "Travel & Transport", "Flights, Rail & Hotels"),
            ("POS DEBIT CVS PHARMACY #0482 HEALTH", 45.0, 0.0, "Healthcare & Medical", "Hospitals, Labs & Pharmacy"),
            ("POS DEBIT WALGREENS DRUG STORE", 38.0, 0.0, "Healthcare & Medical", "Hospitals, Labs & Pharmacy"),
            ("SI DR NETFLIX ENTERTAINMENT SUBSCRIPTION", 649.0, 0.0, "Entertainment & Subscriptions", "OTT, Gaming & Movies"),
            ("NACH DR ZERODHA COIN MUTUAL FUND SIP", 15000.0, 0.0, "Investments & Insurance", "Mutual Funds & Equities"),
            ("NACH DR HDFC HOME LOAN EMI INSTALLMENT", 32500.0, 0.0, "Loans, EMI & Credit Cards", "Loan EMI"),
            ("ATM WDL NFS CASH WITHDRAWAL", 10000.0, 0.0, "ATM & Cash Withdrawal", "ATM Cash Withdrawal"),
            ("ZELLE PAYMENT TO MICHAEL SMITH SPLIT BILL", 150.0, 0.0, "Transfers & Peer-to-Peer", "Bank / UPI Transfer"),
            ("VENMO PAYMENT SARAH JONES DINNER", 45.0, 0.0, "Transfers & Peer-to-Peer", "Bank / UPI Transfer"),
            ("SMS ALERT CHARGES FOR QTR ENDED OCT", 17.7, 0.0, "Bank Fees, Taxes & Charges", "Bank Charges & Taxes"),
            # 5 New Phase 1 Categories
            ("UPI DR BYJUS THINK AND LEARN EDTECH", 12000.0, 0.0, "Education & Learning", "EdTech"),
            ("POS VISA COURSERA ONLINE COURSE", 3999.0, 0.0, "Education & Learning", "Online Courses"),
            ("NEFT DR NATIONAL UNIVERSITY TUITION FEE", 65000.0, 0.0, "Education & Learning", "School/College Fees"),
            ("NETBANKING TIN NSDL INCOME TAX CHALLAN", 18500.0, 0.0, "Government & Taxes", "Income Tax"),
            ("BBPS DR BBMP PROPERTY TAX BENGALURU", 6400.0, 0.0, "Government & Taxes", "Property Tax"),
            ("UPI DR PASSPORT SEVA APPLICATION FEE", 1500.0, 0.0, "Government & Taxes", "Government Services"),
            ("UPI DR TIRUMALA TIRUPATI TEMPLE DONATION", 1100.0, 0.0, "Charity & Donations", "Religious"),
            ("UPI DR KETTO CROWDFUNDING CAMPAIGN", 2000.0, 0.0, "Charity & Donations", "Crowdfunding"),
            ("NEFT DR GIVEINDIA NGO CHARITY DONATION", 5000.0, 0.0, "Charity & Donations", "NGO"),
            ("UPI DR CULT FIT GYM MEMBERSHIP", 1499.0, 0.0, "Personal Care & Wellness", "Gym"),
            ("UPI DR URBAN COMPANY SALON GROOMING", 899.0, 0.0, "Personal Care & Wellness", "Salon"),
            ("UPI DR PAYTM WALLET LOAD TOPUP", 2000.0, 0.0, "Wallet & Recharge", "Digital Wallet Top-ups"),
        ]
        for desc, dr, cr, exp_cat, exp_sub in positive_cases:
            with self.subTest(desc=desc):
                res = clf.classify(make_txn(desc, dr, cr))
                self.assertIsNotNone(res, f"Rule failed to match: {desc}")
                self.assertEqual(res.category, exp_cat)
                self.assertEqual(res.sub_category, exp_sub)

    def test_05_ml_classifier_2000_plus_corpus_subcategory_and_cv_metrics(self):
        """Tests external CSV loading (2,300+ rows), dual-head sub-category prediction, 80/20 split, and 5-fold CV."""
        ml_clf = self.pipeline.classifier.ml_classifier
        corpus = ml_clf.load_corpus_from_csv(self.corpus_csv)
        self.assertGreaterEqual(len(corpus), 2000)
        self.assertEqual(len(ml_clf.classes), 19)
        self.assertGreaterEqual(len(ml_clf.sub_classes), 30)

        # Test dual-head sub-category predictions on unseen narrations (where rule engine is bypassed)
        pred_edu = ml_clf.predict("POS VISA ONLINE LEARNING CERTIFICATION COURSE", "DEBIT", 2400.0)
        self.assertEqual(pred_edu.category, "Education & Learning")
        self.assertIn(pred_edu.sub_category, ["Online Courses", "EdTech", "School/College Fees"])

        pred_tax = ml_clf.predict("NETBANKING MUNICIPAL AUTHORITY PROPERTY ASSESSMENT", "DEBIT", 8200.0)
        self.assertEqual(pred_tax.category, "Government & Taxes")
        self.assertEqual(pred_tax.sub_category, "Property Tax")

        # Verify 80/20 Stratified Split metrics & Confusion Matrix
        split_res = ml_clf.evaluate_train_test_split(test_ratio=0.20)
        self.assertGreaterEqual(split_res["accuracy_pct"], 92.0)
        self.assertGreaterEqual(split_res["macro_f1_pct"], 90.0)
        self.assertGreaterEqual(split_res["sub_category_accuracy_pct"], 88.0)
        self.assertEqual(len(split_res["confusion_matrix"]), 19)
        self.assertEqual(len(split_res["per_class"]), 19)

        # Verify 5-Fold Stratified Cross-Validation
        cv_res = ml_clf.evaluate_cross_validation(n_folds=5)
        self.assertEqual(cv_res["n_folds"], 5)
        self.assertGreaterEqual(cv_res["mean_accuracy_pct"], 92.0)
        self.assertGreaterEqual(cv_res["mean_macro_f1_pct"], 90.0)

    def test_06_balance_reconciliation_deliberate_mismatch_and_polarity_swap(self):
        """Tests balance reconciliation with deliberately mismatched data and swapped Debit/Credit polarity."""
        validator = CleanerAndValidator()
        # Case A: Auto-fix swapped Debit/Credit polarity
        meta_a = AccountMetadata(opening_balance=10000.0, closing_balance=15000.0)
        txns_a = [
            TransactionRecord(
                txn_id="TXN-0001", row_index=1, date="2026-10-01", raw_date="01/10/2026",
                description="IMPS INWARD TRANSFER", debit=5000.0, credit=0.0, balance=15000.0,
            )
        ]
        cleaned_a, rec_a = validator.clean_and_reconcile(meta_a, txns_a)
        self.assertEqual(cleaned_a[0].credit, 5000.0)
        self.assertEqual(cleaned_a[0].debit, 0.0)
        self.assertTrue(rec_a.is_fully_reconciled)

        # Case B: Deliberately mismatched running balance that cannot be explained by polarity swap
        meta_b = AccountMetadata(opening_balance=20000.0, closing_balance=15000.0)
        txns_b = [
            TransactionRecord(
                txn_id="TXN-0001", row_index=1, date="2026-10-01", raw_date="01/10/2026",
                description="UPI DR GROCERY STORE", debit=2000.0, credit=0.0, balance=18000.0,
            ),
            TransactionRecord(
                txn_id="TXN-0002", row_index=2, date="2026-10-02", raw_date="02/10/2026",
                description="UPI DR ELECTRONICS STORE", debit=1500.0, credit=0.0, balance=14000.0,  # Should be 16500.0!
            ),
        ]
        cleaned_b, rec_b = validator.clean_and_reconcile(meta_b, txns_b)
        self.assertFalse(rec_b.is_fully_reconciled)
        self.assertEqual(rec_b.mismatch_rows, 1)
        self.assertEqual(cleaned_b[1].validation_status, ValidationStatus.MISMATCH.value)
        self.assertAlmostEqual(cleaned_b[1].balance_diff, -2500.0, places=2)
        self.assertGreaterEqual(len(rec_b.anomalies), 1)

    def test_07_per_page_error_isolation(self):
        """Verifies that an exception on one page logs a warning and still returns partial results."""
        ext = StatementExtractor()
        with patch.object(ext, "_parse_structured_tables", side_effect=RuntimeError("Simulated corrupted table structure")):
            pdf_type, meta, txns, pages, warnings = ext.extract_from_pdf(self.pdf_paths["hdfc_text"])
            self.assertEqual(pdf_type, PDFType.TEXT_BASED)
            self.assertTrue(any("Simulated corrupted table structure" in w for w in warnings))

    def test_08_end_to_end_statements_and_exports(self):
        """Verifies HDFC text, ICICI single-column, and SBI scanned OCR PDFs plus CSV/Excel exports."""
        res_hdfc = self.pipeline.process_pdf(self.pdf_paths["hdfc_text"])
        self.assertEqual(len(res_hdfc.transactions), 20)
        self.assertTrue(res_hdfc.reconciliation.is_fully_reconciled)

        res_icici = self.pipeline.process_pdf(self.pdf_paths["icici_text"])
        self.assertEqual(len(res_icici.transactions), 14)
        self.assertTrue(res_icici.reconciliation.is_fully_reconciled)
        self.assertGreaterEqual(res_icici.reconciliation.duplicate_rows, 1)

        res_sbi = self.pipeline.process_pdf(self.pdf_paths["sbi_scanned"])
        self.assertEqual(res_sbi.pdf_type, PDFType.IMAGE_BASED)
        if res_sbi.average_ocr_confidence is not None:
            self.assertEqual(len(res_sbi.transactions), 12)
            self.assertTrue(res_sbi.reconciliation.is_fully_reconciled)
        else:
            # Graceful degradation when host system lacks Tesseract OCR binary
            self.assertTrue(any("OCR" in w for w in res_sbi.warnings))

        csv_path = os.path.join(self.tmp_dir, "test_out.csv")
        xlsx_path = os.path.join(self.tmp_dir, "test_out.xlsx")
        CSVExporter.export_transactions_csv(res_hdfc, csv_path)
        ExcelExporter.export_workbook([res_hdfc, res_icici, res_sbi], xlsx_path)
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(xlsx_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
