"""Core data models for the Bank Statement Processing & Classification System."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class PDFType(str, Enum):
    TEXT_BASED = "TEXT_BASED"
    IMAGE_BASED = "IMAGE_BASED"
    HYBRID = "HYBRID"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    MISMATCH = "MISMATCH"
    DUPLICATE = "DUPLICATE"


SUPPORTED_CATEGORIES: List[str] = [
    "Salary & Income",
    "Food & Dining",
    "Groceries & Supermarkets",
    "Shopping & E-Commerce",
    "Utilities & Bills",
    "Rent & Housing",
    "Travel & Transport",
    "Healthcare & Medical",
    "Entertainment & Subscriptions",
    "Investments & Insurance",
    "Loans, EMI & Credit Cards",
    "ATM & Cash Withdrawal",
    "Transfers & Peer-to-Peer",
    "Bank Fees, Taxes & Charges",
    "Education & Learning",
    "Government & Taxes",
    "Charity & Donations",
    "Personal Care & Wellness",
    "Wallet & Recharge",
]


@dataclass
class AccountMetadata:
    bank_name: str = "Unknown Bank"
    account_holder_name: str = ""
    account_number: str = ""
    ifsc_code: str = ""
    micr_code: str = ""
    branch_name: str = ""
    account_type: str = "Savings / Current"
    currency: str = "INR"
    date_format_hint: str = "DD/MM/YYYY"
    statement_period_start: str = ""
    statement_period_end: str = ""
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    customer_id: str = ""
    email: str = ""
    phone: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageInspection:
    page_number: int
    char_count: int
    word_count: int
    image_count: int
    has_tables: bool
    page_type: PDFType
    ocr_confidence: Optional[float] = None
    error: Optional[str] = None


@dataclass
class TransactionRecord:
    txn_id: str
    row_index: int
    date: str  # Normalized ISO YYYY-MM-DD
    raw_date: str
    value_date: str = ""
    description: str = ""
    clean_description: str = ""
    reference_no: str = ""
    debit: float = 0.0
    credit: float = 0.0
    balance: Optional[float] = None
    expected_balance: Optional[float] = None
    balance_diff: float = 0.0
    transaction_type: str = "DEBIT"  # DEBIT or CREDIT
    payment_channel: str = "OTHER"   # UPI, NEFT, IMPS, RTGS, POS_CARD, ATM, ACH_NACH, WIRE, SEPA, DIRECT_DEBIT, ZELLE, VENMO, CASHAPP, CHEQUE, BANK_FEE, INTEREST
    counterparty: str = ""
    # Classification fields (Strictly Non-LLM)
    category: str = "Uncategorized"
    sub_category: str = "General"
    classification_method: str = "Unclassified"  # RULE_ENGINE, ML_MODEL, HYBRID_ENSEMBLE, TRANSFORMER_MODEL
    confidence: float = 0.0
    rule_matched: str = ""
    ml_probabilities: Dict[str, float] = field(default_factory=dict)
    # Validation & Edge Case Flags
    validation_status: str = ValidationStatus.VALID.value
    validation_notes: List[str] = field(default_factory=list)
    is_duplicate: bool = False
    is_reversal: bool = False
    is_high_value: bool = False
    source_page: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationSummary:
    total_transactions: int = 0
    debit_count: int = 0
    credit_count: int = 0
    total_debit: float = 0.0
    total_credit: float = 0.0
    net_cashflow: float = 0.0
    computed_opening_balance: Optional[float] = None
    computed_closing_balance: Optional[float] = None
    reported_opening_balance: Optional[float] = None
    reported_closing_balance: Optional[float] = None
    reconciled_rows: int = 0
    mismatch_rows: int = 0
    duplicate_rows: int = 0
    reversal_rows: int = 0
    high_value_rows: int = 0
    is_fully_reconciled: bool = True
    reconciliation_rate: float = 100.0
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatementProcessingResult:
    file_name: str
    file_size_bytes: int
    pdf_type: PDFType
    total_pages: int
    processing_time_ms: float
    ocr_applied: bool
    average_ocr_confidence: Optional[float]
    metadata: AccountMetadata
    transactions: List[TransactionRecord]
    reconciliation: ReconciliationSummary
    category_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    monthly_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "pdf_type": self.pdf_type.value if isinstance(self.pdf_type, PDFType) else self.pdf_type,
            "total_pages": self.total_pages,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "ocr_applied": self.ocr_applied,
            "average_ocr_confidence": self.average_ocr_confidence,
            "metadata": self.metadata.to_dict(),
            "transactions": [t.to_dict() for t in self.transactions],
            "reconciliation": self.reconciliation.to_dict(),
            "category_summary": self.category_summary,
            "monthly_summary": self.monthly_summary,
            "warnings": self.warnings,
        }
