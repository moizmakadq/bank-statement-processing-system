"""Multi-Bank Account Metadata & Transaction Extractor (Phase 3 Enhanced).

Enhancements Implemented:
1. Date format ambiguity resolution:
   - Indian & UK/EU banks (HDFC, ICICI, SBI, Axis, Kotak, PNB, BoB, Barclays, HSBC, Deutsche) -> DD/MM/YYYY (`prefer_mdy=False`)
   - US banks (JPMorgan Chase, Citibank, Bank of America, Wells Fargo) -> MM/DD/YYYY (`prefer_mdy=True`)
   - Explicit `BANK_DATE_FORMAT_HINTS` per bank template + 20+ date format variations supported
2. Per-page error isolation:
   - Wraps page text/OCR and table/line parsing in `try/except`, logs warnings via `logging`, and continues processing remaining pages
3. Multi-currency detection & European comma-decimal amount parsing:
   - Detects `EUR` (`€`), `GBP` (`£`), `AED`, `SGD`, `JPY` (`¥`), `USD` (`$`), and `INR` (`₹` / `Rs.`)
   - Parses Indian (`1,24,500.00`), US/International (`124,500.00`), and European comma-as-decimal (`1.234,56` / `245,80 €`) amounts
"""

from __future__ import annotations
from datetime import datetime
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import pdfplumber
from src.ingestion.ocr_engine import OCREngine
from src.ingestion.pdf_detector import PDFDetector
from src.utils.models import AccountMetadata, PDFType, PageInspection, TransactionRecord

logger = logging.getLogger(__name__)


class StatementExtractor:
    """Extracts account metadata and raw transaction rows from text-based, scanned, or hybrid PDFs."""

    # Recognized bank signatures and regional templates
    BANK_SIGNATURES = [
        ("HDFC Bank", re.compile(r"\bHDFC\s+BANK\b|\bHDFC0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("ICICI Bank", re.compile(r"\bICICI\s+BANK\b|\bICIC0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("State Bank of India", re.compile(r"\bSTATE\s+BANK\s+OF\s+INDIA\b|\bSBI\b|\bSBIN0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("Axis Bank", re.compile(r"\bAXIS\s+BANK\b|\bUTIB0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("Kotak Mahindra Bank", re.compile(r"\bKOTAK\s+MAHINDRA\b|\bKKBK0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("Punjab National Bank", re.compile(r"\bPUNJAB\s+NATIONAL\s+BANK\b|\bPUNB0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("Bank of Baroda", re.compile(r"\bBANK\s+OF\s+BARODA\b|\bBARB0[A-Z0-9]{6}\b", re.IGNORECASE)),
        ("JPMorgan Chase", re.compile(r"\bJPMORGAN\s+CHASE\b|\bCHASE\s+BANK\b|\bCHASUS33\b", re.IGNORECASE)),
        ("Citibank", re.compile(r"\bCITIBANK\b|\bCITI\s+BANK\b|\bCITIUS33\b", re.IGNORECASE)),
        ("Bank of America", re.compile(r"\bBANK\s+OF\s+AMERICA\b|\bBOFAUS3N\b|\bBOFA\b", re.IGNORECASE)),
        ("Wells Fargo", re.compile(r"\bWELLS\s+FARGO\b|\bWFBIUS6S\b", re.IGNORECASE)),
        ("HSBC Bank", re.compile(r"\bHSBC\b|\bMIDLGB22\b", re.IGNORECASE)),
        ("Barclays Bank", re.compile(r"\bBARCLAYS\b|\bBARCGB22\b", re.IGNORECASE)),
        ("Deutsche Bank", re.compile(r"\bDEUTSCHE\s+BANK\b|\bDEUTDEFF\b", re.IGNORECASE)),
        ("DBS Bank", re.compile(r"\bDBS\s+BANK\b|\bDBSSSGSG\b", re.IGNORECASE)),
        ("Emirates NBD", re.compile(r"\bEMIRATES\s+NBD\b|\bEBILAEAD\b", re.IGNORECASE)),
    ]

    # Explicit date format hints per bank template (US banks use MM/DD/YYYY; Indian/UK/EU/SG/UAE use DD/MM/YYYY)
    BANK_DATE_FORMAT_HINTS: Dict[str, str] = {
        "HDFC Bank": "DD/MM/YYYY",
        "ICICI Bank": "DD/MM/YYYY",
        "State Bank of India": "DD/MM/YYYY",
        "Axis Bank": "DD/MM/YYYY",
        "Kotak Mahindra Bank": "DD/MM/YYYY",
        "Punjab National Bank": "DD/MM/YYYY",
        "Bank of Baroda": "DD/MM/YYYY",
        "JPMorgan Chase": "MM/DD/YYYY",
        "Citibank": "MM/DD/YYYY",
        "Bank of America": "MM/DD/YYYY",
        "Wells Fargo": "MM/DD/YYYY",
        "HSBC Bank": "DD/MM/YYYY",
        "Barclays Bank": "DD/MM/YYYY",
        "Deutsche Bank": "DD.MM.YYYY",
        "DBS Bank": "DD/MM/YYYY",
        "Emirates NBD": "DD/MM/YYYY",
    }

    US_BANKS = frozenset({"JPMorgan Chase", "Citibank", "Bank of America", "Wells Fargo"})

    # Comprehensive date token regex matching 20+ variations:
    # DD/MM/YYYY, MM/DD/YYYY, DD.MM.YYYY, YYYY-MM-DD, DD-MMM-YYYY, DD MMM, YYYY, MMM DD, YYYY, etc.
    DATE_TOKEN_PATTERN = (
        r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"
        r"|\d{4}[/.-]\d{1,2}[/.-]\d{1,2}"
        r"|\d{1,2}(?:st|nd|rd|th)?[\s/.-]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\s,.-]*\d{2,4}"
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})"
    )

    DATE_START_RE = re.compile(
        rf"^\s*(?:(?:S\.?\s*No\.?|TXN|TRX|CHQ|REF|[A-Za-z0-9#\-_]{{1,16}})[.)\s\t|/:-]+)?"
        rf"({DATE_TOKEN_PATTERN})\b",
        re.IGNORECASE
    )
    SECOND_DATE_RE = re.compile(rf"^\s*({DATE_TOKEN_PATTERN})\b", re.IGNORECASE)

    # Monetary amount token supporting Indian, US, and European (1.234,56) formats + multi-currency symbols
    # Accommodates OCR comma-to-dot misclassifications (e.g. 1.36,500.00, 72.000.00, 1.15.500.00) and space-padded digits
    AMOUNT_TOKEN_RE = re.compile(
        r"(?:(?P<sign>[-+])?\s*(?:INR|Rs\.?|USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|NZD|CHF|A\$|C\$|\$|₹|€|£|¥)?\s*"
        r"(?P<num>(?:\d{1,3}(?:[.,]{1,2}\d{2,3})+[.,]\d{2}|\d+[.,]\d{2}|\d{1,3}(?:,\d{3})+))"
        r"\s*(?:INR|Rs\.?|USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|NZD|CHF|A\$|C\$|\$|₹|€|£|¥)?"
        r"(?:\s*(?P<drcr>Cr\.?|Dr\.?|CR|DR))?"
        r"|\((?:INR|Rs\.?|USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|NZD|CHF|A\$|C\$|\$|₹|€|£|¥)?\s*"
        r"(?P<paren_num>(?:\d{1,3}(?:[.,]{1,2}\d{2,3})+[.,]\d{2}|\d+[.,]\d{2}|\d{1,3}(?:,\d{3})+))\))"
    )

    IGNORE_LINE_RE = re.compile(
        r"^\s*(?:Page\s+\d+|Statement\s+of\s+Account|Account\s+Statement|Date\s+Narration|Date\s+Description|"
        r"Txn\s+Date|Transaction\s+Date|Opening\s+Balance|Closing\s+Balance|Brought\s+Forward|Carried\s+Forward|"
        r"\*+End\s+of\s+Statement\*+|This\s+is\s+a\s+computer\s+generated|Summary\s+of\s+Account|Total\s*:)",
        re.IGNORECASE,
    )

    def __init__(self, ocr_dpi: int = 300):
        self.detector = PDFDetector()
        self.ocr_engine = OCREngine(dpi=ocr_dpi)

    @staticmethod
    def parse_amount(val: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
        """Parses a currency string (Indian, US, European comma-decimal, or OCR-scanned) into (float_value, explicit_dr_cr_indicator)."""
        if val is None:
            return None, None
        s = str(val).strip()
        if not s or s in ("-", "--", "NIL", "null", "None", "N/A"):
            return None, None

        # Clean spaces between digits and grouping separators (e.g. "1, 36, 500.00" -> "1,36,500.00")
        s_clean = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", s)
        # Fix common OCR confusions inside numeric tokens when surrounded by digits or separators
        s_clean = re.sub(r"(?<=\d)[Oo](?=\d|[.,])", "0", s_clean)
        s_clean = re.sub(r"(?<=[.,])[Oo](\d)", r"0\1", s_clean)
        s_clean = re.sub(r"(?<=[.,]\d)[Oo]\b", "0", s_clean)
        s_clean = re.sub(r"(?<=\d)[Oo]\b", "0", s_clean)

        drcr = None
        if re.search(r"\b(?:CR|Cr\.?)\b", s_clean):
            drcr = "CR"
        elif re.search(r"\b(?:DR|Dr\.?)\b", s_clean):
            drcr = "DR"

        is_negative = False
        if s_clean.startswith("(") and s_clean.endswith(")"):
            is_negative = True
            drcr = drcr or "DR"
        elif "-" in s_clean and re.search(r"-\s*(?:[A-Z]{3}|\$|₹|€|£|¥)?\s*\d", s_clean):
            is_negative = True
            drcr = drcr or "DR"
        elif "+" in s_clean and re.search(r"\+\s*(?:[A-Z]{3}|\$|₹|€|£|¥)?\s*\d", s_clean):
            drcr = drcr or "CR"

        # Isolate only digits, dots, and commas
        num_raw = re.sub(r"[^0-9.,]", "", s_clean).strip(".,")
        if not num_raw:
            return None, None

        # Detect European format (comma as decimal separator: e.g. "1.234,56" or "1234,56" or "1.234.567,89")
        if re.search(r",\d{1,2}$", num_raw) and ("." in num_raw or num_raw.count(",") == 1):
            idx = num_raw.rfind(",")
            int_pt = re.sub(r"[.,\s]", "", num_raw[:idx])
            dec_pt = num_raw[idx + 1 :]
            num_str = (int_pt or "0") + "." + dec_pt
        elif "." in num_raw:
            # Standard, Indian, or OCR format where '.' is the decimal separator (e.g. "1,36,500.00", "72.000.00", "1.15.500.00")
            # The rightmost '.' is the decimal point; preceding '.' and ',' are grouping/OCR artifacts
            idx = num_raw.rfind(".")
            int_pt = re.sub(r"[.,\s]", "", num_raw[:idx])
            dec_pt = num_raw[idx + 1 :]
            num_str = (int_pt or "0") + "." + dec_pt
        elif "," in num_raw and num_raw.count(",") > 1:
            # Grouped integer without decimals (e.g. "1,36,500" or "72,000")
            num_str = num_raw.replace(",", "")
        else:
            num_str = re.sub(r"[.,\s]", "", num_raw)

        if not num_str or num_str.count(".") > 1:
            return None, None

        try:
            amount = float(num_str)
            if is_negative:
                amount = -abs(amount)
            return round(amount, 2), drcr
        except ValueError:
            return None, None

    @staticmethod
    def normalize_date(raw_date: str, prefer_mdy: bool = False) -> Optional[str]:
        """Normalizes 20+ bank statement date formats to ISO YYYY-MM-DD with regional DD/MM vs MM/DD disambiguation."""
        if not raw_date:
            return None
        s = re.sub(r"\s+", " ", str(raw_date).strip())
        # Strip ordinal suffixes (1st, 2nd, 3rd, 4th)
        s = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", s, flags=re.IGNORECASE)
        s = s.replace(",", " ")
        s = re.sub(r"\s+", " ", s).strip()

        # Common OCR fix: 'O' instead of '0' adjacent to digits or date separators
        s = re.sub(r"(?<=\d)[Oo]\b", "0", s)
        s = re.sub(r"\b[Oo](?=\d)", "0", s)
        s = re.sub(r"(?<=[/.-])[Oo](?=\d)", "0", s)
        s = re.sub(r"(?<=\d)[Oo](?=[/.-])", "0", s)
        s = re.sub(r"\bSept\.?\b", "Sep", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([A-Za-z]{3})\.(?=\s+\d)", r"\1", s)

        dmy_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        ]
        mdy_formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
            "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",
        ]
        iso_and_named_formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
            "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y", "%d.%b.%Y",
            "%d %B %Y", "%d-%B-%Y", "%d/%B/%Y",
            "%d %b %y", "%d-%b-%y", "%d/%b/%y",
            "%d %B %y", "%d-%B-%y",
            "%b %d %Y", "%B %d %Y", "%b %d %y", "%B %d %y",
            "%b-%d-%Y", "%B-%d-%Y",
            "%Y %b %d", "%Y %B %d",
        ]

        if prefer_mdy:
            ordered_formats = mdy_formats + dmy_formats + iso_and_named_formats
        else:
            ordered_formats = dmy_formats + iso_and_named_formats + mdy_formats

        for fmt in ordered_formats:
            try:
                dt = datetime.strptime(s, fmt)
                if dt.year < 100:
                    dt = dt.replace(year=dt.year + 2000)
                if 1990 <= dt.year <= 2035:
                    return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def extract_metadata(self, full_text: str) -> AccountMetadata:
        """Extracts structured account holder, bank, currency, and date format hint details from statement text."""
        meta = AccountMetadata()

        # 1. Bank Name detection
        for bank_name, pattern in self.BANK_SIGNATURES:
            if pattern.search(full_text):
                meta.bank_name = bank_name
                break

        # Fallback to general bank name if not matched by fixed signatures
        if meta.bank_name == "Unknown Bank":
            bank_gen = re.search(r"\b([A-Z][A-Za-z0-9\s&]{2,30}\s+(?:Bank|Credit\s+Union))\b", full_text)
            if bank_gen:
                candidate_bank = re.sub(r"\s+", " ", bank_gen.group(1)).strip()
                if not re.search(r"\b(?:State|Central)\b", candidate_bank, re.IGNORECASE) or len(candidate_bank.split()) <= 4:
                    meta.bank_name = candidate_bank

        # 2. IFSC Code (Standard 11-char Indian IFSC: 4 letters + 0 + 6 alphanumeric)
        ifsc_m = re.search(
            r"(?:IFSC(?:\s*Code)?\s*[:\-]?\s*)?([A-Z]{4}0[A-Z0-9]{6})\b", full_text
        )
        if ifsc_m:
            meta.ifsc_code = ifsc_m.group(1).upper()
            if meta.bank_name == "Unknown Bank":
                prefix = meta.ifsc_code[:4]
                prefix_map = {
                    "HDFC": "HDFC Bank",
                    "ICIC": "ICICI Bank",
                    "SBIN": "State Bank of India",
                    "UTIB": "Axis Bank",
                    "KKBK": "Kotak Mahindra Bank",
                    "PUNB": "Punjab National Bank",
                    "BARB": "Bank of Baroda",
                }
                meta.bank_name = prefix_map.get(prefix, meta.bank_name)

        # Set date format hint based on detected bank/IFSC
        prefer_mdy = meta.bank_name in self.US_BANKS and not meta.ifsc_code
        meta.date_format_hint = self.BANK_DATE_FORMAT_HINTS.get(
            meta.bank_name, "MM/DD/YYYY" if prefer_mdy else "DD/MM/YYYY"
        )

        # 3. MICR Code
        micr_m = re.search(r"MICR(?:\s*Code)?\s*[:\-]?\s*(\d{9})\b", full_text, re.IGNORECASE)
        if micr_m:
            meta.micr_code = micr_m.group(1)

        # 4. Account Number & IBAN
        acct_m = re.search(
            r"(?:Account\s*(?:No\.?|Number|#)|Account\s*[:\-]|A/c\s*(?:No\.?|Number)?|Konto-?Nr\.?|Compte\s*N°?)\s*[:\-]?\s*([X*0-9][X*0-9\-\s]{3,22})\b",
            full_text,
            re.IGNORECASE,
        )
        if acct_m:
            meta.account_number = re.sub(r"\s+", " ", acct_m.group(1)).strip(" :-,.")
        else:
            iban_m = re.search(r"\b(?:IBAN\s*[:\-]?\s*)?([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b", full_text, re.IGNORECASE)
            if iban_m:
                meta.account_number = iban_m.group(1).strip()

        # 5. Account Holder Name
        holder_patterns = [
            r"(?:Account\s*Holder(?:\s*Name)?|Customer\s*Name|Name\s*of\s*Customer|Account\s*Name|Kontoinhaber|Titulaire(?:\s*du\s*compte)?)\s*[:\-]\s*([A-Z][A-Za-z\s.&]{2,50}?)(?:\s{2,}|\n|Account|A/c|Customer|IFSC|Branch|Address|Period|Date|IBAN|$)",
            r"(?:Mr\.|Ms\.|Mrs\.|M/s\.?)\s+([A-Z][A-Za-z\s.&]{3,45}?)(?:\s{2,}|\n|$)",
        ]
        for pat in holder_patterns:
            hm = re.search(pat, full_text, re.IGNORECASE)
            if hm:
                candidate = re.sub(r"\s+", " ", hm.group(1)).strip(" :-,.")
                if candidate and not re.search(r"^(?:Bank|Statement|Branch|Savings|Current)", candidate, re.IGNORECASE):
                    meta.account_holder_name = candidate
                    break

        # 6. Branch Name
        branch_m = re.search(
            r"(?:Branch(?:\s*Name)?|Home\s*Branch)\s*[:\-]\s*([A-Za-z0-9\s,\-]{3,45}?)(?:\s{2,}|\n|IFSC|MICR|Account|Currency|$)",
            full_text,
            re.IGNORECASE,
        )
        if branch_m:
            meta.branch_name = re.sub(r"\s+", " ", branch_m.group(1)).strip(" :-,.")

        # 7. Account Type
        type_m = re.search(
            r"(?:Account\s*Type|Scheme|A/c\s*Type)\s*[:\-]\s*([A-Za-z\s/\-]{3,35}?)(?:\s{2,}|\n|Currency|Branch|IFSC|$)",
            full_text,
            re.IGNORECASE,
        )
        if type_m:
            meta.account_type = re.sub(r"\s+", " ", type_m.group(1)).strip()

        # 8. Multi-Currency Detection (INR, USD, EUR, GBP, AED, SGD, JPY, AUD, CAD, NZD, CHF)
        curr_explicit = re.search(
            r"(?:Currency|Ccy)\s*[:\-]?\s*\b(INR|USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|NZD|CHF)\b",
            full_text,
            re.IGNORECASE,
        )
        if curr_explicit:
            meta.currency = curr_explicit.group(1).upper()
        elif re.search(r"\bAUD\b|A\$", full_text):
            meta.currency = "AUD"
        elif re.search(r"\bEUR\b|€", full_text):
            meta.currency = "EUR"
        elif re.search(r"\bGBP\b|£", full_text):
            meta.currency = "GBP"
        elif re.search(r"\bAED\b", full_text):
            meta.currency = "AED"
        elif re.search(r"\bSGD\b|S\$", full_text):
            meta.currency = "SGD"
        elif re.search(r"\bJPY\b|¥", full_text):
            meta.currency = "JPY"
        elif (re.search(r"\bUSD\b|\$", full_text) or prefer_mdy) and not re.search(r"\bINR\b|₹|\bIFSC\b", full_text):
            meta.currency = "USD"
        else:
            meta.currency = "INR"

        # 9. Statement Period
        period_m = re.search(
            rf"(?:Statement\s*Period|Period|From)\s*[:\-]?\s*({self.DATE_TOKEN_PATTERN})\s*(?:to|through|\-|–)\s*({self.DATE_TOKEN_PATTERN})",
            full_text,
            re.IGNORECASE,
        )
        if period_m:
            meta.statement_period_start = self.normalize_date(period_m.group(1), prefer_mdy=prefer_mdy) or period_m.group(1)
            meta.statement_period_end = self.normalize_date(period_m.group(2), prefer_mdy=prefer_mdy) or period_m.group(2)

        # 10. Opening & Closing Balance (Supporting Multi-currency & International Terms: German, French, Spanish)
        open_m = re.search(
            r"(?:Opening\s*Balance|Beginning\s*Balance|Balance\s*Brought\s*Forward|B/F|Alter\s*Kontostand|Solde\s*Initial|Saldo\s*Anterior|Saldo\s*Inicial)\s*[:\-]?\s*(?:[A-Z]{3}|\$|₹|€|£|¥)?\s*([0-9.,]+[.,]\d{2})",
            full_text,
            re.IGNORECASE,
        )
        if open_m:
            val, _ = self.parse_amount(open_m.group(1))
            meta.opening_balance = val

        close_m = re.search(
            r"(?:Closing\s*Balance|Ending\s*Balance|Balance\s*Carried\s*Forward|Neuer\s*Kontostand|Solde\s*Final|Saldo\s*Final)\s*[:\-]?\s*(?:[A-Z]{3}|\$|₹|€|£|¥)?\s*([0-9.,]+[.,]\d{2})",
            full_text,
            re.IGNORECASE,
        )
        if close_m:
            val, _ = self.parse_amount(close_m.group(1))
            meta.closing_balance = val

        # 11. Customer ID
        cust_m = re.search(
            r"(?:Customer\s*ID|Cust\s*ID|CIF\s*No\.?)\s*[:\-]?\s*([A-Z0-9]{5,16})\b",
            full_text,
            re.IGNORECASE,
        )
        if cust_m:
            meta.customer_id = cust_m.group(1)

        logger.info(
            "Extracted metadata: bank=%s, holder=%s, currency=%s, date_hint=%s",
            meta.bank_name, meta.account_holder_name, meta.currency, meta.date_format_hint
        )
        return meta

    def _parse_structured_tables(
        self,
        tables: List[List[List[Optional[str]]]],
        page_num: int,
        prefer_mdy: bool = False,
        initial_opening_balance: Optional[float] = None,
    ) -> List[TransactionRecord]:
        """Extracts transactions from pdfplumber tables when clear column headers exist."""
        records: List[TransactionRecord] = []
        running_bal_tracker: Optional[float] = initial_opening_balance

        for table in tables:
            if not table or len(table) < 2:
                continue

            header_idx = -1
            col_map: Dict[str, int] = {}

            def _map_cells(cell_list: List[str]) -> Dict[str, int]:
                m: Dict[str, int] = {}
                for c_idx, cell_text in enumerate(cell_list):
                    if not cell_text:
                        continue
                    if ("value" in cell_text and "date" in cell_text) or cell_text in ("val date", "value date"):
                        m["value_date"] = c_idx
                    elif any(k in cell_text for k in ("date", "datum", "buchung", "txn dt", "trans date", "posting date")) and "date" not in m:
                        m["date"] = c_idx
                    elif any(k in cell_text for k in ("description", "narration", "particulars", "remarks", "details", "transaction", "verwendungszweck", "libellé", "concepto", "memo")):
                        if "description" not in m:
                            m["description"] = c_idx
                    elif any(k in cell_text for k in ("chq", "cheque", "ref", "utr", "reference", "instrument", "trans id", "check")):
                        m["ref"] = c_idx
                    elif cell_text in ("type", "dr/cr", "cr/dr") or "cr/dr" in cell_text or "dr/cr" in cell_text:
                        m["type"] = c_idx
                    elif any(k in cell_text for k in ("debit", "withdrawal", "withdrawals", "paid out", "belastung", "débit")) or re.search(r"\bdr\b", cell_text):
                        m["debit"] = c_idx
                    elif any(k in cell_text for k in ("credit", "deposit", "deposits", "paid in", "gutschrift", "crédit")) or re.search(r"\bcr\b", cell_text):
                        m["credit"] = c_idx
                    elif any(k in cell_text for k in ("balance", "closing", "saldo", "solde")):
                        m["balance"] = c_idx
                    elif any(k in cell_text for k in ("amount", "betrag", "montant")) and "amount" not in m:
                        m["amount"] = c_idx
                return m

            for r_idx in range(min(10, len(table))):
                row = table[r_idx]
                cells = [re.sub(r"\s+", " ", (c or "")).strip().lower() for c in row]
                mapping_single = _map_cells(cells)

                if "date" in mapping_single and "description" in mapping_single:
                    has_amt_or_bal = any(k in mapping_single for k in ("debit", "credit", "amount", "balance"))
                    # If this single row is already a full header, use it directly
                    if has_amt_or_bal:
                        header_idx = r_idx
                        col_map = mapping_single
                        break

                    # Check if next row is a valid second-tier sub-header (not a data row)
                    if r_idx + 1 < len(table):
                        next_row = table[r_idx + 1]
                        first_cell = (next_row[0] or "").strip()
                        # If first cell is not a transaction date or number
                        if not re.search(r"\d", first_cell):
                            next_cells = [re.sub(r"\s+", " ", (c or "")).strip().lower() for c in next_row]
                            combined_cells = [f"{c} {next_cells[ci] if ci < len(next_cells) else ''}".strip() for ci, c in enumerate(cells)]
                            mapping_combined = _map_cells(combined_cells)
                            if any(k in mapping_combined for k in ("debit", "credit", "amount", "balance")):
                                header_idx = r_idx + 1
                                col_map = mapping_combined
                                break

                    # Otherwise, accept single-tier header
                    header_idx = r_idx
                    col_map = mapping_single
                    break

            if header_idx == -1 or "date" not in col_map or "description" not in col_map:
                continue

            for row in table[header_idx + 1 :]:
                if not row or all(not (c or "").strip() for c in row):
                    continue

                raw_date_cell = (row[col_map["date"]] or "").strip() if col_map["date"] < len(row) else ""
                raw_desc_cell = (row[col_map["description"]] or "").strip() if col_map["description"] < len(row) else ""
                raw_desc_cell = re.sub(r"\s+", " ", raw_desc_cell)

                norm_date = (
                    self.normalize_date(raw_date_cell.split("\n")[0].strip(), prefer_mdy=prefer_mdy)
                    if raw_date_cell
                    else None
                )
                if not norm_date:
                    if records and raw_desc_cell and not re.search(r"^(?:Opening|Closing|Total)\b", raw_desc_cell, re.IGNORECASE):
                        records[-1].description = f"{records[-1].description} {raw_desc_cell}".strip()
                    continue

                val_date = ""
                if "value_date" in col_map and col_map["value_date"] < len(row):
                    vd_raw = (row[col_map["value_date"]] or "").strip()
                    val_date = self.normalize_date(vd_raw, prefer_mdy=prefer_mdy) or vd_raw

                ref_no = ""
                if "ref" in col_map and col_map["ref"] < len(row):
                    ref_no = re.sub(r"\s+", " ", (row[col_map["ref"]] or "").strip())

                debit_val = 0.0
                credit_val = 0.0

                if "debit" in col_map and col_map["debit"] < len(row):
                    d_amt, _ = self.parse_amount(row[col_map["debit"]])
                    if d_amt is not None and d_amt != 0:
                        debit_val = abs(d_amt)

                if "credit" in col_map and col_map["credit"] < len(row):
                    c_amt, _ = self.parse_amount(row[col_map["credit"]])
                    if c_amt is not None and c_amt != 0:
                        credit_val = abs(c_amt)

                bal_val: Optional[float] = None
                if "balance" in col_map and col_map["balance"] < len(row):
                    b_amt, _ = self.parse_amount(row[col_map["balance"]])
                    bal_val = b_amt

                if debit_val == 0.0 and credit_val == 0.0 and "amount" in col_map and col_map["amount"] < len(row):
                    amt_val, drcr_flag = self.parse_amount(row[col_map["amount"]])
                    if "type" in col_map and col_map["type"] < len(row):
                        t_cell = (row[col_map["type"]] or "").strip().upper()
                        if "CR" in t_cell or "CREDIT" in t_cell:
                            drcr_flag = "CR"
                        elif "DR" in t_cell or "DEBIT" in t_cell:
                            drcr_flag = "DR"
                    if amt_val is not None:
                        if drcr_flag == "CR" or (amt_val > 0 and drcr_flag != "DR" and "+" in str(row[col_map["amount"]])):
                            credit_val = abs(amt_val)
                        elif drcr_flag == "DR" or amt_val < 0:
                            debit_val = abs(amt_val)
                        elif bal_val is not None and running_bal_tracker is not None:
                            delta = round(bal_val - running_bal_tracker, 2)
                            if abs(delta - abs(amt_val)) <= 1.0:
                                credit_val = abs(amt_val)
                            elif abs(delta + abs(amt_val)) <= 1.0:
                                debit_val = abs(amt_val)
                            elif re.search(r"\b(?:SALARY|CREDIT|REFUND|DIVIDEND|INTEREST|DEPOSIT|NEFT[-\s/]*CR|IMPS[-\s/]*CR)\b", raw_desc_cell, re.IGNORECASE):
                                credit_val = abs(amt_val)
                            else:
                                debit_val = abs(amt_val)
                        else:
                            if re.search(r"\b(?:SALARY|CREDIT|REFUND|DIVIDEND|INTEREST|DEPOSIT|NEFT[-\s/]*CR|IMPS[-\s/]*CR)\b", raw_desc_cell, re.IGNORECASE):
                                credit_val = abs(amt_val)
                            else:
                                debit_val = abs(amt_val)

                if bal_val is not None:
                    running_bal_tracker = bal_val

                if debit_val == 0.0 and credit_val == 0.0 and bal_val is None:
                    continue

                txn_type = "CREDIT" if credit_val > 0 and debit_val == 0 else "DEBIT"
                records.append(
                    TransactionRecord(
                        txn_id=f"TXN-{page_num:02d}-{len(records)+1:04d}",
                        row_index=len(records) + 1,
                        date=norm_date,
                        raw_date=raw_date_cell,
                        value_date=val_date or norm_date,
                        description=raw_desc_cell,
                        reference_no=ref_no,
                        debit=round(debit_val, 2),
                        credit=round(credit_val, 2),
                        balance=bal_val,
                        transaction_type=txn_type,
                        source_page=page_num,
                    )
                )

        return records

    def parse_lines_fallback(
        self,
        lines: List[str],
        page_num: int,
        initial_opening_balance: Optional[float] = None,
        prefer_mdy: bool = False,
    ) -> List[TransactionRecord]:
        """Parses raw text/OCR lines into TransactionRecords when structured table grids are absent."""
        records: List[TransactionRecord] = []
        running_prev_bal: Optional[float] = initial_opening_balance

        for raw_line in lines:
            line = raw_line.strip()
            if not line or self.IGNORE_LINE_RE.search(line):
                ob_m = re.search(
                    r"(?:Opening\s+Balance|Brought\s+Forward)\s*[:\-]?\s*(?:INR|Rs\.?|USD|EUR|GBP|AED|SGD|JPY|\$|₹|€|£|¥)?\s*([0-9.,]+[.,]\d{2})",
                    line,
                    re.IGNORECASE,
                )
                if ob_m and running_prev_bal is None:
                    running_prev_bal, _ = self.parse_amount(ob_m.group(1))
                continue

            date_m = self.DATE_START_RE.match(line)
            if date_m:
                raw_date_1 = date_m.group(1).strip()
                norm_date_1 = self.normalize_date(raw_date_1, prefer_mdy=prefer_mdy)
                if not norm_date_1:
                    continue

                remainder = line[date_m.end() :].strip()
                val_date = norm_date_1

                date2_m = self.SECOND_DATE_RE.match(remainder)
                if date2_m:
                    raw_date_2 = date2_m.group(1).strip()
                    norm_date_2 = self.normalize_date(raw_date_2, prefer_mdy=prefer_mdy)
                    if norm_date_2:
                        val_date = norm_date_2
                        remainder = remainder[date2_m.end() :].strip()

                amount_matches = list(self.AMOUNT_TOKEN_RE.finditer(remainder))
                if not amount_matches:
                    continue

                tail_matches = amount_matches[-3:] if len(amount_matches) >= 3 else amount_matches
                first_amt_start = tail_matches[0].start()
                desc_part = remainder[:first_amt_start].strip(" |-:")
                if not desc_part:
                    desc_part = "Transaction"

                # Check if desc_part starts with a second date (Value Date)
                val_date_m = re.match(rf"^\s*({self.DATE_TOKEN_PATTERN})\b\s*(.*)$", desc_part, re.IGNORECASE)
                if val_date_m:
                    vd_cand = self.normalize_date(val_date_m.group(1).strip(), prefer_mdy=prefer_mdy)
                    if vd_cand:
                        val_date = vd_cand
                        desc_part = val_date_m.group(2).strip()

                parsed_tail: List[Tuple[float, Optional[str], str]] = []
                for m in tail_matches:
                    raw_tok = m.group(0)
                    val, drcr = self.parse_amount(raw_tok)
                    if val is not None:
                        parsed_tail.append((val, drcr, raw_tok))

                if not parsed_tail:
                    continue

                debit_val = 0.0
                credit_val = 0.0
                bal_val: Optional[float] = None

                if len(parsed_tail) >= 3:
                    t1_amt, t1_drcr, _ = parsed_tail[-3]
                    t2_amt, t2_drcr, _ = parsed_tail[-2]
                    b_v, _, _ = parsed_tail[-1]
                    bal_val = b_v

                    if t1_amt > 0 and t2_amt == 0:
                        debit_val = abs(t1_amt)
                        credit_val = 0.0
                    elif t1_amt == 0 and t2_amt > 0:
                        credit_val = abs(t2_amt)
                        debit_val = 0.0
                    elif t1_drcr == "CR" or t2_drcr == "CR":
                        credit_val = abs(t2_amt if t2_drcr == "CR" else t1_amt)
                        debit_val = 0.0
                    elif t1_drcr == "DR" or t2_drcr == "DR":
                        debit_val = abs(t1_amt if t1_drcr == "DR" else t2_amt)
                        credit_val = 0.0
                    elif running_prev_bal is not None and bal_val is not None:
                        delta = round(bal_val - running_prev_bal, 2)
                        if delta > 0:
                            credit_val = abs(t2_amt if abs(delta - abs(t2_amt)) <= 1.0 else t1_amt)
                            debit_val = 0.0
                        elif delta < 0:
                            debit_val = abs(t1_amt if abs(delta + abs(t1_amt)) <= 1.0 else t2_amt)
                            credit_val = 0.0
                        else:
                            if re.search(r"\b(?:CR|CREDIT|SALARY|DEPOSIT|REFUND|DIVIDEND|INTEREST|INWARD)\b", desc_part, re.IGNORECASE):
                                credit_val = abs(t2_amt or t1_amt)
                                debit_val = 0.0
                            else:
                                debit_val = abs(t1_amt or t2_amt)
                                credit_val = 0.0
                    else:
                        if re.search(r"\b(?:CR|CREDIT|SALARY|DEPOSIT|REFUND|DIVIDEND|INTEREST|INWARD)\b", desc_part, re.IGNORECASE):
                            credit_val = abs(t2_amt or t1_amt)
                            debit_val = 0.0
                        else:
                            debit_val = abs(t1_amt or t2_amt)
                            credit_val = 0.0
                elif len(parsed_tail) == 2:
                    txn_amt, txn_drcr, _ = parsed_tail[0]
                    bal_val, _, _ = parsed_tail[1]

                    if txn_drcr == "CR" or re.search(r"\b(?:CR|CREDIT|SALARY|DEPOSIT|REFUND|DIVIDEND|INTEREST|INWARD|NEFT[-\s/]*CR|IMPS[-\s/]*CR)\b", desc_part, re.IGNORECASE):
                        credit_val = abs(txn_amt)
                        debit_val = 0.0
                    elif txn_drcr == "DR" or txn_amt < 0 or re.search(r"\b(?:DR|DEBIT|PURCHASE|UPI|PAYMENT|WITHDRAWAL|ATM|POS|CHARGES|TAX)\b", desc_part, re.IGNORECASE):
                        debit_val = abs(txn_amt)
                        credit_val = 0.0
                    elif running_prev_bal is not None and bal_val is not None:
                        delta = round(bal_val - running_prev_bal, 2)
                        if delta > 0:
                            credit_val = abs(txn_amt)
                            debit_val = 0.0
                        else:
                            debit_val = abs(txn_amt)
                            credit_val = 0.0
                    else:
                        debit_val = abs(txn_amt)
                        credit_val = 0.0
                elif len(parsed_tail) == 1:
                    txn_amt, txn_drcr, _ = parsed_tail[0]
                    if txn_drcr == "CR":
                        credit_val = abs(txn_amt)
                    else:
                        debit_val = abs(txn_amt)

                if bal_val is not None:
                    running_prev_bal = bal_val

                ref_no = ""
                ref_m = re.search(r"\b(?:UTR|Ref|Chq|Cheque|RRN|MMT|IMPS|NEFT|UPI)[/\-:\s]*([A-Z0-9]{6,22})\b", desc_part, re.IGNORECASE)
                if ref_m:
                    ref_no = ref_m.group(1)

                txn_type = "CREDIT" if credit_val > 0 and debit_val == 0 else "DEBIT"
                records.append(
                    TransactionRecord(
                        txn_id=f"TXN-{page_num:02d}-{len(records)+1:04d}",
                        row_index=len(records) + 1,
                        date=norm_date_1,
                        raw_date=raw_date_1,
                        value_date=val_date,
                        description=re.sub(r"\s+", " ", desc_part).strip(),
                        reference_no=ref_no,
                        debit=round(debit_val, 2),
                        credit=round(credit_val, 2),
                        balance=bal_val,
                        transaction_type=txn_type,
                        source_page=page_num,
                    )
                )
            else:
                if (
                    records
                    and len(line) > 2
                    and not self.AMOUNT_TOKEN_RE.search(line)
                    and not re.search(r"^(?:Total|Closing|Summary|Page|Generated|Branch|IFSC|Customer)", line, re.IGNORECASE)
                ):
                    cleaned_cont = re.sub(r"\s+", " ", line)
                    records[-1].description = f"{records[-1].description} {cleaned_cont}".strip()

        return records

    def extract_from_pdf(
        self, pdf_path: str, password: str = ""
    ) -> Tuple[PDFType, AccountMetadata, List[TransactionRecord], List[PageInspection], List[str]]:
        """Full extraction pipeline for a PDF or image statement with per-page error isolation."""
        pdf_type, inspections, warnings = self.detector.inspect_pdf(pdf_path, password=password)

        full_text_chunks: List[str] = []
        all_transactions: List[TransactionRecord] = []

        is_image_file = os.path.splitext(pdf_path)[1].lower() in (
            ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"
        )

        if is_image_file:
            p_info = inspections[0]
            try:
                ocr_res = self.ocr_engine.ocr_page(pdf_path, 1)
                p_info.ocr_confidence = ocr_res.confidence
                full_text_chunks.append(ocr_res.text)
                if abs(ocr_res.skew_angle_deg) > 0.1:
                    warnings.append(f"Image deskewed by {ocr_res.skew_angle_deg}° (OCR confidence: {ocr_res.confidence:.1f}%).")
            except Exception as exc:
                err_msg = f"Image OCR extraction failed ({exc}); returning empty results."
                logger.warning(err_msg)
                warnings.append(err_msg)
                p_info.error = str(exc)
                full_text_chunks.append("")

            combined_text = "\n".join(full_text_chunks)
            metadata = self.extract_metadata(combined_text)
            prefer_mdy = metadata.date_format_hint.startswith("MM/")
            ocr_lines = full_text_chunks[0].splitlines() if full_text_chunks else []
            all_transactions = self.parse_lines_fallback(
                ocr_lines, page_num=1, initial_opening_balance=metadata.opening_balance, prefer_mdy=prefer_mdy
            )
            for i, txn in enumerate(all_transactions, start=1):
                txn.row_index = i
                txn.txn_id = f"TXN-{i:04d}"
            logger.info("Extracted %d total transactions from image %s", len(all_transactions), pdf_path)
            return pdf_type, metadata, all_transactions, inspections, warnings

        # Single-pass extraction: cache normal text, layout text & tables per page with error isolation
        cached_pages: List[Tuple[str, str, List[List[List[Optional[str]]]]]] = []
        with pdfplumber.open(pdf_path, password=password or None) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                p_info = inspections[idx - 1]
                try:
                    if p_info.page_type == PDFType.TEXT_BASED:
                        page_text_layout = page.extract_text(layout=True) or ""
                        page_text_norm = page.extract_text() or ""
                        tables = page.extract_tables() or []
                        full_text_chunks.append(page_text_layout or page_text_norm)
                        cached_pages.append((page_text_norm, page_text_layout, tables))
                    else:
                        ocr_res = self.ocr_engine.ocr_page(pdf_path, idx)
                        p_info.ocr_confidence = ocr_res.confidence
                        full_text_chunks.append(ocr_res.text)
                        cached_pages.append((ocr_res.text, ocr_res.text, []))
                        if abs(ocr_res.skew_angle_deg) > 0.1:
                            msg = f"Page {idx}: Deskewed scanned image by {ocr_res.skew_angle_deg}° (OCR confidence: {ocr_res.confidence:.1f}%)."
                            warnings.append(msg)
                            logger.info(msg)
                except Exception as page_exc:
                    err_msg = f"Page {idx} text/OCR extraction failed ({page_exc}); skipping page and continuing."
                    logger.warning(err_msg)
                    warnings.append(err_msg)
                    p_info.error = str(page_exc)
                    full_text_chunks.append("")
                    cached_pages.append(("", "", []))

        combined_text = "\n".join(full_text_chunks)
        metadata = self.extract_metadata(combined_text)
        prefer_mdy = metadata.date_format_hint.startswith("MM/")

        # Parse transactions from memory-cached pages (zero redundant PDF file I/O)
        running_bal = metadata.opening_balance
        for idx, (p_text_norm, p_text_layout, tables) in enumerate(cached_pages, start=1):
            p_info = inspections[idx - 1]
            page_txns: List[TransactionRecord] = []
            try:
                if p_info.page_type == PDFType.TEXT_BASED and not p_info.error:
                    candidates: List[List[TransactionRecord]] = []
                    if tables:
                        table_txns = self._parse_structured_tables(
                            tables, page_num=idx, prefer_mdy=prefer_mdy, initial_opening_balance=running_bal
                        )
                        if table_txns:
                            candidates.append(table_txns)
                    if p_text_layout:
                        layout_txns = self.parse_lines_fallback(
                            p_text_layout.splitlines(), page_num=idx, initial_opening_balance=running_bal, prefer_mdy=prefer_mdy
                        )
                        if layout_txns:
                            candidates.append(layout_txns)
                    if p_text_norm:
                        norm_txns = self.parse_lines_fallback(
                            p_text_norm.splitlines(), page_num=idx, initial_opening_balance=running_bal, prefer_mdy=prefer_mdy
                        )
                        if norm_txns:
                            candidates.append(norm_txns)

                    if candidates:
                        page_txns = max(
                            candidates,
                            key=lambda cand: (sum(1 for t in cand if (t.debit > 0 or t.credit > 0)), len(cand))
                        )
                else:
                    ocr_lines = (p_text_norm or p_text_layout).splitlines()
                    page_txns = self.parse_lines_fallback(
                        ocr_lines, page_num=idx, initial_opening_balance=running_bal, prefer_mdy=prefer_mdy
                    )
            except Exception as parse_exc:
                warn_msg = f"Page {idx} transaction parsing failed ({parse_exc}); returning partial results from remaining pages."
                logger.warning(warn_msg)
                warnings.append(warn_msg)
                p_info.error = str(parse_exc)

            if page_txns:
                if page_txns[-1].balance is not None:
                    running_bal = page_txns[-1].balance
                all_transactions.extend(page_txns)

        for i, txn in enumerate(all_transactions, start=1):
            txn.row_index = i
            txn.txn_id = f"TXN-{i:04d}"

        logger.info("Extracted %d total transactions from %s", len(all_transactions), pdf_path)
        return pdf_type, metadata, all_transactions, inspections, warnings
