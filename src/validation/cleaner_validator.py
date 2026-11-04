"""Financial Data Cleaning, Normalization, Channel Extraction & Balance Reconciliation Module.

Performs:
1. Narration normalization & counterparty extraction
2. Domestic & International payment channel detection (UPI, NEFT, IMPS, RTGS, WIRE, SEPA,
   DIRECT_DEBIT, ZELLE, VENMO, CASHAPP, POS_CARD, ATM, ACH_NACH, CHEQUE, BANK_FEE, INTEREST)
3. Mathematical running-balance reconciliation (Opening Balance + Credit - Debit == Closing Balance)
4. Automatic correction of swapped Debit/Credit columns when running balance proves the direction
5. Detection of duplicate transactions, reversals/refunds, and high-value transactions
"""

from __future__ import annotations
import logging
import re
from typing import List, Optional, Tuple
from src.utils.models import (
    AccountMetadata,
    ReconciliationSummary,
    TransactionRecord,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


class CleanerAndValidator:
    """Cleans extracted transaction descriptions and reconciles running balances."""

    HIGH_VALUE_THRESHOLD = 50000.0

    PAYMENT_CHANNEL_PATTERNS = [
        ("ZELLE", re.compile(r"\bZELLE\b", re.IGNORECASE)),
        ("VENMO", re.compile(r"\bVENMO\b", re.IGNORECASE)),
        ("CASHAPP", re.compile(r"\b(?:CASH\s*APP|CASHAPP|SQ\s*\*)\b", re.IGNORECASE)),
        ("SEPA", re.compile(r"\bSEPA\b", re.IGNORECASE)),
        ("WIRE", re.compile(r"\b(?:WIRE\s*TRANSFER|INTL\s*WIRE|FEDWIRE|SWIFT)\b", re.IGNORECASE)),
        ("DIRECT_DEBIT", re.compile(r"\b(?:DIRECT\s*DEBIT|BACS|FASTER\s*PAYMENTS)\b", re.IGNORECASE)),
        ("UPI", re.compile(r"\bUPI\b|@[a-z]{3,}", re.IGNORECASE)),
        ("NEFT", re.compile(r"\bNEFT\b", re.IGNORECASE)),
        ("RTGS", re.compile(r"\bRTGS\b", re.IGNORECASE)),
        ("IMPS", re.compile(r"\bIMPS\b|\bMMT\b", re.IGNORECASE)),
        ("ACH_NACH", re.compile(r"\b(?:ACH|NACH|ECS|MANDATE|SI|AUTOPAY)\b", re.IGNORECASE)),
        ("ATM", re.compile(r"\b(?:ATM|CASH\s*WDL|ATW|NFS)\b", re.IGNORECASE)),
        ("POS_CARD", re.compile(r"\b(?:POS|ECOM|DEBIT\s*CARD|VISA|MASTERCARD|RUPAY|AMEX)\b", re.IGNORECASE)),
        ("CHEQUE", re.compile(r"\b(?:CHQ|CHEQUE|CLG|CLEARING)\b", re.IGNORECASE)),
        ("INTEREST", re.compile(r"\b(?:INT\.?\s*PD|INTEREST\s*PAID|INTEREST\s*CREDIT|SB\s*INT)\b", re.IGNORECASE)),
        ("BANK_FEE", re.compile(r"\b(?:SMS\s*CHG|MIN\s*BAL\s*CHG|AMC|ANNUAL\s*FEE|PENALTY|CHARGES|OVERDRAFT\s*FEE)\b", re.IGNORECASE)),
    ]

    REVERSAL_RE = re.compile(r"\b(?:REV|REVERSAL|REFUND|CHARGEBACK|RET|RETURNED|BOUNCE)\b", re.IGNORECASE)

    def __init__(self, high_value_threshold: float = 50000.0, tolerance: float = 0.05):
        self.high_value_threshold = high_value_threshold
        self.tolerance = tolerance

    def detect_channel_and_counterparty(self, description: str) -> Tuple[str, str, str]:
        """Returns (payment_channel, counterparty, cleaned_description_for_ml)."""
        raw = re.sub(r"\s+", " ", description or "").strip()
        channel = "OTHER"
        for ch_name, pat in self.PAYMENT_CHANNEL_PATTERNS:
            if pat.search(raw):
                channel = ch_name
                break

        counterparty = ""
        upi_m = re.search(r"UPI[/-](?:CR[/-]|DR[/-])?(?:\d+[/-])?([A-Za-z0-9\s.&'_-]{2,35}?)[/-]", raw, re.IGNORECASE)
        if upi_m:
            candidate = upi_m.group(1).strip()
            if not candidate.isdigit():
                counterparty = candidate

        if not counterparty:
            intl_p2p = re.search(
                r"(?:ZELLE|VENMO|CASH\s*APP|SEPA|WIRE\s*TRANSFER|DIRECT\s*DEBIT)\s*(?:PAYMENT|TRANSFER)?\s*(?:TO|FROM|-)?\s*([A-Za-z][A-Za-z0-9\s.&_-]{2,35})",
                raw,
                re.IGNORECASE,
            )
            if intl_p2p:
                counterparty = intl_p2p.group(1).strip()

        if not counterparty:
            neft_m = re.search(
                r"(?:NEFT|IMPS|RTGS|ACH|NACH)[/-](?:CR[/-]|DR[/-])?(?:[A-Z0-9]{6,18}[/-])?([A-Za-z][A-Za-z0-9\s.&_-]{2,40}?)(?:[/-]|$)",
                raw,
                re.IGNORECASE,
            )
            if neft_m:
                counterparty = neft_m.group(1).strip()

        if not counterparty:
            pos_m = re.search(r"(?:POS|ECOM)[/\s\-]+(?:\d+[/\s\-]+)?([A-Za-z][A-Za-z0-9\s.&'_-]{2,35})", raw, re.IGNORECASE)
            if pos_m:
                counterparty = pos_m.group(1).strip()

        clean = raw.upper()
        clean = re.sub(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", " ", clean)
        clean = re.sub(r"\b\d{6,}\b", " ", clean)
        clean = re.sub(r"[^A-Z0-9@&\s]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        return channel, counterparty.upper(), clean

    def clean_and_reconcile(
        self, metadata: AccountMetadata, transactions: List[TransactionRecord]
    ) -> Tuple[List[TransactionRecord], ReconciliationSummary]:
        """Cleans descriptions, validates running balances, and builds reconciliation audit report."""
        summary = ReconciliationSummary(
            total_transactions=len(transactions),
            reported_opening_balance=metadata.opening_balance,
            reported_closing_balance=metadata.closing_balance,
        )

        if not transactions:
            logger.warning("clean_and_reconcile called with 0 transactions.")
            return transactions, summary

        # Detect if transactions are ordered in descending chronological order (newest-first)
        is_descending = False
        if len(transactions) >= 2:
            d_start = transactions[0].date
            d_end = transactions[-1].date
            if d_start and d_end and d_start > d_end:
                is_descending = True
                logger.info("Detected descending statement order (%s to %s); reconciling chronologically.", d_start, d_end)

        txns_ordered = list(reversed(transactions)) if is_descending else transactions
        seen_fingerprints: dict[tuple, int] = {}
        prev_balance: Optional[float] = metadata.opening_balance

        first_txn = txns_ordered[0]
        if prev_balance is None and first_txn.balance is not None:
            prev_balance = round(first_txn.balance - first_txn.credit + first_txn.debit, 2)
            metadata.opening_balance = prev_balance
            logger.debug("Inferred opening balance %.2f from first transaction.", prev_balance)

        summary.computed_opening_balance = prev_balance

        for txn in txns_ordered:
            channel, counterparty, clean_desc = self.detect_channel_and_counterparty(txn.description)
            txn.payment_channel = channel
            txn.counterparty = counterparty
            txn.clean_description = clean_desc

            if self.REVERSAL_RE.search(txn.description):
                txn.is_reversal = True
                summary.reversal_rows += 1
                txn.validation_notes.append("Reversal/Refund transaction detected.")

            if prev_balance is not None and txn.balance is not None:
                expected = round(prev_balance + txn.credit - txn.debit, 2)
                diff = round(txn.balance - expected, 2)

                swapped_expected = round(prev_balance + txn.debit - txn.credit, 2)
                if abs(diff) > self.tolerance and abs(txn.balance - swapped_expected) <= self.tolerance:
                    txn.debit, txn.credit = txn.credit, txn.debit
                    txn.transaction_type = "CREDIT" if txn.credit > 0 else "DEBIT"
                    expected = swapped_expected
                    diff = round(txn.balance - expected, 2)
                    note = "Auto-corrected Debit/Credit column polarity using running balance."
                    txn.validation_notes.append(note)
                    logger.info("Row %d: %s", txn.row_index, note)

                txn.expected_balance = expected
                txn.balance_diff = diff

                if abs(diff) <= self.tolerance:
                    txn.validation_status = ValidationStatus.VALID.value
                    summary.reconciled_rows += 1
                else:
                    txn.validation_status = ValidationStatus.MISMATCH.value
                    summary.mismatch_rows += 1
                    msg = f"Row {txn.row_index} ({txn.date}): Expected balance {expected:,.2f}, found {txn.balance:,.2f} (diff: {diff:+,.2f})."
                    txn.validation_notes.append(msg)
                    summary.anomalies.append(msg)
                    logger.warning(msg)

                prev_balance = txn.balance
            elif prev_balance is not None and txn.balance is None:
                txn.balance = round(prev_balance + txn.credit - txn.debit, 2)
                txn.expected_balance = txn.balance
                txn.balance_diff = 0.0
                txn.validation_notes.append("Imputed missing balance cell from running total.")
                summary.reconciled_rows += 1
                prev_balance = txn.balance
            else:
                summary.reconciled_rows += 1

            amt_val = txn.credit if txn.credit > 0 else -txn.debit
            fp = (txn.date, round(amt_val, 2), txn.clean_description[:40], txn.reference_no)
            if fp in seen_fingerprints:
                txn.is_duplicate = True
                summary.duplicate_rows += 1
                prev_row = seen_fingerprints[fp]
                note = f"Potential duplicate of Row {prev_row} on {txn.date} for {abs(amt_val):,.2f}."
                txn.validation_notes.append(note)
                logger.warning("Row %d: %s", txn.row_index, note)
                if txn.validation_status == ValidationStatus.VALID.value:
                    txn.validation_status = ValidationStatus.WARNING.value
            else:
                seen_fingerprints[fp] = txn.row_index

            if max(txn.debit, txn.credit) >= self.high_value_threshold:
                txn.is_high_value = True
                summary.high_value_rows += 1
                txn.validation_notes.append(f"High-value transaction (>= {self.high_value_threshold:,.0f}).")

            if txn.debit > 0:
                summary.debit_count += 1
                summary.total_debit = round(summary.total_debit + txn.debit, 2)
            if txn.credit > 0:
                summary.credit_count += 1
                summary.total_credit = round(summary.total_credit + txn.credit, 2)

        summary.net_cashflow = round(summary.total_credit - summary.total_debit, 2)
        summary.computed_closing_balance = prev_balance

        if metadata.closing_balance is None and prev_balance is not None:
            metadata.closing_balance = prev_balance

        if (
            metadata.closing_balance is not None
            and summary.computed_closing_balance is not None
            and abs(metadata.closing_balance - summary.computed_closing_balance) > self.tolerance
        ):
            msg = (
                f"Header closing balance ({metadata.closing_balance:,.2f}) differs from final transaction balance "
                f"({summary.computed_closing_balance:,.2f})."
            )
            summary.anomalies.append(msg)
            logger.warning(msg)

        summary.is_fully_reconciled = summary.mismatch_rows == 0
        summary.reconciliation_rate = round(
            (summary.reconciled_rows / summary.total_transactions) * 100.0, 2
        )

        logger.info(
            "Reconciliation complete: %d/%d rows reconciled (%.2f%%)",
            summary.reconciled_rows,
            summary.total_transactions,
            summary.reconciliation_rate,
        )
        return transactions, summary
