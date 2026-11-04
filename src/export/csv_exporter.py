"""CSV Export Module for Processed & Classified Bank Statements."""

from __future__ import annotations
import csv
import os
from typing import List
from src.utils.models import StatementProcessingResult, TransactionRecord


class CSVExporter:
    """Exports classified bank statement transactions and optional account summary to CSV."""

    TRANSACTION_HEADERS = [
        "Transaction_ID",
        "Row_No",
        "Date",
        "Value_Date",
        "Description",
        "Clean_Description",
        "Reference_No",
        "Payment_Channel",
        "Counterparty",
        "Transaction_Type",
        "Debit",
        "Credit",
        "Balance",
        "Expected_Balance",
        "Category",
        "Sub_Category",
        "Classification_Method",
        "Confidence_Score",
        "Rule_Or_Feature_Matched",
        "Validation_Status",
        "Is_Duplicate",
        "Is_Reversal",
        "Is_High_Value",
        "Validation_Notes",
    ]

    @classmethod
    def export_transactions_csv(cls, result: StatementProcessingResult, output_path: str) -> str:
        """Writes clean, tabular transaction records with assigned categories to a standard UTF-8 CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cls.TRANSACTION_HEADERS)
            for t in result.transactions:
                writer.writerow(
                    [
                        t.txn_id,
                        t.row_index,
                        t.date,
                        t.value_date,
                        t.description,
                        t.clean_description,
                        t.reference_no,
                        t.payment_channel,
                        t.counterparty,
                        t.transaction_type,
                        f"{t.debit:.2f}",
                        f"{t.credit:.2f}",
                        f"{t.balance:.2f}" if t.balance is not None else "",
                        f"{t.expected_balance:.2f}" if t.expected_balance is not None else "",
                        t.category,
                        t.sub_category,
                        t.classification_method,
                        f"{t.confidence:.3f}",
                        t.rule_matched,
                        t.validation_status,
                        "YES" if t.is_duplicate else "NO",
                        "YES" if t.is_reversal else "NO",
                        "YES" if t.is_high_value else "NO",
                        "; ".join(t.validation_notes),
                    ]
                )
        return output_path
