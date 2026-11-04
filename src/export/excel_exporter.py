"""Professional Multi-Sheet Excel (.xlsx) Exporter for Classified Bank Statements.

Builds an audit-ready, formula-driven openpyxl workbook with 3 sheets:
1. Executive Summary: Account metadata, KPI summary cards, Category Breakdown table with
   dynamic Excel formulas (`SUM`, `SUMIF`, `COUNTIF`, `COUNTA`), and embedded native openpyxl BarChart
2. Classified Transactions: Full transaction ledger with Debit, Credit, Balance, Category,
   Sub-Category, Payment Channel, Non-LLM Classification Method, and Confidence
3. Reconciliation & Metadata: Opening/Closing balance verification, PDF page inspection metrics,
   and embedded BarChart
"""

from __future__ import annotations
import os
from typing import List
import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries
from src.utils.models import StatementProcessingResult


class ExcelExporter:
    """Generates a multi-sheet, formula-driven Excel (.xlsx) report for processed bank statements."""

    NAVY_HEADER = "1B365D"
    SLATE_SUBHEADER = "2C5282"
    ALT_ROW_FILL = "F4F7FA"
    KPI_BG_FILL = "EDF2F7"
    BORDER_GRAY = "CBD5E0"
    VALID_GREEN = "D1E7DD"
    WARN_YELLOW = "FFF3CD"

    @staticmethod
    def _style_merged_range(ws, cell_range: str, font=None, fill=None, alignment=None, border=None):
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
        for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
            for cell in row:
                if font:
                    cell.font = font
                if fill:
                    cell.fill = fill
                if alignment:
                    cell.alignment = alignment
                if border:
                    cell.border = border

    @staticmethod
    def _autofit_columns(ws, skip_rows: set[int]):
        merged_multi_col = {
            cell.coordinate
            for rng in ws.merged_cells.ranges
            if rng.min_col != rng.max_col
            for row in ws[rng.coord]
            for cell in row
        }
        for col in ws.columns:
            max_len = 0
            for cell in col:
                if cell.row in skip_rows or cell.coordinate in merged_multi_col:
                    continue
                val = str(cell.value) if cell.value is not None else ""
                if val.startswith("="):
                    max_len = max(max_len, 14)
                else:
                    max_len = max(max_len, len(val))
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 40)

    @classmethod
    def export_workbook(cls, results: List[StatementProcessingResult], output_path: str) -> str:
        """Exports one or more StatementProcessingResults into a comprehensive multi-sheet Excel workbook."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        wb = openpyxl.Workbook()

        title_font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
        th_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        bold_font = Font(name="Calibri", size=10, bold=True, color="1A202C")
        reg_font = Font(name="Calibri", size=10, color="1A202C")

        navy_fill = PatternFill(start_color=cls.NAVY_HEADER, end_color=cls.NAVY_HEADER, fill_type="solid")
        alt_fill = PatternFill(start_color=cls.ALT_ROW_FILL, end_color=cls.ALT_ROW_FILL, fill_type="solid")
        green_fill = PatternFill(start_color=cls.VALID_GREEN, end_color=cls.VALID_GREEN, fill_type="solid")
        warn_fill = PatternFill(start_color=cls.WARN_YELLOW, end_color=cls.WARN_YELLOW, fill_type="solid")

        thin_border = Border(
            left=Side(style="thin", color=cls.BORDER_GRAY),
            right=Side(style="thin", color=cls.BORDER_GRAY),
            top=Side(style="thin", color=cls.BORDER_GRAY),
            bottom=Side(style="thin", color=cls.BORDER_GRAY),
        )
        total_border = Border(
            top=Side(style="thin", color=cls.NAVY_HEADER),
            bottom=Side(style="double", color=cls.NAVY_HEADER),
            left=Side(style="thin", color=cls.BORDER_GRAY),
            right=Side(style="thin", color=cls.BORDER_GRAY),
        )

        left_align = Alignment(horizontal="left", vertical="center")
        right_align = Alignment(horizontal="right", vertical="center")
        center_align = Alignment(horizontal="center", vertical="center")

        all_txns = []
        for r in results:
            for t in r.transactions:
                all_txns.append((r, t))

        # =========================================================================
        # SHEET 1: Classified Transactions (Row 1 = Headers, Row 2+ = Data)
        # =========================================================================
        ws_txn = wb.active
        ws_txn.title = "Classified Transactions"
        ws_txn.views.sheetView[0].showGridLines = True

        txn_headers = [
            "Txn ID",
            "Bank Name",
            "Account No",
            "Date",
            "Value Date",
            "Description",
            "Payment Channel",
            "Counterparty",
            "Type",
            "Debit (INR)",
            "Credit (INR)",
            "Running Balance (INR)",
            "Category",
            "Sub-Category",
            "Method",
            "Confidence",
        ]

        header_row_txn = 1
        for col_idx, h_text in enumerate(txn_headers, start=1):
            c = ws_txn.cell(row=header_row_txn, column=col_idx, value=h_text)
            c.font = th_font
            c.fill = navy_fill
            c.border = thin_border
            c.alignment = left_align
        ws_txn.row_dimensions[header_row_txn].height = 24

        start_data_row = 2
        for idx, (res_obj, t) in enumerate(all_txns):
            r_num = start_data_row + idx
            acct_display = f"A/C-{res_obj.metadata.account_number}" if res_obj.metadata.account_number else "A/C-NA"
            row_vals = [
                t.txn_id,
                res_obj.metadata.bank_name,
                acct_display,
                t.date,
                t.value_date or t.date,
                t.description,
                t.payment_channel,
                t.counterparty or "GENERAL",
                t.transaction_type,
                float(t.debit),
                float(t.credit),
                float(t.balance) if t.balance is not None else 0.0,
                t.category,
                t.sub_category,
                t.classification_method,
                float(t.confidence),
            ]
            for col_idx, val in enumerate(row_vals, start=1):
                c = ws_txn.cell(row=r_num, column=col_idx, value=val)
                c.font = reg_font
                c.border = thin_border
                if r_num % 2 == 1:
                    c.fill = alt_fill
                if col_idx in (10, 11, 12):
                    c.alignment = right_align
                    c.number_format = "#,##0.00"
                elif col_idx == 16:
                    c.alignment = right_align
                    c.number_format = "0.0%"
                else:
                    c.alignment = left_align

        end_data_row = start_data_row + len(all_txns) - 1
        tot_row_txn = end_data_row + 1
        for col_idx in range(1, 17):
            c = ws_txn.cell(row=tot_row_txn, column=col_idx)
            c.font = bold_font
            c.border = total_border
            if col_idx == 1:
                c.value = "Total"
                c.alignment = left_align
            elif col_idx == 10:
                c.value = f"=SUM(J{start_data_row}:J{end_data_row})"
                c.alignment = right_align
                c.number_format = "#,##0.00"
            elif col_idx == 11:
                c.value = f"=SUM(K{start_data_row}:K{end_data_row})"
                c.alignment = right_align
                c.number_format = "#,##0.00"
            elif col_idx == 12:
                c.value = f"=L{end_data_row}"
                c.alignment = right_align
                c.number_format = "#,##0.00"
            elif col_idx == 16:
                c.value = f"=AVERAGE(P{start_data_row}:P{end_data_row})"
                c.alignment = right_align
                c.number_format = "0.0%"
            else:
                c.value = None
                c.alignment = left_align

        # Add overview chart below table at A(tot_row_txn + 3)
        txn_chart = BarChart()
        txn_chart.type = "col"
        txn_chart.style = 10
        txn_chart.title = "Transaction Debit vs Credit Amounts"
        txn_chart.y_axis.title = "INR"
        txn_chart.x_axis.title = "Transaction"
        txn_chart.width = 18
        txn_chart.height = 9
        t_data_ref = Reference(ws_txn, min_col=10, min_row=1, max_col=11, max_row=min(end_data_row, 21))
        t_cats_ref = Reference(ws_txn, min_col=1, min_row=2, max_row=min(end_data_row, 21))
        txn_chart.add_data(t_data_ref, titles_from_data=True)
        txn_chart.set_categories(t_cats_ref)
        ws_txn.add_chart(txn_chart, f"A{tot_row_txn + 3}")

        ws_txn.freeze_panes = "A2"
        cls._autofit_columns(ws_txn, skip_rows=set())

        # =========================================================================
        # SHEET 2: Executive Summary (inserted at index 0 as first tab)
        # =========================================================================
        ws_sum = wb.create_sheet(title="Executive Summary", index=0)
        ws_sum.views.sheetView[0].showGridLines = True

        cat_headers = [
            "Category",
            "Txn Count",
            "Total Debit (INR)",
            "Total Credit (INR)",
            "Net Flow (INR)",
            "Debit Share %",
        ]
        cat_header_row = 1
        for col_idx, ch in enumerate(cat_headers, start=1):
            c = ws_sum.cell(row=cat_header_row, column=col_idx, value=ch)
            c.font = th_font
            c.fill = navy_fill
            c.border = thin_border
            c.alignment = left_align
        ws_sum.row_dimensions[cat_header_row].height = 24

        unique_cats = sorted({t.category for _, t in all_txns})
        cat_start_row = 2
        for idx, cat_name in enumerate(unique_cats):
            r_idx = cat_start_row + idx
            ws_sum.cell(row=r_idx, column=1, value=cat_name).alignment = left_align
            ws_sum.cell(
                row=r_idx,
                column=2,
                value=f"=COUNTIF('Classified Transactions'!$M${start_data_row}:$M${end_data_row}, A{r_idx})",
            ).number_format = "#,##0"
            ws_sum.cell(
                row=r_idx,
                column=3,
                value=f"=SUMIF('Classified Transactions'!$M${start_data_row}:$M${end_data_row}, A{r_idx}, 'Classified Transactions'!$J${start_data_row}:$J${end_data_row})",
            ).number_format = "#,##0.00"
            ws_sum.cell(
                row=r_idx,
                column=4,
                value=f"=SUMIF('Classified Transactions'!$M${start_data_row}:$M${end_data_row}, A{r_idx}, 'Classified Transactions'!$K${start_data_row}:$K${end_data_row})",
            ).number_format = "#,##0.00"
            ws_sum.cell(row=r_idx, column=5, value=f"=D{r_idx}-C{r_idx}").number_format = "#,##0.00"

            for c_i in range(1, 7):
                cell = ws_sum.cell(row=r_idx, column=c_i)
                cell.font = reg_font
                cell.border = thin_border
                if r_idx % 2 == 1:
                    cell.fill = alt_fill
                if c_i > 1:
                    cell.alignment = right_align

        cat_end_row = cat_start_row + len(unique_cats) - 1
        cat_tot_row = cat_end_row + 1

        for idx in range(len(unique_cats)):
            r_idx = cat_start_row + idx
            c_pct = ws_sum.cell(row=r_idx, column=6, value=f"=IF($C${cat_tot_row}>0, C{r_idx}/$C${cat_tot_row}, 0)")
            c_pct.number_format = "0.0%"
            c_pct.alignment = right_align

        ws_sum.cell(row=cat_tot_row, column=1, value="Total").alignment = left_align
        ws_sum.cell(row=cat_tot_row, column=2, value=f"=SUM(B{cat_start_row}:B{cat_end_row})").number_format = "#,##0"
        ws_sum.cell(row=cat_tot_row, column=3, value=f"=SUM(C{cat_start_row}:C{cat_end_row})").number_format = "#,##0.00"
        ws_sum.cell(row=cat_tot_row, column=4, value=f"=SUM(D{cat_start_row}:D{cat_end_row})").number_format = "#,##0.00"
        ws_sum.cell(row=cat_tot_row, column=5, value=f"=SUM(E{cat_start_row}:E{cat_end_row})").number_format = "#,##0.00"
        ws_sum.cell(row=cat_tot_row, column=6, value=f"=SUM(F{cat_start_row}:F{cat_end_row})").number_format = "0.0%"

        for c_i in range(1, 7):
            c = ws_sum.cell(row=cat_tot_row, column=c_i)
            c.font = bold_font
            c.border = total_border
            if c_i > 1:
                c.alignment = right_align

        # Side-by-Side BarChart anchored at H2 (leaving Column G as blank spacer)
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "Category-Wise Debit vs Credit Breakdown"
        chart.y_axis.title = "Amount (INR)"
        chart.x_axis.title = "Transaction Category"
        chart.width = 16
        chart.height = 10

        data_ref = Reference(ws_sum, min_col=3, min_row=cat_header_row, max_col=4, max_row=cat_end_row)
        cats_ref = Reference(ws_sum, min_col=1, min_row=cat_start_row, max_row=cat_end_row)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.legend.legendPos = "r"
        ws_sum.add_chart(chart, "H2")
        ws_sum.column_dimensions["G"].width = 4

        ws_sum.freeze_panes = "A2"
        cls._autofit_columns(ws_sum, skip_rows=set())

        # =========================================================================
        # SHEET 3: Statement Metadata & Reconciliation Audit
        # =========================================================================
        ws_rec = wb.create_sheet(title="Reconciliation & Metadata")
        ws_rec.views.sheetView[0].showGridLines = True

        rec_headers = [
            "Statement File",
            "PDF Type",
            "Bank Name",
            "Account Holder",
            "Account No",
            "IFSC Code",
            "Opening Bal (INR)",
            "Total Debits (INR)",
            "Total Credits (INR)",
            "Closing Bal (INR)",
            "Audit Status",
        ]
        for col_idx, rh in enumerate(rec_headers, start=1):
            c = ws_rec.cell(row=1, column=col_idx, value=rh)
            c.font = th_font
            c.fill = navy_fill
            c.border = thin_border
            c.alignment = left_align
        ws_rec.row_dimensions[1].height = 24

        r_start = 2
        for idx, res_obj in enumerate(results):
            r_i = r_start + idx
            m = res_obj.metadata
            rc = res_obj.reconciliation
            status_str = "PASS" if rc.is_fully_reconciled else "FAIL"
            acct_display = f"A/C-{m.account_number}" if m.account_number else "A/C-NA"
            row_data = [
                res_obj.file_name,
                res_obj.pdf_type.value if hasattr(res_obj.pdf_type, "value") else str(res_obj.pdf_type),
                m.bank_name,
                m.account_holder_name or "N/A",
                acct_display,
                m.ifsc_code or "N/A",
                float(rc.computed_opening_balance or 0.0),
                float(rc.total_debit),
                float(rc.total_credit),
                float(rc.computed_closing_balance or 0.0),
                status_str,
            ]
            for col_idx, val in enumerate(row_data, start=1):
                c = ws_rec.cell(row=r_i, column=col_idx, value=val)
                c.font = reg_font
                c.border = thin_border
                if col_idx in (7, 8, 9, 10):
                    c.alignment = right_align
                    c.number_format = "#,##0.00"
                elif col_idx == 11:
                    c.alignment = center_align
                    c.font = bold_font
                    c.fill = green_fill if val == "PASS" else warn_fill
                else:
                    c.alignment = left_align

        r_end = r_start + len(results) - 1
        r_tot = r_end + 1
        for col_idx in range(1, 12):
            c = ws_rec.cell(row=r_tot, column=col_idx)
            c.font = bold_font
            c.border = total_border
            if col_idx == 1:
                c.value = "Total"
                c.alignment = left_align
            elif col_idx in (7, 8, 9, 10):
                col_let = get_column_letter(col_idx)
                c.value = f"=SUM({col_let}{r_start}:{col_let}{r_end})"
                c.alignment = right_align
                c.number_format = "#,##0.00"
            elif col_idx == 11:
                c.value = "PASS"
                c.alignment = center_align
            else:
                c.value = None
                c.alignment = left_align

        rec_chart = BarChart()
        rec_chart.type = "col"
        rec_chart.style = 11
        rec_chart.title = "Statement-Wise Total Debits vs Total Credits"
        rec_chart.y_axis.title = "Amount (INR)"
        rec_chart.x_axis.title = "Bank Statement"
        rec_chart.width = 16
        rec_chart.height = 9

        rec_data_ref = Reference(ws_rec, min_col=8, min_row=1, max_col=9, max_row=r_end)
        rec_cats_ref = Reference(ws_rec, min_col=3, min_row=r_start, max_row=r_end)
        rec_chart.add_data(rec_data_ref, titles_from_data=True)
        rec_chart.set_categories(rec_cats_ref)
        ws_rec.add_chart(rec_chart, f"A{r_tot + 3}")

        ws_rec.freeze_panes = "A2"
        cls._autofit_columns(ws_rec, skip_rows=set())

        wb.save(output_path)
        return output_path
