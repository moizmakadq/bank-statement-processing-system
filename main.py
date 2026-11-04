#!/usr/bin/env python3
"""Command-Line Interface & Batch Execution Runner for the Bank Statement Processing & Classification System."""

from __future__ import annotations
import argparse
import copy
import json
import logging
import os

from src.pipeline import BankStatementPipeline
from src.utils.generate_training_corpus import build_training_corpus
from src.utils.sample_pdf_generator import generate_all_sample_pdfs

logger = logging.getLogger(__name__)


def run_benchmark_and_export(base_dir: str, final_output_dir: str) -> dict:
    """Generates training corpus & sample statements, runs full pipeline, evaluates ML (80/20 + 5-fold CV), and exports CSV/Excel."""
    sample_pdf_dir = os.path.join(base_dir, "sample_data", "pdfs")
    sample_out_dir = os.path.join(base_dir, "sample_data", "outputs")
    corpus_csv_path = os.path.join(base_dir, "data", "training_corpus.csv")
    model_path = os.path.join(base_dir, "models", "traditional_ml_classifier.pkl")

    os.makedirs(sample_pdf_dir, exist_ok=True)
    os.makedirs(sample_out_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    # Always ensure fresh 2,300+ sample training corpus and retrained 19-category dual-head model
    build_training_corpus(corpus_csv_path)
    if os.path.exists(model_path):
        os.unlink(model_path)

    pdf_paths = generate_all_sample_pdfs(sample_pdf_dir)
    au_scanned_path = os.path.join(base_dir, "OCR_TEST", "AU_100_transactions_scanned.pdf")
    if os.path.exists(au_scanned_path):
        pdf_paths["au_scanned"] = au_scanned_path
    pipeline = BankStatementPipeline(model_path=model_path, corpus_csv_path=corpus_csv_path, ocr_dpi=300)

    results = []
    for key, pdf_path in pdf_paths.items():
        logger.info("Processing %s ...", os.path.basename(pdf_path))
        res = pipeline.process_pdf(pdf_path)
        results.append(res)

        csv_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_Classified.csv"
        pipeline.export_to_csv(res, os.path.join(sample_out_dir, csv_name))
        logger.info(
            "  -> Detected Type: %s | Bank: %s | Txns: %d | Reconciled: %.1f%% | Time: %.1f ms",
            res.pdf_type.value,
            res.metadata.bank_name,
            len(res.transactions),
            res.reconciliation.reconciliation_rate,
            res.processing_time_ms,
        )

    combined_csv_path = os.path.join(final_output_dir, "Classified_Bank_Transactions.csv")
    combined_excel_path = os.path.join(final_output_dir, "Bank_Statement_Classification_Report.xlsx")

    merged_res = copy.deepcopy(results[0])
    all_txns = []
    for r in results:
        for t in r.transactions:
            t_copy = copy.deepcopy(t)
            t_copy.txn_id = f"{r.metadata.bank_name.split()[0].upper()}-{t.txn_id}"
            all_txns.append(t_copy)
    merged_res.transactions = all_txns
    pipeline.export_to_csv(merged_res, combined_csv_path)
    pipeline.export_to_csv(merged_res, os.path.join(sample_out_dir, "Classified_Bank_Transactions.csv"))

    pipeline.export_to_excel(results, combined_excel_path)
    pipeline.export_to_excel(results, os.path.join(sample_out_dir, "Bank_Statement_Classification_Report.xlsx"))

    # Compute full ML evaluation report (80/20 split + 5-fold CV + per-class metrics + confusion matrix)
    ml_clf = pipeline.classifier.ml_classifier
    split_metrics = ml_clf.evaluate_train_test_split(test_ratio=0.20)
    cv_metrics = ml_clf.evaluate_cross_validation(n_folds=5)
    summary_eval = ml_clf.evaluate_training_accuracy()

    summary_payload = {
        "ml_evaluation": {
            **summary_eval,
            "stratified_80_20_split": split_metrics,
            "stratified_5_fold_cv": cv_metrics,
        },
        "statements": [r.to_dict() for r in results],
    }
    json_path = os.path.join(sample_out_dir, "processed_statements_bundle.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return {
        "results": results,
        "combined_csv": combined_csv_path,
        "combined_excel": combined_excel_path,
        "json_bundle": json_path,
        "pdf_paths": pdf_paths,
        "ml_evaluation": summary_payload["ml_evaluation"],
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Bank Statement Processing & Classification System (Text + OCR + Non-LLM Rule/ML Engine)"
    )
    parser.add_argument("--input", "-i", nargs="+", help="Path(s) to input PDF bank statement(s)")
    parser.add_argument("--password", "-p", default="", help="Optional password for encrypted PDF")
    parser.add_argument("--csv-out", default="output_transactions.csv", help="Output CSV path")
    parser.add_argument("--excel-out", default="output_report.xlsx", help="Output Excel (.xlsx) path")
    parser.add_argument("--demo", action="store_true", help="Generate sample PDFs and run full benchmark suite")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    final_output_dir = os.path.abspath(os.path.join(base_dir, "output"))

    if args.demo or not args.input:
        run_benchmark_and_export(base_dir, final_output_dir)
        return

    model_path = os.path.join(base_dir, "models", "traditional_ml_classifier.pkl")
    pipeline = BankStatementPipeline(model_path=model_path)
    results = [pipeline.process_pdf(p, password=args.password) for p in args.input]
    pipeline.export_to_csv(results[0], args.csv_out)
    pipeline.export_to_excel(results, args.excel_out)
    logger.info("Exported CSV to %s and Excel to %s", args.csv_out, args.excel_out)


if __name__ == "__main__":
    main()
