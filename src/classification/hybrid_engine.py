"""Hybrid Non-LLM Classification Orchestrator (Phases 1, 2 & 5 Enhanced).

Combines:
1. Heuristic Rule-Based Classifier (32 priority-weighted regex rule groups across 19 categories)
2. Dual-Head Traditional ML Classifier (TF-IDF word/char n-grams + Multinomial NB + L2 Softmax Logistic Regression
   predicting both `category` and `sub_category`)
3. Optional Transformer/MiniLM Classifier (`TransformerMLClassifier` from `models/distilbert_classifier/` if enabled)
4. Agreement boost & fallback logic:
   - Uses ML-predicted `sub_category` (`ml_res.sub_category`) whenever a deterministic rule does not match!
   - Strictly zero LLM API usage.
"""

from __future__ import annotations
from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Tuple
from src.classification.ml_classifier import TraditionalMLClassifier
from src.classification.rule_classifier import HeuristicRuleClassifier
from src.classification.transformer_classifier import TransformerMLClassifier
from src.utils.models import TransactionRecord

logger = logging.getLogger(__name__)


class HybridClassificationEngine:
    """Orchestrates Heuristic Rule-Based, Dual-Head Traditional ML, and optional Transformer classification."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        corpus_csv_path: Optional[str] = None,
        transformer_dir: Optional[str] = None,
    ):
        self.rule_classifier = HeuristicRuleClassifier()
        self.ml_classifier = TraditionalMLClassifier(model_path=model_path, corpus_csv_path=corpus_csv_path)
        self.transformer_classifier = TransformerMLClassifier(model_dir=transformer_dir)

    def classify_transaction(self, txn: TransactionRecord) -> TransactionRecord:
        """Classifies a single TransactionRecord using Heuristic Rules, Dual-Head ML, and optional Transformer."""
        polarity = "CREDIT" if txn.credit > 0 and txn.debit == 0 else "DEBIT"
        amount = txn.credit if txn.credit > 0 else txn.debit
        combined_text = f"{txn.description} {txn.clean_description}".strip()

        rule_res = self.rule_classifier.classify(txn)
        ml_res = self.ml_classifier.predict(
            description=combined_text,
            polarity=polarity,
            amount=amount,
        )
        tr_res = self.transformer_classifier.predict(combined_text) if self.transformer_classifier.is_available else None

        txn.ml_probabilities = ml_res.probabilities

        if rule_res is not None:
            if ml_res.category == rule_res.category:
                txn.category = rule_res.category
                txn.sub_category = rule_res.sub_category
                txn.classification_method = "HYBRID_ENSEMBLE"
                txn.confidence = min(0.995, round(0.65 * rule_res.confidence + 0.35 * ml_res.confidence + 0.05, 3))
                txn.rule_matched = f"{rule_res.rule_id} ({rule_res.matched_token}) + ML Agreement"
            else:
                txn.category = rule_res.category
                txn.sub_category = rule_res.sub_category
                txn.classification_method = "RULE_ENGINE"
                txn.confidence = rule_res.confidence
                txn.rule_matched = f"{rule_res.rule_id} ({rule_res.matched_token})"
        elif tr_res is not None and tr_res.confidence >= max(ml_res.confidence, 0.75):
            txn.category = tr_res.category
            txn.sub_category = ml_res.sub_category if tr_res.category == ml_res.category else tr_res.sub_category
            txn.classification_method = "TRANSFORMER_MODEL"
            txn.confidence = tr_res.confidence
            txn.rule_matched = f"{tr_res.model_name} (conf={tr_res.confidence:.2f})"
        else:
            # Pure Dual-Head Traditional ML prediction (uses ML-predicted sub_category!)
            if ml_res.confidence >= 0.18:
                txn.category = ml_res.category
                txn.sub_category = ml_res.sub_category
                txn.classification_method = "ML_MODEL"
                txn.confidence = ml_res.confidence
                txn.rule_matched = f"ML Top Features: {', '.join(ml_res.top_features[:3])}"
            else:
                if polarity == "CREDIT":
                    txn.category = "Transfers & Peer-to-Peer"
                    txn.sub_category = "Incoming Transfer / Deposit"
                elif txn.payment_channel in ("UPI", "NEFT", "IMPS", "RTGS", "ZELLE", "VENMO", "CASHAPP", "SEPA", "WIRE"):
                    txn.category = "Transfers & Peer-to-Peer"
                    txn.sub_category = "Bank / UPI Transfer"
                else:
                    txn.category = "Shopping & E-Commerce"
                    txn.sub_category = "Online & Retail Shopping"
                txn.classification_method = "ML_MODEL"
                txn.confidence = round(max(ml_res.confidence, 0.45), 3)
                txn.rule_matched = f"ML Fallback ({', '.join(ml_res.top_features[:2])})"

        # Strict Polarity Guard: Never allow Salary & Income (Payroll / Salary) on DEBIT transactions
        if polarity == "DEBIT" and txn.category == "Salary & Income" and txn.sub_category in ("Payroll / Salary", "Stipend", "Bonus", "Salary"):
            txn.category = "Transfers & Peer-to-Peer"
            txn.sub_category = "Bank / UPI Transfer"

        return txn

    def classify_all(
        self, transactions: List[TransactionRecord]
    ) -> Tuple[List[TransactionRecord], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Classifies all transactions and computes category-level and monthly aggregations."""
        for txn in transactions:
            self.classify_transaction(txn)

        total_debit_all = sum(t.debit for t in transactions) or 1.0
        total_credit_all = sum(t.credit for t in transactions) or 1.0

        cat_agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "debit_total": 0.0,
                "credit_total": 0.0,
                "net_amount": 0.0,
                "debit_share_pct": 0.0,
                "credit_share_pct": 0.0,
                "avg_confidence": 0.0,
                "methods": defaultdict(int),
            }
        )

        month_agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "txn_count": 0,
                "debit_total": 0.0,
                "credit_total": 0.0,
                "net_cashflow": 0.0,
                "closing_balance": 0.0,
            }
        )

        for txn in transactions:
            c = cat_agg[txn.category]
            c["count"] += 1
            c["debit_total"] = round(c["debit_total"] + txn.debit, 2)
            c["credit_total"] = round(c["credit_total"] + txn.credit, 2)
            c["net_amount"] = round(c["credit_total"] - c["debit_total"], 2)
            c["avg_confidence"] += txn.confidence
            c["methods"][txn.classification_method] += 1

            month_key = txn.date[:7] if len(txn.date) >= 7 else "Unknown"
            m = month_agg[month_key]
            m["txn_count"] += 1
            m["debit_total"] = round(m["debit_total"] + txn.debit, 2)
            m["credit_total"] = round(m["credit_total"] + txn.credit, 2)
            m["net_cashflow"] = round(m["credit_total"] - m["debit_total"], 2)
            if txn.balance is not None:
                m["closing_balance"] = txn.balance

        final_cat_summary: Dict[str, Dict[str, Any]] = {}
        for cat_name, data in sorted(
            cat_agg.items(), key=lambda item: (item[1]["debit_total"] + item[1]["credit_total"]), reverse=True
        ):
            cnt = data["count"]
            final_cat_summary[cat_name] = {
                "count": cnt,
                "debit_total": data["debit_total"],
                "credit_total": data["credit_total"],
                "net_amount": data["net_amount"],
                "debit_share_pct": round((data["debit_total"] / total_debit_all) * 100.0, 2) if total_debit_all else 0.0,
                "credit_share_pct": round((data["credit_total"] / total_credit_all) * 100.0, 2) if total_credit_all else 0.0,
                "avg_confidence": round(data["avg_confidence"] / cnt, 3) if cnt else 0.0,
                "methods": dict(data["methods"]),
            }

        final_month_summary = dict(sorted(month_agg.items(), key=lambda x: x[0]))
        logger.info("Classified %d transactions across %d active categories.", len(transactions), len(final_cat_summary))
        return transactions, final_cat_summary, final_month_summary
