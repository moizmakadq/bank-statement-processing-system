"""Traditional Machine Learning Transaction Classifier (Phase 2 Enhanced — Strictly Non-LLM).

Enhancements Implemented:
1. Externalized Training Corpus (`data/training_corpus.csv`):
   - Automatically loads 2,300+ training examples across all 19 primary categories and 36 sub-categories
     (including 1,000+ international transactions, 1,000+ P2P transfer variations, and OCR-corrupted edge cases).
2. Dual-Head ML Prediction (Category + Sub-Category):
   - Primary Ensemble (Multinomial Naive Bayes + L2 Softmax Logistic Regression) predicts `category` (19 classes).
   - Secondary Ensemble predicts `sub_category` conditioned on TF-IDF features + predicted primary category!
3. Rigorous Evaluation Methodology:
   - Stratified 80/20 Train/Test Split (`evaluate_train_test_split`)
   - 5-Fold Stratified Cross-Validation with Macro-F1 scoring (`evaluate_cross_validation`)
   - Per-class Precision, Recall, F1, Support report and full Confusion Matrix generation
"""

from __future__ import annotations
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import logging
import math
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
from src.utils.generate_training_corpus import build_training_corpus

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    category: str
    sub_category: str
    confidence: float
    sub_category_confidence: float
    probabilities: Dict[str, float]
    top_features: List[str]


class TraditionalMLClassifier:
    """Dual-Head TF-IDF (Word + Char N-gram) + Multinomial Naive Bayes & L2 Logistic Regression Ensemble."""

    DEFAULT_SUBCATEGORY_MAP = {
        "Salary & Income": "Payroll / Salary",
        "Food & Dining": "Restaurants & Food Delivery",
        "Groceries & Supermarkets": "Supermarket & Quick Commerce",
        "Shopping & E-Commerce": "Online & Retail Shopping",
        "Utilities & Bills": "Electricity, Gas & Water",
        "Rent & Housing": "House Rent & Maintenance",
        "Travel & Transport": "Flights, Rail & Hotels",
        "Healthcare & Medical": "Hospitals, Labs & Pharmacy",
        "Entertainment & Subscriptions": "OTT, Gaming & Movies",
        "Investments & Insurance": "Mutual Funds & Equities",
        "Loans, EMI & Credit Cards": "Loan EMI",
        "ATM & Cash Withdrawal": "ATM Cash Withdrawal",
        "Transfers & Peer-to-Peer": "Bank / UPI Transfer",
        "Bank Fees, Taxes & Charges": "Bank Charges & Taxes",
        "Education & Learning": "Online Courses",
        "Government & Taxes": "Income Tax",
        "Charity & Donations": "NGO",
        "Personal Care & Wellness": "Gym",
        "Wallet & Recharge": "Digital Wallet Top-ups",
    }

    def __init__(self, model_path: Optional[str] = None, corpus_csv_path: Optional[str] = None):
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

        # Primary Category Model parameters
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.nb_log_prior: Optional[np.ndarray] = None
        self.nb_log_prob: Optional[np.ndarray] = None
        self.lr_weights: Optional[np.ndarray] = None
        self.lr_bias: Optional[np.ndarray] = None

        # Sub-Category Model parameters (Dual-Head)
        self.sub_classes: List[str] = []
        self.sub_to_idx: Dict[str, int] = {}
        self.cat_to_valid_subs: Dict[str, List[int]] = {}
        self.sub_nb_log_prior: Optional[np.ndarray] = None
        self.sub_nb_log_prob: Optional[np.ndarray] = None

        self.is_trained: bool = False

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.corpus_csv_path = corpus_csv_path or os.path.join(base_dir, "data", "training_corpus.csv")

        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
                # Rebuild if older model lacks dual-head sub_classes or 19 categories
                if len(self.classes) < 19 or not self.sub_classes:
                    logger.info("Upgrading persisted ML model to 19-category dual-head architecture...")
                    corpus = self.load_corpus_from_csv(self.corpus_csv_path)
                    self.train(corpus)
                    self.save(model_path)
            except Exception as exc:
                logger.warning("Failed to load existing ML model (%s); retraining from CSV.", exc)
                corpus = self.load_corpus_from_csv(self.corpus_csv_path)
                self.train(corpus)
                if model_path:
                    self.save(model_path)
        else:
            corpus = self.load_corpus_from_csv(self.corpus_csv_path)
            self.train(corpus)
            if model_path:
                self.save(model_path)

    @classmethod
    def load_corpus_from_csv(cls, csv_path: str) -> List[Tuple[str, str, float, str, str]]:
        """Loads externalized training corpus from `data/training_corpus.csv` (generating it if not yet present)."""
        if not os.path.exists(csv_path):
            logger.info("Generating externalized training corpus at %s", csv_path)
            build_training_corpus(csv_path)

        corpus: List[Tuple[str, str, float, str, str]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                corpus.append(
                    (
                        row["description"],
                        row["polarity"],
                        float(row["amount"]),
                        row["category"],
                        row["sub_category"],
                    )
                )
        logger.info("Loaded %d training samples from %s", len(corpus), csv_path)
        return corpus

    @staticmethod
    def _extract_tokens(text: str, polarity: str, amount: float) -> List[str]:
        """Extracts word unigrams, word bigrams, char 4-grams, and polarity/amount bucket tokens."""
        clean = re.sub(r"[^A-Z0-9@\s]", " ", (text or "").upper())
        words = [w for w in clean.split() if len(w) >= 2 and not w.isdigit()]

        tokens: List[str] = []
        for w in words:
            tokens.append(f"W_{w}")
        for i in range(len(words) - 1):
            tokens.append(f"BG_{words[i]}_{words[i+1]}")
        for w in words:
            if len(w) >= 4:
                padded = f"#{w}#"
                for i in range(len(padded) - 3):
                    tokens.append(f"C4_{padded[i:i+4]}")

        tokens.append(f"POL_{polarity.upper()}")
        if amount <= 100:
            tokens.append("AMT_MICRO")
        elif amount <= 1000:
            tokens.append("AMT_SMALL")
        elif amount <= 10000:
            tokens.append("AMT_MEDIUM")
        elif amount <= 50000:
            tokens.append("AMT_LARGE")
        else:
            tokens.append("AMT_HIGH")

        return tokens

    def _build_tfidf_matrix(self, docs_tokens: List[List[str]], fit: bool = False, min_df: int = 2) -> np.ndarray:
        """Builds L2-normalized sublinear TF-IDF matrix from tokenized documents."""
        if fit:
            df_counter: Counter = Counter()
            for tokens in docs_tokens:
                for tok in set(tokens):
                    df_counter[tok] += 1
            sorted_toks = sorted(tok for tok, count in df_counter.items() if count >= min_df or tok.startswith(("W_", "POL_", "AMT_")))
            self.vocab = {tok: idx for idx, tok in enumerate(sorted_toks)}
            self.feature_names = sorted_toks
            n_docs = len(docs_tokens)
            idf_vec = np.zeros(len(self.vocab), dtype=np.float64)
            for tok, idx in self.vocab.items():
                df = df_counter[tok]
                idf_vec[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            self.idf = idf_vec

        assert self.idf is not None
        n_samples = len(docs_tokens)
        n_features = len(self.vocab)
        rows_list, cols_list, vals_list = [], [], []

        for row_idx, tokens in enumerate(docs_tokens):
            tf_counts = Counter(tokens)
            row_cols = []
            row_vals = []
            norm_sq = 0.0
            for tok, count in tf_counts.items():
                col_idx = self.vocab.get(tok)
                if col_idx is not None:
                    v = (1.0 + math.log(count)) * float(self.idf[col_idx])
                    row_cols.append(col_idx)
                    row_vals.append(v)
                    norm_sq += v * v
            if norm_sq > 0.0:
                inv_norm = 1.0 / math.sqrt(norm_sq)
                for c_i, v in zip(row_cols, row_vals):
                    rows_list.append(row_idx)
                    cols_list.append(c_i)
                    vals_list.append(v * inv_norm)

        return sparse.csr_matrix((vals_list, (rows_list, cols_list)), shape=(n_samples, n_features), dtype=np.float64)

    def train(
        self,
        corpus: List[Tuple[str, str, float, str, str]],
        alpha: float = 0.20,
        l2_reg: float = 0.6,
    ) -> None:
        """Trains both the Primary Category Ensemble and Secondary Sub-Category Model."""
        docs_tokens = [self._extract_tokens(desc, pol, amt) for desc, pol, amt, _, _ in corpus]
        labels = [cat for _, _, _, cat, _ in corpus]
        sub_labels = [sub for _, _, _, _, sub in corpus]

        unique_classes = sorted(set(labels))
        self.classes = unique_classes
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        y = np.array([self.class_to_idx[c] for c in labels], dtype=np.int64)

        unique_subs = sorted(set(sub_labels))
        self.sub_classes = unique_subs
        self.sub_to_idx = {s: i for i, s in enumerate(unique_subs)}
        y_sub = np.array([self.sub_to_idx[s] for s in sub_labels], dtype=np.int64)

        # Map each primary category to its valid sub-category indices
        cat_subs_map: Dict[str, set[int]] = defaultdict(set)
        for cat, sub in zip(labels, sub_labels):
            cat_subs_map[cat].add(self.sub_to_idx[sub])
        self.cat_to_valid_subs = {cat: sorted(idxs) for cat, idxs in cat_subs_map.items()}

        X = self._build_tfidf_matrix(docs_tokens, fit=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        n_subs = len(self.sub_classes)

        # 1. Fit Primary Multinomial Naive Bayes (Sparse One-Hot Aggregation)
        Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y_onehot[np.arange(n_samples), y] = 1.0
        class_counts = Y_onehot.sum(axis=0)
        feature_counts = (Y_onehot.T @ X).astype(np.float64)

        self.nb_log_prior = np.log((class_counts + 1.0) / (n_samples + n_classes))
        smoothed_fc = feature_counts + alpha
        smoothed_totals = smoothed_fc.sum(axis=1, keepdims=True)
        self.nb_log_prob = np.log(smoothed_fc / smoothed_totals)

        # 2. Fit Primary L2-Regularized Multinomial Logistic Regression via L-BFGS

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            W = params[: n_features * n_classes].reshape((n_features, n_classes))
            b = params[n_features * n_classes :]
            logits = X @ W + b
            probs = softmax(logits, axis=1)
            ce_loss = -np.sum(Y_onehot * np.log(np.clip(probs, 1e-12, 1.0))) / n_samples
            reg_loss = 0.5 * l2_reg * np.sum(W * W) / n_samples
            total_loss = float(ce_loss + reg_loss)

            diff = (probs - Y_onehot) / n_samples
            grad_W = (X.T @ diff) + (l2_reg / n_samples) * W
            grad_b = np.sum(diff, axis=0)
            grad = np.concatenate([grad_W.ravel(), grad_b])
            return total_loss, grad

        init_params = np.zeros(n_features * n_classes + n_classes, dtype=np.float64)
        res = minimize(
            loss_and_grad,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 180, "ftol": 1e-6},
        )
        opt_params = res.x
        self.lr_weights = opt_params[: n_features * n_classes].reshape((n_features, n_classes))
        self.lr_bias = opt_params[n_features * n_classes :]

        # 3. Fit Secondary Sub-Category Multinomial Naive Bayes (Sparse Aggregation)
        Y_sub_oh = np.zeros((n_samples, n_subs), dtype=np.float64)
        Y_sub_oh[np.arange(n_samples), y_sub] = 1.0
        sub_counts = Y_sub_oh.sum(axis=0)
        sub_fc = (Y_sub_oh.T @ X).astype(np.float64)

        self.sub_nb_log_prior = np.log((sub_counts + 1.0) / (n_samples + n_subs))
        sub_smoothed = sub_fc + alpha
        sub_totals = sub_smoothed.sum(axis=1, keepdims=True)
        self.sub_nb_log_prob = np.log(sub_smoothed / sub_totals)

        self.is_trained = True
        logger.info(
            "Trained TraditionalMLClassifier on %d samples (%d features, %d categories, %d sub-categories).",
            n_samples, n_features, n_classes, n_subs
        )

    def predict(self, description: str, polarity: str = "DEBIT", amount: float = 500.0) -> MLPrediction:
        """Predicts primary category and sub-category using the Dual-Head Soft-Voting Ensemble."""
        if not self.is_trained or self.nb_log_prob is None or self.lr_weights is None:
            raise RuntimeError("TraditionalMLClassifier is not trained.")

        tokens = self._extract_tokens(description, polarity, amount)
        X = self._build_tfidf_matrix([tokens], fit=False)
        x_vec = X.toarray()[0]

        nb_logits = (self.nb_log_prior + (x_vec @ self.nb_log_prob.T)) * 2.2
        nb_probs = softmax(nb_logits)

        lr_logits = (x_vec @ self.lr_weights + self.lr_bias) * 2.5
        lr_probs = softmax(lr_logits)

        ensemble_probs = 0.55 * lr_probs + 0.45 * nb_probs
        best_idx = int(np.argmax(ensemble_probs))
        best_cat = self.classes[best_idx]
        best_conf = round(float(ensemble_probs[best_idx]), 3)

        # Predict Sub-Category conditioned on predicted primary category
        best_sub = self.DEFAULT_SUBCATEGORY_MAP.get(best_cat, "General")
        sub_conf = best_conf
        if self.sub_nb_log_prob is not None and self.sub_nb_log_prior is not None:
            valid_sub_idxs = self.cat_to_valid_subs.get(best_cat, [])
            if len(valid_sub_idxs) == 1:
                best_sub = self.sub_classes[valid_sub_idxs[0]]
                sub_conf = min(0.99, round(best_conf + 0.05, 3))
            elif len(valid_sub_idxs) > 1:
                sub_logits = (self.sub_nb_log_prior[valid_sub_idxs] + (x_vec @ self.sub_nb_log_prob[valid_sub_idxs].T)) * 2.5
                sub_probs = softmax(sub_logits)
                win_pos = int(np.argmax(sub_probs))
                best_sub = self.sub_classes[valid_sub_idxs[win_pos]]
                sub_conf = round(float(sub_probs[win_pos]), 3)

        prob_dict = {
            self.classes[i]: round(float(ensemble_probs[i]), 4)
            for i in np.argsort(ensemble_probs)[::-1][:5]
        }

        non_zero_indices = np.where(x_vec > 0)[0]
        scored_feats = [
            (self.feature_names[idx], float(x_vec[idx] * self.lr_weights[idx, best_idx]))
            for idx in non_zero_indices
        ]
        scored_feats.sort(key=lambda item: item[1], reverse=True)
        top_features = [f[0] for f in scored_feats[:4]]

        return MLPrediction(
            category=best_cat,
            sub_category=best_sub,
            confidence=best_conf,
            sub_category_confidence=sub_conf,
            probabilities=prob_dict,
            top_features=top_features,
        )

    @staticmethod
    def _compute_classification_metrics(
        y_true: List[str], y_pred: List[str], classes: List[str]
    ) -> Dict[str, Any]:
        """Computes accuracy, macro-precision, macro-recall, macro-F1, per-class metrics, and confusion matrix."""
        n = len(y_true)
        c_idx = {c: i for i, c in enumerate(classes)}
        k = len(classes)
        cm = np.zeros((k, k), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in c_idx and p in c_idx:
                cm[c_idx[t], c_idx[p]] += 1

        per_class: Dict[str, Dict[str, float]] = {}
        precs, recs, f1s = [], [], []
        for i, c_name in enumerate(classes):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            support = int(cm[i, :].sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_class[c_name] = {
                "precision": round(prec * 100.0, 2),
                "recall": round(rec * 100.0, 2),
                "f1_score": round(f1 * 100.0, 2),
                "support": support,
            }
            if support > 0:
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)

        acc = (np.trace(cm) / n) * 100.0 if n > 0 else 0.0
        return {
            "sample_count": n,
            "accuracy_pct": round(float(acc), 2),
            "macro_precision_pct": round(float(np.mean(precs) * 100.0), 2) if precs else 0.0,
            "macro_recall_pct": round(float(np.mean(recs) * 100.0), 2) if recs else 0.0,
            "macro_f1_pct": round(float(np.mean(f1s) * 100.0), 2) if f1s else 0.0,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "classes": classes,
        }

    def evaluate_train_test_split(self, test_ratio: float = 0.20, random_seed: int = 42) -> Dict[str, Any]:
        """Performs an 80/20 stratified train/test split and reports test accuracy, macro-F1, per-class metrics, and confusion matrix."""
        corpus = self.load_corpus_from_csv(self.corpus_csv_path)
        rng = np.random.default_rng(random_seed)

        by_class: Dict[str, List[Tuple[str, str, float, str, str]]] = defaultdict(list)
        for item in corpus:
            by_class[item[3]].append(item)

        train_set, test_set = [], []
        for cat, items in sorted(by_class.items()):
            idxs = np.arange(len(items))
            rng.shuffle(idxs)
            n_test = max(1, int(round(len(items) * test_ratio)))
            for i in idxs[:n_test]:
                test_set.append(items[i])
            for i in idxs[n_test:]:
                train_set.append(items[i])

        eval_clf = TraditionalMLClassifier.__new__(TraditionalMLClassifier)
        eval_clf.vocab = {}
        eval_clf.idf = None
        eval_clf.feature_names = []
        eval_clf.classes = []
        eval_clf.class_to_idx = {}
        eval_clf.sub_classes = []
        eval_clf.sub_to_idx = {}
        eval_clf.cat_to_valid_subs = {}
        eval_clf.is_trained = False
        eval_clf.train(train_set)

        y_true_cat, y_pred_cat = [], []
        y_true_sub, y_pred_sub = [], []
        for desc, pol, amt, true_cat, true_sub in test_set:
            pred = eval_clf.predict(desc, pol, amt)
            y_true_cat.append(true_cat)
            y_pred_cat.append(pred.category)
            y_true_sub.append(true_sub)
            y_pred_sub.append(pred.sub_category)

        metrics = self._compute_classification_metrics(y_true_cat, y_pred_cat, eval_clf.classes)
        sub_correct = sum(1 for t, p in zip(y_true_sub, y_pred_sub) if t == p)
        metrics["train_size"] = len(train_set)
        metrics["test_size"] = len(test_set)
        metrics["sub_category_accuracy_pct"] = round((sub_correct / len(test_set)) * 100.0, 2)
        return metrics

    def evaluate_cross_validation(self, n_folds: int = 5, random_seed: int = 42) -> Dict[str, Any]:
        """Performs 5-fold stratified cross-validation and returns fold metrics and mean macro-F1."""
        corpus = self.load_corpus_from_csv(self.corpus_csv_path)
        rng = np.random.default_rng(random_seed)

        by_class: Dict[str, List[Tuple[str, str, float, str, str]]] = defaultdict(list)
        for item in corpus:
            by_class[item[3]].append(item)

        folds: List[List[Tuple[str, str, float, str, str]]] = [[] for _ in range(n_folds)]
        for cat, items in sorted(by_class.items()):
            idxs = np.arange(len(items))
            rng.shuffle(idxs)
            for pos, idx in enumerate(idxs):
                folds[pos % n_folds].append(items[idx])

        fold_accuracies = []
        fold_f1s = []
        for f_idx in range(n_folds):
            val_set = folds[f_idx]
            tr_set = [item for k in range(n_folds) if k != f_idx for item in folds[k]]

            cv_clf = TraditionalMLClassifier.__new__(TraditionalMLClassifier)
            cv_clf.vocab = {}
            cv_clf.idf = None
            cv_clf.feature_names = []
            cv_clf.classes = []
            cv_clf.class_to_idx = {}
            cv_clf.sub_classes = []
            cv_clf.sub_to_idx = {}
            cv_clf.cat_to_valid_subs = {}
            cv_clf.is_trained = False
            cv_clf.train(tr_set)

            y_true, y_pred = [], []
            for desc, pol, amt, true_cat, _ in val_set:
                pred = cv_clf.predict(desc, pol, amt)
                y_true.append(true_cat)
                y_pred.append(pred.category)

            m = self._compute_classification_metrics(y_true, y_pred, cv_clf.classes)
            fold_accuracies.append(m["accuracy_pct"])
            fold_f1s.append(m["macro_f1_pct"])

        return {
            "n_folds": n_folds,
            "total_samples": len(corpus),
            "fold_accuracies_pct": fold_accuracies,
            "fold_macro_f1s_pct": fold_f1s,
            "mean_accuracy_pct": round(float(np.mean(fold_accuracies)), 2),
            "std_accuracy_pct": round(float(np.std(fold_accuracies)), 2),
            "mean_macro_f1_pct": round(float(np.mean(fold_f1s)), 2),
            "std_macro_f1_pct": round(float(np.std(fold_f1s)), 2),
        }

    def evaluate_training_accuracy(self) -> Dict[str, Any]:
        """Returns comprehensive training, 80/20 test split, and 5-fold cross-validation summary metrics."""
        corpus = self.load_corpus_from_csv(self.corpus_csv_path)
        correct = sum(1 for desc, pol, amt, cat, _ in corpus if self.predict(desc, pol, amt).category == cat)
        train_acc = round((correct / len(corpus)) * 100.0, 2)
        split_res = self.evaluate_train_test_split(test_ratio=0.20)
        return {
            "corpus_size": len(corpus),
            "num_categories": len(self.classes),
            "num_sub_categories": len(self.sub_classes),
            "accuracy_pct": train_acc,
            "test_accuracy_pct": split_res["accuracy_pct"],
            "test_macro_f1_pct": split_res["macro_f1_pct"],
            "sub_category_test_accuracy_pct": split_res["sub_category_accuracy_pct"],
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "classes": self.classes,
                    "class_to_idx": self.class_to_idx,
                    "nb_log_prior": self.nb_log_prior,
                    "nb_log_prob": self.nb_log_prob,
                    "lr_weights": self.lr_weights,
                    "lr_bias": self.lr_bias,
                    "feature_names": self.feature_names,
                    "sub_classes": self.sub_classes,
                    "sub_to_idx": self.sub_to_idx,
                    "cat_to_valid_subs": self.cat_to_valid_subs,
                    "sub_nb_log_prior": self.sub_nb_log_prior,
                    "sub_nb_log_prob": self.sub_nb_log_prob,
                },
                f,
            )

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.vocab = state["vocab"]
        self.idf = state["idf"]
        self.classes = state["classes"]
        self.class_to_idx = state["class_to_idx"]
        self.nb_log_prior = state["nb_log_prior"]
        self.nb_log_prob = state["nb_log_prob"]
        self.lr_weights = state["lr_weights"]
        self.lr_bias = state["lr_bias"]
        self.feature_names = state["feature_names"]
        self.sub_classes = state.get("sub_classes", [])
        self.sub_to_idx = state.get("sub_to_idx", {})
        self.cat_to_valid_subs = state.get("cat_to_valid_subs", {})
        self.sub_nb_log_prior = state.get("sub_nb_log_prior")
        self.sub_nb_log_prob = state.get("sub_nb_log_prob")
        self.is_trained = True
