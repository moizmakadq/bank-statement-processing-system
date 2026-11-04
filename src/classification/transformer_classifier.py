"""Optional Transformer / MiniLM / DistilBERT Embedding Classifier Backend (Phase 5).

Provides `TransformerMLClassifier` as an optional third classification tier in `HybridClassificationEngine`:
- Loads fine-tuned weights from `models/distilbert_classifier/` (exported from the Kaggle GPU notebook
  `notebooks/kaggle_distilbert_finetuning.ipynb`) if present.
- Supports `sentence-transformers` (`all-MiniLM-L6-v2`) or `transformers` (`distilbert-base-uncased`)
  with a lightweight linear classification head (`classifier_head.pkl` / `config.json`).
- Gracefully disables itself (`is_available = False`) when offline or when GPU/transformer weights
  have not yet been placed in `models/distilbert_classifier/`.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import os
import pickle
from typing import Dict, List, Optional
import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


@dataclass
class TransformerPrediction:
    category: str
    sub_category: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str


class TransformerMLClassifier:
    """Optional DistilBERT / MiniLM embedding classifier head loaded from `models/distilbert_classifier/`."""

    def __init__(self, model_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = model_dir or os.path.join(base_dir, "models", "distilbert_classifier")
        self.is_available: bool = False
        self.encoder = None
        self.head_weights: Optional[np.ndarray] = None
        self.head_bias: Optional[np.ndarray] = None
        self.classes: List[str] = []
        self.sub_category_map: Dict[str, str] = {}
        self.model_name: str = "all-MiniLM-L6-v2 + LinearHead"

        self._try_load()

    def _try_load(self) -> None:
        head_path = os.path.join(self.model_dir, "classifier_head.pkl")
        if not os.path.exists(head_path):
            logger.debug("Optional transformer head not found at %s; TransformerMLClassifier inactive.", head_path)
            return

        try:
            with open(head_path, "rb") as f:
                state = pickle.load(f)
            self.head_weights = state["weights"]
            self.head_bias = state["bias"]
            self.classes = state["classes"]
            self.sub_category_map = state.get("sub_category_map", {})
            self.model_name = state.get("model_name", self.model_name)

            # Check if sentence_transformers is installed for live embedding inference
            from sentence_transformers import SentenceTransformer  # type: ignore
            encoder_path = os.path.join(self.model_dir, "encoder")
            if os.path.exists(encoder_path):
                self.encoder = SentenceTransformer(encoder_path)
            else:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.is_available = True
            logger.info("Loaded TransformerMLClassifier (%s) from %s", self.model_name, self.model_dir)
        except Exception as exc:
            logger.debug("TransformerMLClassifier optional backend not active: %s", exc)
            self.is_available = False

    def predict(self, description: str) -> Optional[TransformerPrediction]:
        if not self.is_available or self.encoder is None or self.head_weights is None or self.head_bias is None:
            return None

        emb = self.encoder.encode([description], normalize_embeddings=True)[0]
        logits = (emb @ self.head_weights + self.head_bias) * 2.0
        probs = softmax(logits)
        best_idx = int(np.argmax(probs))
        best_cat = self.classes[best_idx]
        best_conf = round(float(probs[best_idx]), 3)

        top_probs = {
            self.classes[i]: round(float(probs[i]), 4)
            for i in np.argsort(probs)[::-1][:5]
        }
        return TransformerPrediction(
            category=best_cat,
            sub_category=self.sub_category_map.get(best_cat, "General"),
            confidence=best_conf,
            probabilities=top_probs,
            model_name=self.model_name,
        )
