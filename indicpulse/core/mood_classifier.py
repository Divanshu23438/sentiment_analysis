"""
IndicPulse – Step 3: MuRIL Mood & Toxicity Classifier
======================================================
Uses google/muril-base-cased for multilingual Indic text understanding.

Label Schema (4-class):
  0 → Positive (Genuine Praise / Hype)
  1 → Sarcastic / Trolling
  2 → Neutral / Informational
  3 → Toxic / Abusive

Two execution modes:
  A. Fine-tuned model  – load from models/ directory (if available)
  B. Rule-based fallback – keyword + pattern heuristics when no GPU/model

Training entry-point: train_muril() – designed for Google Colab T4 GPU.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

LABEL_MAP = {
    0: "Positive",
    1: "Sarcastic",
    2: "Neutral",
    3: "Toxic",
}

LABEL_COLORS = {
    "Positive": "#10b981",   # Emerald Green
    "Sarcastic": "#f59e0b",  # Amber
    "Neutral": "#6b7280",    # Slate Gray
    "Toxic": "#ef4444",      # Coral Red
}


# ──────────────────────────────────────────────
# RULE-BASED FALLBACK CLASSIFIER
# Works without GPU – used in demo / offline mode
# ──────────────────────────────────────────────

_POSITIVE_SIGNALS = [
    r"\b(fire|mast|badhiya|amazing|outstanding|superb|brilliant|ekdum|jabardast|"
    r"subscribe|loved|best|fantastic|wonderful|perfect|great|excellent|waah|wah|"
    r"fabulous|marvelous|phenomenal|damadar|kamaal)\b",
    r"\[fire_hype\]|\[heart_love\]|\[celebration_praise\]|\[ok_perfect\]|"
    r"\[thumbs_up\]|\[love_amazed\]|\[mind_blown\]|\[rocket_amazing\]|\[hundred_perfect\]",
]

_SARCASTIC_SIGNALS = [
    r"\[clown_mocking\]|\[eye_roll_sarcasm\]|\[smirk_sarcasm\]",
    r"\b(haan haan|bilkul|oh wow|very helpful|so helpful|genius|expert|enlightening|"
    r"obviously|clearly|definitely|sure|right|sure sure)\b.*(\?|!|\.\.\.|😏|🙄|🤡)",
    r"jaise\s+yeh\s+(koi|khud)",
    r"(bahut|ek dum|ekdum)\s+(helpful|useful|informative|great)\s+(video|content)",
]

_TOXIC_SIGNALS = [
    r"\b(chutiya|bhenchod|madarchod|gaand|gandu|lodu|harami|kamine|bastard|"
    r"bakwaas|ghatiya|bekar|bekaar|faltu|ch\*tiya|bkl|g\*\*nd|mc|bc)\b",
    r"\[poop_bad\]|\[trash_bad\]|\[angry_frustrated\]",
    r"\b(hate|disgusting|pathetic|idiot|stupid|moron|loser|waste|rubbish)\b",
]

_NEGATIVE_SIGNALS = [
    r"\b(problem|issue|complaint|broken|wrong|bad|terrible|awful|horrible|"
    r"disappoint|refund|delay|late|expensive|overpriced|galat|mehenga|"
    r"kharab|bekar|fail|crash|bug)\b",
]


def _rule_based_classify(text: str, profanity_flag: bool = False) -> Dict:
    """
    Heuristic classifier using keyword patterns.
    Returns {'label': str, 'score': float, 'label_id': int}
    """
    text_lower = text.lower()

    toxic_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in _TOXIC_SIGNALS
    )
    if profanity_flag:
        toxic_score += 2

    sarcasm_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in _SARCASTIC_SIGNALS
    )

    positive_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in _POSITIVE_SIGNALS
    )

    negative_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in _NEGATIVE_SIGNALS
    )

    scores = {
        "Toxic": toxic_score,
        "Sarcastic": sarcasm_score,
        "Positive": positive_score,
        "Neutral": 0.5,  # baseline
    }

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    total = sum(scores.values()) or 1
    confidence = min(best_score / total, 0.95) if best_score > 0 else 0.5

    label_id_map = {"Positive": 0, "Sarcastic": 1, "Neutral": 2, "Toxic": 3}

    return {
        "label": best_label,
        "score": round(confidence, 3),
        "label_id": label_id_map[best_label],
    }


# ──────────────────────────────────────────────
# HARDWARE ACCELERATION & CACHING HELPERS
# ──────────────────────────────────────────────

def get_optimal_device() -> str:
    """Auto-detect optimal hardware acceleration (CUDA GPU, Apple MPS, or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


_CACHED_CLASSIFIER: Optional["MuRILClassifier"] = None


def get_classifier(model_name: Optional[str] = None, use_gpu: bool = True) -> "MuRILClassifier":
    """Return singleton cached MuRILClassifier instance in RAM, avoiding reloads."""
    global _CACHED_CLASSIFIER
    target_device = get_optimal_device() if use_gpu else "cpu"
    if _CACHED_CLASSIFIER is None:
        _CACHED_CLASSIFIER = MuRILClassifier(device=target_device)
        if model_name and model_name != "google/muril-base-cased":
            _CACHED_CLASSIFIER.MODEL_NAME = model_name
        _CACHED_CLASSIFIER.load()
    return _CACHED_CLASSIFIER


# ──────────────────────────────────────────────
# MURIL TRANSFORMER CLASSIFIER
# ──────────────────────────────────────────────

class MuRILClassifier:
    """
    Wrapper around google/muril-base-cased for 4-class sentiment/mood classification.

    Loads fine-tuned weights from models/muril_finetuned/ if available,
    otherwise loads base MuRIL for feature extraction + rule head.
    """

    MODEL_NAME = "google/muril-base-cased"
    FINETUNED_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "muril_finetuned"
    )

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        """Load MuRIL model and tokenizer. Falls back to rule-based if unavailable."""
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            self.device = self.device or get_optimal_device()
            logger.info(f"Loading MuRIL on device: {self.device}")

            model_path = (
                self.FINETUNED_PATH
                if os.path.exists(self.FINETUNED_PATH)
                else self.MODEL_NAME
            )
            logger.info(f"Model source: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=4,
                ignore_mismatched_sizes=True,
            )
            
            pipeline_dev = 0 if "cuda" in str(self.device).lower() else (-1 if self.device == "cpu" else self.device)
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_dev,
                return_all_scores=False,
                truncation=True,
                max_length=128,
            )
            self._loaded = True
            logger.info("MuRIL loaded successfully.")

        except ImportError:
            logger.warning("transformers/torch not installed. Using rule-based fallback.")
        except Exception as e:
            logger.error(f"Failed to load MuRIL: {e}. Using rule-based fallback.")

    def classify_batch(self, texts: List[str], profanity_flags: List[bool] = None) -> List[Dict]:
        """
        Classify a list of texts.

        Returns list of {'label': str, 'score': float, 'label_id': int}
        """
        if profanity_flags is None:
            profanity_flags = [False] * len(texts)

        if not self._loaded or self.pipeline is None:
            logger.info("Using rule-based classifier.")
            return [
                _rule_based_classify(t, p) for t, p in zip(texts, profanity_flags)
            ]

        results = []
        try:
            raw = self.pipeline(texts, batch_size=32)
            for item, prof_flag in zip(raw, profanity_flags):
                label_raw = item["label"]
                # Map LABEL_X → named label
                if label_raw.startswith("LABEL_"):
                    label_id = int(label_raw.split("_")[1])
                    label = LABEL_MAP.get(label_id, "Neutral")
                else:
                    label = label_raw
                    label_id = {v: k for k, v in LABEL_MAP.items()}.get(label, 2)

                # Override: if profanity found + Neutral predicted, bump to Toxic
                if prof_flag and label == "Neutral":
                    label = "Toxic"
                    label_id = 3

                results.append(
                    {"label": label, "score": round(item["score"], 3), "label_id": label_id}
                )
        except Exception as e:
            logger.error(f"MuRIL inference error: {e}. Falling back to rules.")
            results = [
                _rule_based_classify(t, p) for t, p in zip(texts, profanity_flags)
            ]
        return results


def classify_dataframe(
    df: pd.DataFrame,
    classifier: Optional[MuRILClassifier] = None,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Run mood classification on all preprocessed comments.
    Utilizes cached singleton instance to prevent redundant model loads.

    Adds columns: mood_label, mood_score, is_toxic
    """
    if classifier is None:
        classifier = get_classifier(use_gpu=use_gpu)

    texts = df["text_for_mood"].tolist()
    prof_flags = df["has_profanity"].tolist() if "has_profanity" in df.columns else [False] * len(df)

    logger.info(f"Classifying {len(texts)} comments...")
    results = classifier.classify_batch(texts, prof_flags)

    df = df.copy()
    df["mood_label"] = [r["label"] for r in results]
    df["mood_score"] = [r["score"] for r in results]
    df["is_toxic"] = df["mood_label"] == "Toxic"
    logger.info(f"Classification done. Toxic: {df['is_toxic'].sum()}, "
                f"Positive: {(df['mood_label']=='Positive').sum()}")
    return df


# ──────────────────────────────────────────────
# COLAB FINE-TUNING ENTRY POINT
# ──────────────────────────────────────────────

def train_muril(
    train_csv: str,
    output_dir: str = "models/muril_finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune MuRIL on a labeled dataset.

    Expected CSV columns: text, label (0=Positive, 1=Sarcastic, 2=Neutral, 3=Toxic)

    Designed to run on Google Colab T4 GPU.
    Copy this function into a Colab notebook cell and run.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as e:
        raise ImportError(f"Training dependencies missing: {e}")

    # Load data
    data = pd.read_csv(train_csv)
    assert "text" in data.columns and "label" in data.columns, \
        "CSV must have 'text' and 'label' columns"

    dataset = Dataset.from_pandas(data[["text", "label"]])
    split = dataset.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized = split.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "google/muril-base-cased",
        num_labels=4,
    )

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=f"{output_dir}/logs",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fine-tuned MuRIL saved to: {output_dir}")
    return trainer
