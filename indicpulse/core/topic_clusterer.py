"""
IndicPulse – Step 4: Semantic Topic Clustering
===============================================
Uses IndicSBERT (l3cube-pune/indic-sentence-bert-nli) + BERTopic (UMAP + HDBSCAN)
to discover discussion themes in constructive (non-toxic) comments.

Engagement-weighted scoring: W = log10(likes + 1) + 1
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

INDICBERT_MODEL = "l3cube-pune/indic-sentence-bert-nli"


# ──────────────────────────────────────────────
# ENGAGEMENT WEIGHT
# ──────────────────────────────────────────────

def engagement_weight(like_count: int) -> float:
    """W = log10(likes + 1) + 1  — never zero, log-scaled."""
    return float(np.log10(max(like_count, 0) + 1) + 1)


def add_engagement_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'engagement_weight' column to dataframe."""
    df = df.copy()
    df["engagement_weight"] = df["like_count"].apply(engagement_weight)
    return df


# ──────────────────────────────────────────────
# EMBEDDING GENERATION & HARDWARE ACCELERATION
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


_CACHED_EMBEDDER: Optional["IndicSBERTEmbedder"] = None


def get_embedder(model_name: str = INDICBERT_MODEL, device: Optional[str] = None) -> "IndicSBERTEmbedder":
    """Return singleton cached IndicSBERTEmbedder instance in RAM, avoiding reloads."""
    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is None:
        target_device = device or get_optimal_device()
        _CACHED_EMBEDDER = IndicSBERTEmbedder(model_name=model_name, device=target_device)
        try:
            _CACHED_EMBEDDER.load()
        except Exception as e:
            logger.warning(f"Could not pre-load IndicSBERT: {e}")
    return _CACHED_EMBEDDER


class IndicSBERTEmbedder:
    """Wrapper for l3cube-pune/indic-sentence-bert-nli."""

    def __init__(self, model_name: str = INDICBERT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self):
        """Load IndicSBERT model."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"IndicSBERT loaded on {device}.")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode texts to dense embeddings."""
        if self._model is None:
            self.load()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )


# ──────────────────────────────────────────────
# BERTOPIC CLUSTERING
# ──────────────────────────────────────────────

def build_topic_model(
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    hdbscan_min_cluster_size: int = 5,
    top_n_words: int = 8,
    min_topic_size: int = 3,
) -> Any:
    """Build a BERTopic model with UMAP + HDBSCAN."""
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as e:
        raise ImportError(f"BERTopic dependencies missing: {e}. Run: pip install bertopic umap-learn hdbscan")

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None,
        min_df=2,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=top_n_words,
        min_topic_size=min_topic_size,
        verbose=False,
        calculate_probabilities=True,
    )

    return topic_model


def _fallback_keyword_clustering(df_constructive: pd.DataFrame, top_n_words: int = 8) -> Tuple[pd.DataFrame, Any, Dict]:
    """
    Lightweight heuristic clustering fallback when sentence-transformers or BERTopic
    is not installed. Clusters comments into intuitive thematic buckets based on
    Hinglish and English domain keywords.
    """
    themes = {
        0: {
            "label": "Video Production & Editing",
            "keywords": ["editing", "video", "camera", "transition", "transitions", "color", "grading", "clean", "smooth", "thumbnail", "production"],
        },
        1: {
            "label": "Audio & Sound Quality",
            "keywords": ["audio", "sound", "voice", "music", "loud", "volume", "background", "dab", "low"],
        },
        2: {
            "label": "Pricing, Delivery & Service",
            "keywords": ["price", "pricing", "mehenga", "delivery", "refund", "service", "customer", "packaging", "expensive", "cost"],
        },
        3: {
            "label": "Script, Pacing & Research",
            "keywords": ["script", "research", "speed", "explanation", "notes", "concept", "data", "point", "weak", "information", "detail"],
        },
        4: {
            "label": "Audience Hype & Appreciation",
            "keywords": ["fire", "love", "subscriber", "badiya", "mast", "superb", "outstanding", "waah", "best", "keep", "genuine"],
        },
    }

    assigned_topics = []
    for _, row in df_constructive.iterrows():
        text_lower = str(row.get("text_for_topic", "")).lower()
        matched_tid = -1
        max_matches = 0
        for tid, tinfo in themes.items():
            matches = sum(1 for kw in tinfo["keywords"] if kw in text_lower)
            if matches > max_matches:
                max_matches = matches
                matched_tid = tid
        assigned_topics.append(matched_tid if max_matches > 0 else -1)

    df_constructive["topic_id"] = assigned_topics
    topic_label_map = {tid: tinfo["label"] for tid, tinfo in themes.items()}
    topic_label_map[-1] = "General Discussion"
    df_constructive["topic_label"] = df_constructive["topic_id"].map(lambda x: topic_label_map.get(x, "General Discussion"))

    topic_summary = {}
    for tid_raw in df_constructive["topic_id"].unique():
        tid = int(tid_raw)
        cluster = df_constructive[df_constructive["topic_id"] == tid_raw].copy()
        if cluster.empty:
            continue
        if "engagement_weight" in cluster.columns:
            cluster = cluster.sort_values("engagement_weight", ascending=False)
        else:
            cluster = cluster.sort_values("like_count", ascending=False)

        top_words = themes.get(tid, {}).get("keywords", ["discussion", "feedback"])[:top_n_words]
        rep_comments = (
            cluster.head(5)[["text", "like_count", "mood_label"]].to_dict("records")
            if "mood_label" in cluster.columns
            else cluster.head(5)[["text", "like_count"]].to_dict("records")
        )

        topic_summary[tid] = {
            "label": topic_label_map.get(tid, "General Discussion"),
            "size": int(len(cluster)),
            "top_words": top_words,
            "representative_comments": rep_comments,
            "avg_engagement": float(cluster["engagement_weight"].mean()) if "engagement_weight" in cluster.columns else 1.0,
        }

    return df_constructive, None, topic_summary


def cluster_topics(
    df: pd.DataFrame,
    embedder: Optional[IndicSBERTEmbedder] = None,
    umap_n_neighbors: int = 15,
    hdbscan_min_cluster_size: int = 5,
    top_n_words: int = 8,
) -> Tuple[pd.DataFrame, Any, Dict]:
    """
    Run full semantic topic clustering on non-toxic comments.

    Returns
    -------
    df_with_topics : pd.DataFrame
    topic_model    : BERTopic instance
    topic_summary  : Dict  – {topic_id: {label, size, top_words, representative_comments}}
    """
    constructive_mask = ~df["is_toxic"] if "is_toxic" in df.columns else pd.Series([True] * len(df))
    df_constructive = df[constructive_mask].copy().reset_index(drop=True)

    if len(df_constructive) < 5:
        logger.warning(f"Only {len(df_constructive)} constructive comments — too few to cluster.")
        df["topic_id"] = -1
        df["topic_label"] = "Unclustered"
        return df, None, {}

    texts = df_constructive["text_for_topic"].tolist()
    n_constructive = len(df_constructive)

    # Adaptive min_topic_size scaling for 100 to 2,000 comments
    if n_constructive < 150:
        adaptive_min_topic_size = 5
    elif n_constructive <= 600:
        adaptive_min_topic_size = 10
    else:
        adaptive_min_topic_size = 18

    # Attempt transformer-based embeddings
    embeddings = None
    try:
        if embedder is None:
            embedder = get_embedder()
        logger.info(f"Generating IndicSBERT embeddings for {len(texts)} comments (batch_size=64)...")
        embeddings = embedder.encode(texts, batch_size=64, show_progress=False)
    except Exception as e:
        logger.warning(f"IndicSBERT embedding unavailable ({e}). Using smart heuristic fallback.")
        df_clustered, _, topic_summary = _fallback_keyword_clustering(df_constructive, top_n_words)
        return _merge_topics_back(df, df_clustered), None, topic_summary

    logger.info(f"Fitting BERTopic model with adaptive min_topic_size={adaptive_min_topic_size}...")
    try:
        topic_model = build_topic_model(
            umap_n_neighbors=min(umap_n_neighbors, max(5, len(texts) - 1)),
            hdbscan_min_cluster_size=min(hdbscan_min_cluster_size, max(3, adaptive_min_topic_size // 2)),
            top_n_words=top_n_words,
            min_topic_size=adaptive_min_topic_size,
        )
        topics, _ = topic_model.fit_transform(texts, embeddings)
    except Exception as e:
        logger.warning(f"BERTopic model fitting failed ({e}). Using smart heuristic fallback.")
        df_clustered, _, topic_summary = _fallback_keyword_clustering(df_constructive, top_n_words)
        return _merge_topics_back(df, df_clustered), None, topic_summary

    df_constructive["topic_id"] = topics

    # Generate topic labels
    topic_info = topic_model.get_topic_info()
    topic_label_map = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_label_map[-1] = "Miscellaneous"
            continue
        words = topic_model.get_topic(tid)
        label = " · ".join([w[0].title() for w in words[:3]])
        topic_label_map[tid] = label

    df_constructive["topic_label"] = df_constructive["topic_id"].map(
        lambda x: topic_label_map.get(x, "Other")
    )

    # Build topic summary
    topic_summary = {}
    for tid_raw in df_constructive["topic_id"].unique():
        tid = int(tid_raw)
        cluster = df_constructive[df_constructive["topic_id"] == tid_raw].copy()
        if "engagement_weight" in cluster.columns:
            cluster = cluster.sort_values("engagement_weight", ascending=False)
        else:
            cluster = cluster.sort_values("like_count", ascending=False)

        words = topic_model.get_topic(tid_raw) if tid_raw != -1 else []
        top_words = [w[0] for w in words[:top_n_words]]

        rep_comments = (
            cluster.head(5)[["text", "like_count", "mood_label"]].to_dict("records")
            if "mood_label" in cluster.columns
            else cluster.head(5)[["text", "like_count"]].to_dict("records")
        )

        topic_summary[tid] = {
            "label": topic_label_map.get(tid, "Other"),
            "size": int(len(cluster)),
            "top_words": top_words,
            "representative_comments": rep_comments,
            "avg_engagement": float(cluster["engagement_weight"].mean())
            if "engagement_weight" in cluster.columns
            else 0.0,
        }

    df_out = _merge_topics_back(df, df_constructive)
    logger.info(f"Clustering complete. Topics found: {len(topic_summary)}")
    return df_out, topic_model, topic_summary


def _merge_topics_back(df_full: pd.DataFrame, df_constructive: pd.DataFrame) -> pd.DataFrame:
    """Helper to merge topic columns back into the master dataframe."""
    df_out = df_full.copy()
    if "comment_id" not in df_out.columns:
        df_out["comment_id"] = [f"cid_{i}" for i in range(len(df_out))]
    if "comment_id" not in df_constructive.columns:
        # Match using index if preserved
        df_constructive["comment_id"] = df_out.loc[df_constructive.index, "comment_id"].values if set(df_constructive.index).issubset(df_out.index) else [f"cid_{i}" for i in range(len(df_constructive))]

    topic_cols = df_constructive[["comment_id", "topic_id", "topic_label"]]
    if "topic_id" not in df_out.columns:
        df_out["topic_id"] = -1
        df_out["topic_label"] = "Toxic (Quarantined)"
    df_out = df_out.merge(topic_cols, on="comment_id", how="left", suffixes=("_old", ""))
    for col in ["topic_id", "topic_label"]:
        if f"{col}_old" in df_out.columns:
            df_out[col] = df_out[col].fillna(df_out[f"{col}_old"])
            df_out.drop(columns=[f"{col}_old"], inplace=True)

    df_out["topic_id"] = df_out["topic_id"].fillna(-1).astype(int)
    df_out["topic_label"] = df_out["topic_label"].fillna("Toxic (Quarantined)")
    return df_out
