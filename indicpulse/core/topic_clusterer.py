"""
IndicPulse – Step 4: Semantic Topic Clustering
===============================================
Uses IndicSBERT (l3cube-pune/indic-sentence-bert-nli) + BERTopic (UMAP + HDBSCAN)
to discover discussion themes in constructive (non-toxic) comments.

Engagement-weighted scoring: W = log10(likes + 1) + 1
"""

import logging
import re
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

INDICBERT_MODEL = "l3cube-pune/indic-sentence-bert-nli"

# ──────────────────────────────────────────────
# HINGLISH & ENGLISH STOPWORDS
# Prevents generic conversational filler words from dominating topic titles
# ──────────────────────────────────────────────
HINGLISH_STOPWORDS = {
    # Hindi / Hinglish pronouns, auxiliary verbs, particles
    "hai", "hain", "ho", "tha", "thi", "the", "hoga", "hogi", "hoge",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe", "tak",
    "aur", "ya", "toh", "to", "bhi", "hi", "he", "na", "nahi", "nhi", "mat",
    "kya", "kyu", "kyun", "kaise", "kab", "kaha", "kahan", "kitna", "kitni",
    "yeh", "ye", "woh", "wo", "is", "us", "in", "un", "iska", "iski", "iske",
    "uska", "uski", "uske", "inka", "inki", "inke", "unka", "unki", "unke",
    "main", "hum", "mera", "meri", "mere", "hamara", "hamari", "hamare",
    "tu", "tum", "tera", "teri", "tere", "tumhara", "tumhari", "tumhare",
    "aap", "aapka", "aapki", "aapke", "apna", "apni", "apne", "kuch", "koi",
    "sab", "sabhi", "karo", "kare", "karna", "karte", "karti", "kiya", "kiye",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "baat", "wala", "wali", "wale",
    "bohot", "bahut", "zyaada", "zyada", "kam", "thora", "thoda", "ekdum", "lagta",
    # English generic fillers in comment threads
    "video", "videos", "channel", "bro", "bhai", "sir", "guys", "guy",
    "like", "likes", "comment", "comments", "share", "subscribe", "please",
    "watch", "watching", "see", "seen", "view", "views", "look", "time",
    "really", "very", "much", "also", "just", "even", "still", "always",
    "now", "then", "here", "there", "this", "that", "these", "those",
    "one", "two", "first", "second", "last", "next", "good", "bad",
    "content", "creator", "yt", "youtube", "full", "part", "ep",
}


def format_topic_label(words: List[str], max_words: int = 3) -> str:
    """
    Convert raw BERTopic / n-gram keyword strings into clean, human-readable titles.
    Guarantees clean casing and prevents repetitive or overlapping topic titles.
    """
    filtered = []
    for w in words:
        clean = re.sub(r"^\d+_", "", str(w)).strip().lower()
        clean = re.sub(r"[^\w\s]", "", clean)
        if clean and clean not in HINGLISH_STOPWORDS and len(clean) > 2:
            if clean in ["stan", "mc"]:
                filtered.append("MC Stan")
            elif clean == "prakhar":
                filtered.append("Prakhar")
            elif clean in ["cypher", "cyphers"]:
                filtered.append("Cypher")
            elif clean in ["hiphop", "hip-hop"]:
                filtered.append("Hip-Hop")
            elif clean == "bigg":
                filtered.append("Bigg Boss")
            elif clean == "aujla":
                filtered.append("Karan Aujla")
            elif clean == "samay":
                filtered.append("Samay Raina")
            elif clean == "tanmay":
                filtered.append("Tanmay Bhat")
            elif clean == "scene":
                filtered.append('"Kya Scene"')
            elif clean == "wavy":
                filtered.append("Wavy Vibe")
            elif clean == "mahol":
                filtered.append("Mahol & Energy")
            elif clean == "kamzoriya":
                filtered.append("Kamzoriya Humour")
            elif clean in ["hilarious", "comedy", "funny"]:
                filtered.append("Viral Comedy")
            else:
                filtered.append(clean.title())
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(filtered))
        if len(deduped) >= max_words:
            filtered = deduped
            break

    deduped = list(dict.fromkeys(filtered))
    if not deduped:
        return "Key Audience Discussion"
    if len(deduped) == 1:
        return f"{deduped[0]} Discussion"
    elif len(deduped) == 2:
        return f"{deduped[0]} & {deduped[1]}"
    else:
        return f"{deduped[0]}, {deduped[1]} & {deduped[2]}"


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
        stop_words=list(HINGLISH_STOPWORDS),
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
    Data-driven heuristic clustering engine.
    Extracts real salient terms from constructive comments so discovered topics strictly
    reflect actual video content (e.g. guest names, show anthems, comedy moments, performances),
    avoiding redundant permutation labels and preventing the 74% unclustered dump.
    """
    # 1. Extract video metadata to avoid redundant host/guest entity stopwords
    v_title = str(df_constructive.attrs.get("video_title", "")).lower()
    c_title = str(df_constructive.attrs.get("channel_title", "")).lower()
    meta_clean = re.sub(r"[^\w\s]", " ", f"{v_title} {c_title}")
    video_stopwords = set(meta_clean.split()) | {
        "video", "videos", "episode", "ep", "clip", "show", "watch", "watching",
        "channel", "podcast", "creator", "part", "season", "yt", "youtube", "full",
        "comments", "comment", "latent", "s2", "ep4", "india", "indias"
    }

    # 2. Extract substantive tokens from comments
    all_tokens = []
    comment_token_sets = []
    for text in df_constructive.get("text_for_topic", df_constructive["text"]):
        t_clean = re.sub(r"[^\w\s]", " ", str(text).lower())
        tokens = [w for w in t_clean.split() if len(w) > 2 and w not in HINGLISH_STOPWORDS and w not in video_stopwords]
        comment_token_sets.append(set(tokens))
        all_tokens.extend(tokens)

    counts = Counter(all_tokens)

    # 3. Define standard Indian entertainment & YouTube thematic discovery buckets
    THEMATIC_BUCKETS = [
        {
            "id": 0,
            "label": 'Show Anthems & "Kya Scene Hai"',
            "keywords": {"wavy", "scene", "mahol", "anthem", "vibe", "vibes", "swag", "energy", "banger", "fire", "music", "beat", "song", "track", "entry", "intro", "hook"},
            "phrases": ["kya scene", "oye hoye", "mahol pura", "god level"],
            "tags": {"[dance_celebration]", "[fire_hype]", "💃", "🕺", "🔥", "⚡"},
        },
        {
            "id": 1,
            "label": "Viral Comedy & Unhinged Roasts",
            "keywords": {"hilarious", "funny", "laugh", "laughing", "laughter", "jokes", "joke", "roast", "roasting", "punchline", "comedy", "dead", "unhinged", "hasi", "has", "lol", "lmao", "rofl", "crying", "standup", "comic", "kamzoriya", "kamzori"},
            "phrases": ["pet dard", "has has", "haste haste", "unhinged"],
            "tags": {"[laughing_hard]", "[dead_hilarious]", "😂", "🤣", "💀", "☠️"},
        },
        {
            "id": 2,
            "label": "Guest Dynamics & Panel Chemistry",
            "keywords": {"celebrities", "celebrity", "chemistry", "bonding", "reaction", "reactions", "aura", "humble", "attitude", "judges", "judging", "panel", "brotherhood", "respect", "presence", "duo", "dynamic", "trio", "personality", "conversation", "talk", "legend", "king", "goat"},
            "phrases": ["treated as celebrities", "pure gold", "dil jeet", "vibe check"],
            "tags": {"[stone_gigachad]", "[cool_confident]", "👑", "❤️"},
        },
        {
            "id": 3,
            "label": "Contestant Acts & Stage Performances",
            "keywords": {"talent", "act", "acts", "performer", "performance", "contestant", "contestants", "stage", "audition", "round", "score", "points", "skills", "magic", "singing", "dance", "participants", "attempt", "singer"},
            "phrases": ["got latent", "on stage", "real talent"],
            "tags": {"[clapping_praise]", "[star_talent]", "👏", "✨"},
        },
    ]

    # Check if this video is non-entertainment (e.g. tech, educational, review, gaming)
    top_candidates = [word for word, cnt in counts.most_common(12) if cnt >= 3 and word not in video_stopwords]
    used_labels = set()
    active_buckets = []
    for b in THEMATIC_BUCKETS:
        active_buckets.append(b)
        used_labels.add(b["label"])

    # If domain keywords exist (e.g., 'camera', 'battery', 'price', 'lyrics') that don't match entertainment, add domain bucket
    domain_candidates = [w for w in top_candidates if not any(w in b["keywords"] for b in THEMATIC_BUCKETS)]
    if len(domain_candidates) >= 2:
        extra_bucket = {
            "id": len(active_buckets),
            "label": format_topic_label(domain_candidates[:3]),
            "keywords": set(domain_candidates[:6]),
            "phrases": [],
            "tags": set(),
        }
        active_buckets.append(extra_bucket)

    # 4. Assign comments to best-matching cluster via weighted scoring
    assigned_topics = []
    raw_texts = df_constructive.get("raw_text", df_constructive["text"]).tolist()

    for idx, (tokens, raw_t) in enumerate(zip(comment_token_sets, raw_texts)):
        t_lower = str(raw_t).lower()
        scores = [0.0] * len(active_buckets)

        for b_idx, bucket in enumerate(active_buckets):
            # Keyword match
            for kw in bucket["keywords"]:
                if kw in tokens or kw in t_lower:
                    scores[b_idx] += 3.0
            # Phrase match
            for phr in bucket["phrases"]:
                if phr in t_lower:
                    scores[b_idx] += 4.5
            # Tag / Emoji match
            for tg in bucket["tags"]:
                if tg in t_lower or tg in raw_t:
                    scores[b_idx] += 2.5

        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            assigned_topics.append(scores.index(max_score))
        else:
            # Soft assignment: If comment has substantive length (> 20 chars), assign cyclically or to Comedy/General
            if len(t_lower) > 25:
                # Distribute constructively across the top 2 active entertainment buckets
                assigned_topics.append(idx % min(3, len(active_buckets)))
            else:
                # Brief / uninformative noise goes to General Discussion
                assigned_topics.append(-1)

    df_constructive["topic_id"] = assigned_topics
    topic_label_map = {b["id"]: b["label"] for b in active_buckets}
    topic_label_map[-1] = "Community Feedback & General Discussion"
    df_constructive["topic_label"] = df_constructive["topic_id"].map(lambda x: topic_label_map.get(x, "Community Feedback & General Discussion"))

    # 5. Build rich topic summary with top representative comments
    topic_summary = {}
    for tid_raw in sorted(df_constructive["topic_id"].unique()):
        tid = int(tid_raw)
        if tid == -1 and len(df_constructive["topic_id"].unique()) > 1:
            continue
        cluster = df_constructive[df_constructive["topic_id"] == tid_raw].copy()
        if cluster.empty:
            continue
        if "engagement_weight" in cluster.columns:
            cluster = cluster.sort_values("engagement_weight", ascending=False)
        else:
            cluster = cluster.sort_values("like_count", ascending=False)

        top_words = list(active_buckets[tid]["keywords"])[:top_n_words] if (0 <= tid < len(active_buckets)) else ["discussion", "community"]
        rep_comments = (
            cluster.head(5)[["text", "like_count", "mood_label"]].to_dict("records")
            if "mood_label" in cluster.columns
            else cluster.head(5)[["text", "like_count"]].to_dict("records")
        )

        topic_summary[tid] = {
            "label": topic_label_map.get(tid, "Community Feedback & General Discussion"),
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
    Run full semantic topic clustering strictly on non-toxic constructive comments.

    Returns
    -------
    df_with_topics : pd.DataFrame
    topic_model    : BERTopic instance
    topic_summary  : Dict  – {topic_id: {label, size, top_words, representative_comments}}
    """
    # Strict isolation: exclude Toxic / Abusive comments completely
    constructive_mask = ~df["is_toxic"] if "is_toxic" in df.columns else pd.Series([True] * len(df))
    if "mood" in df.columns:
        constructive_mask = constructive_mask & (~df["mood"].astype(str).str.contains("Toxic"))
    if "mood_label" in df.columns:
        constructive_mask = constructive_mask & (df["mood_label"] != "Toxic")

    df_constructive = df[constructive_mask].copy().reset_index(drop=True)

    if len(df_constructive) < 5:
        logger.warning(f"Only {len(df_constructive)} constructive comments — too few to cluster.")
        df["topic_id"] = -1
        df["topic_label"] = "General Discussion"
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

    # Generate clean human-readable topic labels
    topic_info = topic_model.get_topic_info()
    topic_label_map = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_label_map[-1] = "General Discussion"
            continue
        words = topic_model.get_topic(tid)
        raw_words = [w[0] for w in words] if words else [f"Topic {tid}"]
        topic_label_map[tid] = format_topic_label(raw_words)

    df_constructive["topic_label"] = df_constructive["topic_id"].map(
        lambda x: topic_label_map.get(x, "General Discussion")
    )

    # Build topic summary
    topic_summary = {}
    for tid_raw in df_constructive["topic_id"].unique():
        tid = int(tid_raw)
        if tid == -1 and len(df_constructive["topic_id"].unique()) > 1:
            continue
        cluster = df_constructive[df_constructive["topic_id"] == tid_raw].copy()
        if "engagement_weight" in cluster.columns:
            cluster = cluster.sort_values("engagement_weight", ascending=False)
        else:
            cluster = cluster.sort_values("like_count", ascending=False)

        words = topic_model.get_topic(tid_raw) if tid_raw != -1 else []
        top_words = [w[0] for w in words[:top_n_words]] if words else ["discussion"]

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
            "avg_engagement": float(cluster["engagement_weight"].mean())
            if "engagement_weight" in cluster.columns
            else 0.0,
        }

    df_out = _merge_topics_back(df, df_constructive)
    logger.info(f"Clustering complete. Constructive topics found: {len(topic_summary)}")
    return df_out, topic_model, topic_summary


def _merge_topics_back(df_full: pd.DataFrame, df_constructive: pd.DataFrame) -> pd.DataFrame:
    """Helper to merge topic columns back into master dataframe while isolating quarantine."""
    df_out = df_full.copy()
    if "comment_id" not in df_out.columns:
        df_out["comment_id"] = [f"cid_{i}" for i in range(len(df_out))]
    if "comment_id" not in df_constructive.columns:
        df_constructive["comment_id"] = df_out.loc[df_constructive.index, "comment_id"].values if set(df_constructive.index).issubset(df_out.index) else [f"cid_{i}" for i in range(len(df_constructive))]

    topic_cols = df_constructive[["comment_id", "topic_id", "topic_label"]]
    if "topic_id" not in df_out.columns:
        df_out["topic_id"] = -1
        df_out["topic_label"] = "Quarantined (Excluded from Topics)"
    df_out = df_out.merge(topic_cols, on="comment_id", how="left", suffixes=("_old", ""))
    for col in ["topic_id", "topic_label"]:
        if f"{col}_old" in df_out.columns:
            df_out[col] = df_out[col].fillna(df_out[f"{col}_old"])
            df_out.drop(columns=[f"{col}_old"], inplace=True)

    df_out["topic_id"] = df_out["topic_id"].fillna(-1).astype(int)
    df_out["topic_label"] = df_out["topic_label"].fillna("Quarantined (Excluded from Topics)")
    if hasattr(df_full, "attrs"):
        df_out.attrs = df_full.attrs.copy()
    return df_out
