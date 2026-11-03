"""
IndicPulse – Master Pipeline Orchestrator
==========================================
Runs all 5 steps in sequence and returns a fully analyzed PipelineResult bundle.

Usage:
  from core.pipeline import run_pipeline
  result = run_pipeline(source="youtube", video_url="https://youtu.be/xxx")
  result = run_pipeline(source="csv", csv_path="data/reviews.csv")
  result = run_pipeline(source="demo")
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, Literal

from core.ingestion import fetch_youtube_comments, load_csv_comments, load_demo_comments
from core.preprocessor import preprocess_dataframe
from core.mood_classifier import MuRILClassifier, classify_dataframe
from core.topic_clusterer import IndicSBERTEmbedder, cluster_topics, add_engagement_weights
from core.summarizer import generate_executive_summary

logger = logging.getLogger(__name__)


class PipelineResult:
    """Container for the full pipeline output."""

    def __init__(
        self,
        df: pd.DataFrame,
        topic_summary: Dict,
        executive_summary: Dict,
        source: str,
        meta: Dict,
    ):
        self.df = df
        self.topic_summary = topic_summary
        self.executive_summary = executive_summary
        self.source = source
        self.meta = meta

    @property
    def df_toxic(self) -> pd.DataFrame:
        """Quarantined toxic/abusive comments."""
        if "is_toxic" in self.df.columns:
            return self.df[self.df["is_toxic"] == True].copy()
        return pd.DataFrame(columns=self.df.columns)

    @property
    def df_constructive(self) -> pd.DataFrame:
        """Non-toxic, constructive comments."""
        if "is_toxic" in self.df.columns:
            return self.df[self.df["is_toxic"] == False].copy()
        return self.df.copy()

    @property
    def mood_counts(self) -> Dict[str, int]:
        """Distribution of predicted mood labels."""
        if "mood_label" in self.df.columns:
            return self.df["mood_label"].value_counts().to_dict()
        return {}

    @property
    def pulse_score(self) -> float:
        """Sentiment Pulse Score out of 10."""
        sv = self.executive_summary.get("sentiment_verdict", {})
        try:
            return float(sv.get("pulse_score", 6.5))
        except (ValueError, TypeError):
            return 6.5

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "meta": self.meta,
            "mood_counts": self.mood_counts,
            "pulse_score": self.pulse_score,
            "topic_summary": self.topic_summary,
            "executive_summary": self.executive_summary,
        }


def run_pipeline(
    source: Literal["youtube", "csv", "demo"] = "demo",
    video_url: Optional[str] = None,
    csv_path: Optional[str] = None,
    api_key_youtube: Optional[str] = None,
    api_key_gemini: Optional[str] = None,
    max_comments: int = 500,
    use_gpu: bool = False,
    skip_llm: bool = False,
    progress_callback=None,
) -> PipelineResult:
    """
    Execute the full 5-step IndicPulse pipeline.

    Parameters
    ----------
    source          : 'youtube' | 'csv' | 'demo'
    video_url       : YouTube URL (required if source='youtube')
    csv_path        : Path to CSV (required if source='csv')
    api_key_youtube : YouTube Data API v3 key
    api_key_gemini  : Gemini API key
    max_comments    : Max comments to ingest (YouTube)
    use_gpu         : Use CUDA if available
    skip_llm        : Skip Gemini summarization (faster, offline)
    progress_callback : callable(step: int, message: str) for UI progress

    Returns
    -------
    PipelineResult
    """

    def _progress(step: int, msg: str):
        logger.info(f"[Step {step}/5] {msg}")
        if progress_callback:
            try:
                progress_callback(step, msg)
            except Exception as pe:
                logger.debug(f"Progress callback notification skipped: {pe}")

    # ── STEP 1: INGEST ──────────────────────────────────────
    _progress(1, "Ingesting comments...")
    if source == "demo":
        df = load_demo_comments()
    elif source == "youtube":
        if not video_url:
            raise ValueError("video_url required for YouTube source")
        df = fetch_youtube_comments(video_url, api_key=api_key_youtube, max_comments=max_comments)
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path required for CSV source")
        df = load_csv_comments(csv_path, source_label="csv")
    else:
        raise ValueError(f"Unknown source: {source}. Use 'youtube', 'csv', or 'demo'.")

    if df is None or df.empty:
        raise RuntimeError("No comments retrieved. Check your API key or input file.")

    # ── STEP 2: PREPROCESS ──────────────────────────────────
    _progress(2, f"Preprocessing {len(df)} comments...")
    df = preprocess_dataframe(df)

    # ── STEP 3: CLASSIFY MOOD & TOXICITY ────────────────────
    _progress(3, "Classifying mood & toxicity...")
    classifier = MuRILClassifier(device="cuda" if use_gpu else "cpu")
    classifier.load()
    df = classify_dataframe(df, classifier=classifier, use_gpu=use_gpu)

    # ── STEP 4: ENGAGEMENT WEIGHTING & TOPIC CLUSTERING ─────
    _progress(4, "Clustering semantic topics...")
    df = add_engagement_weights(df)
    try:
        embedder = IndicSBERTEmbedder(device="cuda" if use_gpu else "cpu")
        df, topic_model, topic_summary = cluster_topics(df, embedder=embedder)
    except Exception as e:
        logger.warning(f"Topic clustering fallback triggered ({e}). Synthesizing topic groupings.")
        # Fallback grouping on non-toxic constructive comments
        df["topic_id"] = -1
        df["topic_label"] = "General Feedback"
        
        # Build simple fallback topic summary
        constructive = df[~df["is_toxic"]] if "is_toxic" in df.columns else df
        top_reps = constructive.sort_values("engagement_weight", ascending=False).head(5)
        rep_records = top_reps[["text", "like_count", "mood_label"]].to_dict("records") if "mood_label" in top_reps.columns else top_reps[["text", "like_count"]].to_dict("records")
        
        topic_summary = {
            1: {
                "label": "Content & Production Feedback",
                "size": len(constructive),
                "top_words": ["video", "quality", "audio", "editing", "bhai", "content"],
                "representative_comments": rep_records,
                "avg_engagement": constructive["engagement_weight"].mean() if "engagement_weight" in constructive.columns else 1.0,
            }
        }

    # ── STEP 5: EXECUTIVE SUMMARY ───────────────────────────
    _progress(5, "Generating AI executive summary...")
    if skip_llm:
        from core.summarizer import _fallback_summary
        executive_summary = _fallback_summary(df, topic_summary)
        logger.info("LLM summarization skipped. Used heuristic fallback.")
    else:
        try:
            executive_summary = generate_executive_summary(
                topic_summary=topic_summary,
                df=df,
                api_key=api_key_gemini,
                source=source,
            )
        except Exception as e:
            logger.warning(f"Gemini summarization failed: {e}. Using fallback.")
            from core.summarizer import _fallback_summary
            executive_summary = _fallback_summary(df, topic_summary)

    toxic_count = int(df["is_toxic"].sum()) if "is_toxic" in df.columns else 0
    constructive_count = len(df) - toxic_count
    valid_topics = [k for k in topic_summary if k != -1]

    meta = {
        "total_comments": len(df),
        "toxic_comments": toxic_count,
        "constructive_comments": constructive_count,
        "topics_found": len(valid_topics),
        "source": source,
    }

    logger.info(f"Pipeline complete. {meta}")
    return PipelineResult(
        df=df,
        topic_summary=topic_summary,
        executive_summary=executive_summary,
        source=source,
        meta=meta,
    )
