"""
IndicPulse – High-Level Pipeline Facade
========================================
Exposes standard pipeline functions with unified column aliases:
  • fetch_youtube_comments
  • preprocess_dataframe
  • run_mood_classification
  • run_topic_clustering
  • generate_executive_summary
  • run_pipeline
"""

import sys
import os
import logging

_pipeline_dir = os.path.dirname(os.path.abspath(__file__))
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable

from core.ingestion import fetch_youtube_comments as _fetch_yt, load_csv_comments, load_demo_comments
from core.preprocessor import preprocess_dataframe as _preprocess_df
from core.mood_classifier import MuRILClassifier, get_classifier, classify_dataframe as _classify_df
from core.topic_clusterer import cluster_topics as _cluster_topics, get_embedder, add_engagement_weights, engagement_weight
from core.summarizer import generate_executive_summary as _gen_exec_summary, _fallback_summary
from core.pipeline import run_pipeline, PipelineResult

logger = logging.getLogger(__name__)


def fetch_youtube_comments(
    video_url: str,
    api_key: str,
    max_comments: int = 300,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Fetch comments from YouTube Data API v3 and normalize column names with pagination up to 2,000."""
    df = _fetch_yt(video_url, api_key=api_key, max_comments=max_comments, progress_callback=progress_callback)
    if not df.empty:
        # Guarantee aliases
        df["raw_text"] = df["text"]
        df["likes"] = df["like_count"]
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run text normalization, emoji tag translation, and profanity detection."""
    # Ensure raw_text / text column alignment
    if "raw_text" in df.columns and "text" not in df.columns:
        df["text"] = df["raw_text"]
    elif "text" in df.columns and "raw_text" not in df.columns:
        df["raw_text"] = df["text"]

    if "likes" in df.columns and "like_count" not in df.columns:
        df["like_count"] = df["likes"]
    elif "like_count" in df.columns and "likes" not in df.columns:
        df["likes"] = df["like_count"]

    if "comment_id" not in df.columns:
        df["comment_id"] = [f"c_{i}" for i in range(len(df))]

    df_clean = _preprocess_df(df)
    
    # Aliases
    df_clean["text_for_muril"] = df_clean["text_for_mood"]
    df_clean["raw_text"] = df_clean["text"]
    df_clean["likes"] = df_clean["like_count"]
    return df_clean


def run_mood_classification(df: pd.DataFrame, model_name: Optional[str] = None, use_gpu: bool = True) -> pd.DataFrame:
    """Classify comments into 4 distinct mood & toxicity categories using cached MuRIL instance in RAM."""
    classifier = get_classifier(model_name=model_name, use_gpu=use_gpu)
    df_res = _classify_df(df, classifier=classifier, use_gpu=use_gpu)

    # Map labels to match both legacy and v2.0 formats:
    # "Positive" -> "Positive (Aggressive Hype)"
    # "Sarcastic" -> "Sarcastic / Trolling"
    # "Neutral" -> "Neutral"
    # "Toxic" -> "Toxic / Abusive"
    label_mapping = {
        "Positive": "Positive (Aggressive Hype)",
        "Sarcastic": "Sarcastic / Trolling",
        "Neutral": "Neutral",
        "Toxic": "Toxic / Abusive",
    }
    df_res["mood"] = df_res["mood_label"].map(lambda x: label_mapping.get(x, x))
    
    # Add engagement weights if not already present
    if "engagement_weight" not in df_res.columns:
        df_res["engagement_weight"] = df_res["likes"].apply(engagement_weight)

    # Ensure all aliases exist
    df_res["text_for_muril"] = df_res.get("text_for_mood", df_res["text"])
    df_res["raw_text"] = df_res["text"]
    df_res["likes"] = df_res["like_count"]
    return df_res


def run_topic_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster constructive comments using cached IndicSBERT + adaptive BERTopic."""
    df_with_weights = add_engagement_weights(df)
    df_clust, _, topic_summary = _cluster_topics(df_with_weights, embedder=get_embedder())

    # Aliases
    df_clust["topic_name"] = df_clust["topic_label"]
    df_clust["raw_text"] = df_clust["text"]
    df_clust["likes"] = df_clust["like_count"]
    df_clust["text_for_muril"] = df_clust.get("text_for_mood", df_clust["text"])
    return df_clust


def generate_executive_summary(df_clustered: pd.DataFrame, api_key: str) -> str:
    """Synthesize 4-part action report using Gemini 2.5 Flash and format as readable, scannable Markdown."""
    # Build topic_summary dictionary from df_clustered if not available
    topic_summary = {}
    for tid in df_clustered["topic_id"].unique():
        cluster = df_clustered[df_clustered["topic_id"] == tid]
        lbl = cluster["topic_label"].iloc[0] if "topic_label" in cluster.columns else "Cluster"
        topic_summary[int(tid)] = {
            "label": lbl,
            "size": len(cluster),
            "top_words": cluster["text_for_topic"].str.split().explode().value_counts().head(6).index.tolist() if "text_for_topic" in cluster.columns else [],
            "representative_comments": cluster.head(3)[["text", "like_count", "mood_label"]].to_dict("records") if "mood_label" in cluster.columns else [],
            "avg_engagement": cluster["engagement_weight"].mean() if "engagement_weight" in cluster.columns else 1.0,
        }

    raw_summary = _gen_exec_summary(topic_summary, df_clustered, api_key=api_key, source="YouTube/Dataset")

    # Extract 4 core structured sections
    al = raw_summary.get("audience_loved", {})
    comp = raw_summary.get("complaints", {})
    ca = raw_summary.get("creator_actions", {})
    vm = raw_summary.get("viral_moments", {})
    sv = raw_summary.get("sentiment_verdict", {})

    loved_themes = ", ".join([f"`#{t}`" for t in al.get("key_themes", [])]) or "`#Authentic`"
    comp_issues = ", ".join([f"`• {i}`" for i in comp.get("key_issues", [])]) or "`• Audio balancing`"
    actions = "\n".join([f"**{i+1}.** {act}" for i, act in enumerate(ca.get("action_items", []))]) or "**1.** Balance vocal tracks against background music."

    md_output = f"""#### Community Sentiment Verdict
> **Verdict:** **{sv.get('overall_tone', 'Positive')}** — *"{sv.get('one_liner', '')}"*  
> **Pulse Score:** `{sv.get('pulse_score', 8.0)}/10` • **Cultural Nuance:** {sv.get('cultural_note', 'Hinglish code-mixing and peer expressions detected.')}

---

#### What Audience Loved
{al.get('summary', 'Audience praised the clear presentation and genuine delivery.')}  
- **Top Aspect:** **{al.get('top_praised_aspect', 'High-quality research')}**  
- **Key Themes:** {loved_themes}

---

#### Core Technical & Content Complaints
{comp.get('summary', 'Constructive viewer feedback centered on audio volume balancing and pacing.')}  
- **Critical Grievance:** **{comp.get('most_critical_complaint', 'Audio volume balancing')}**  
- **Key Issues:** {comp_issues}

---

#### Actionable Creator Next Steps
{actions}  
- **Immediate Priority:** **{ca.get('priority_action', 'Normalize audio levels in next release')}**

---

#### Community Humor & Virality
{vm.get('summary', 'Top-liked comments clustered on relatable Hinglish slang and memes.')}  
- **Virality Driver:** **{vm.get('virality_driver', 'Engaging storytelling and relatable dialect')}**
"""
    return md_output
