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
    if hasattr(df, "attrs"):
        df_res.attrs = df.attrs.copy()
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
    if hasattr(df, "attrs"):
        df_clust.attrs = df.attrs.copy()
    return df_clust


def generate_executive_summary(
    df_clustered: pd.DataFrame,
    api_key: str,
    video_title: Optional[str] = None,
    channel_title: Optional[str] = None,
) -> str:
    """Synthesize 4-part video-specific action report using Gemini 2.5 Flash and format as clean Markdown."""
    v_title = video_title or df_clustered.attrs.get("video_title", "Video Analysis")
    c_title = channel_title or df_clustered.attrs.get("channel_title", "Creator")

    # Build topic_summary dictionary from df_clustered if not available
    topic_summary = {}
    for tid in df_clustered["topic_id"].unique():
        cluster = df_clustered[df_clustered["topic_id"] == tid]
        lbl = cluster["topic_label"].iloc[0] if "topic_label" in cluster.columns else "Cluster"
        topic_summary[int(tid)] = {
            "label": lbl,
            "size": len(cluster),
            "top_words": cluster["text_for_topic"].str.split().explode().value_counts().head(6).index.tolist() if "text_for_topic" in cluster.columns else [],
            "representative_comments": cluster.head(4)[["text", "like_count", "mood_label"]].to_dict("records") if "mood_label" in cluster.columns else [],
            "avg_engagement": cluster["engagement_weight"].mean() if "engagement_weight" in cluster.columns else 1.0,
        }

    raw_summary = _gen_exec_summary(
        topic_summary=topic_summary,
        df=df_clustered,
        api_key=api_key,
        video_title=v_title,
        channel_title=c_title,
    )

    vc = raw_summary.get("viewer_consensus", {})
    wvl = raw_summary.get("what_viewers_loved", {})
    cd = raw_summary.get("criticisms_and_debates", {})
    vr = raw_summary.get("viewer_requests_and_next_steps", {})

    loved_quotes = "\n".join([f"> 💬 *{q}*" for q in wvl.get("quotes", [])])
    debate_quotes = "\n".join([f"> 💬 *{q}*" for q in cd.get("quotes", [])])
    action_items = "\n".join([f"**{i+1}.** {act}" for i, act in enumerate(vr.get("actionable_steps", []))])

    # Status badge
    if raw_summary.get("_gemini_success"):
        model_used = raw_summary.get("_model_used", "gemini-2.5-flash")
        status_banner = f"""<div style="background:#E2F6EE; border:1px solid #CCEFE2; color:#0F5132; padding:8px 12px; border-radius:10px; font-size:12px; font-weight:700; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
    <span>🟢</span> Live Gemini AI Active &bull; Engine: <code>{model_used}</code>
</div>"""
    elif raw_summary.get("_gemini_error"):
        raw_err = str(raw_summary.get("_gemini_error"))
        if "404" in raw_err:
            status_banner = """<div style="background:#FFFBEB; border:1px solid #FCD34D; border-radius:12px; padding:12px 16px; margin-bottom:14px; font-size:12px; color:#92400E; line-height:1.5;">
    <div style="font-weight:800; font-size:13px; color:#B45309; margin-bottom:4px;">⚠️ Google Gemini API Notice: Generative Language API Key Needed</div>
    Google returned <code>404 NOT_FOUND</code> for <code>generateContent</code>. This occurs when an API key was created in Google Cloud Console without the <b>Generative Language API</b> enabled, or when using a YouTube-restricted key.<br><br>
    <b>Quick 10-Second Fix:</b><br>
    1. Get a free, instant Gemini API key directly from <b><a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#B45309; text-decoration:underline; font-weight:700;">Google AI Studio (aistudio.google.com/app/apikey)</a></b>.<br>
    2. Paste it in the <b>Neural Engine Controls</b> below and click <b>⚡ Synthesize with Gemini</b>.<br>
    <i>(The analysis below is currently derived using grounded heuristics from real comment likes and detected sentiments.)</i>
</div>"""
        else:
            status_banner = f"""<div style="background:#FEF3D6; border:1px solid #FCE4AD; color:#713F12; padding:10px 14px; border-radius:12px; font-size:12px; margin-bottom:14px;">
    <b>⚠️ Notice:</b> Live Gemini call encountered an issue (<code>{raw_err[:120]}</code>). Displaying video-grounded heuristic synthesis derived from real comment likes.
</div>"""
    else:
        status_banner = """<div style="background:#EDE9FE; border:1px solid #DDD6FE; color:#4C1D95; padding:8px 12px; border-radius:10px; font-size:12px; margin-bottom:14px;">
    <b>💡 Grounded Heuristic Briefing:</b> Derived from top-liked comments across discovered topics. Enter your Gemini API key below to activate live neural synthesis.
</div>"""

    md_output = f"""{status_banner}

### 🎯 What Viewers Actually Think About This Video
> **Audience Verdict:** **{vc.get('overall_sentiment', 'Engaged & Positive')}** • **Pulse Score:** `{vc.get('pulse_score', 8.5)}/10`  
> **Summary:** {vc.get('subject_summary', f'Strong viewer engagement around discussions in {v_title}.')}  
> **Cultural Nuance:** {vc.get('cultural_nuance', 'Hinglish slang and regional expressions detected across top reactions.')}

---

### ❤️ What Viewers Loved Most
{wvl.get('summary', 'Audience expressed strong appreciation for the conversation and guest delivery.')}  
- **Top Acclaimed Element:** **{wvl.get('top_praised_moment', 'Key conversational segments')}**  
{loved_quotes}

---

### ⚡ Viewer Criticisms, Debates & Friction
{cd.get('summary', 'Debates centered on discussion interpretations and humorous peer banter.')}  
- **Primary Debate / Discussion:** **{cd.get('key_controversy', 'Differing perspectives on conversational statements')}**  
{debate_quotes}

---

### 🚀 Top Viewer Demands & What They Want Next
{vr.get('summary', 'Audience actively asked for follow-up content and related guest collaborations.')}  
- **Top Community Request:** **{vr.get('top_request', 'Part 2 / uncut conversation release')}**  
{action_items}
"""
    return md_output
