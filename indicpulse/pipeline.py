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
import re
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
        if cluster.empty:
            continue
        if "engagement_weight" in cluster.columns:
            cluster_sorted = cluster.sort_values("engagement_weight", ascending=False)
        elif "likes" in cluster.columns:
            cluster_sorted = cluster.sort_values("likes", ascending=False)
        elif "like_count" in cluster.columns:
            cluster_sorted = cluster.sort_values("like_count", ascending=False)
        else:
            cluster_sorted = cluster

        lbl = cluster["topic_label"].iloc[0] if "topic_label" in cluster.columns else "Cluster"
        topic_summary[int(tid)] = {
            "label": lbl,
            "size": len(cluster),
            "top_words": cluster["text_for_topic"].str.split().explode().value_counts().head(6).index.tolist() if "text_for_topic" in cluster.columns else [],
            "representative_comments": cluster_sorted.head(4)[["text", "like_count", "mood_label"]].to_dict("records") if "mood_label" in cluster_sorted.columns else cluster_sorted.head(4)[["text", "like_count"]].to_dict("records"),
            "avg_engagement": float(cluster["engagement_weight"].mean()) if "engagement_weight" in cluster.columns else 1.0,
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

    # Helper to parse quote and likes cleanly
    def format_quote_card(q_str: str, badge_bg: str = "#FEF3D6", badge_color: str = "#92400E", badge_icon: str = "🔥") -> str:
        q_clean = str(q_str).strip()
        # Extract likes if present
        m = re.search(r'\((\d[\d,]*)\s*likes?\)', q_clean, re.IGNORECASE)
        likes_str = m.group(1) if m else ""
        text_clean = re.sub(r'\s*\(\d[\d,]*\s*likes?\)', '', q_clean).strip(' "“’')
        
        like_badge = f'<span style="background:{badge_bg}; color:{badge_color}; font-size:11px; font-weight:700; padding:4px 9px; border-radius:20px; white-space:nowrap; display:inline-flex; align-items:center; gap:4px;">{badge_icon} {likes_str} likes</span>' if likes_str else ''
        
        return f"""<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:11px 15px; margin-top:8px; box-shadow:0 1px 3px rgba(0,0,0,0.03); display:flex; justify-content:space-between; align-items:center; gap:12px;">
    <div style="font-size:13px; color:#1E293B; font-style:italic; line-height:1.5;">&ldquo;{text_clean}&rdquo;</div>
    {like_badge}
</div>"""

    loved_cards = "\n".join([format_quote_card(q, badge_bg="#FEF3D6", badge_color="#92400E", badge_icon="🔥") for q in wvl.get("quotes", [])])
    debate_cards = "\n".join([format_quote_card(q, badge_bg="#F1F5F9", badge_color="#475569", badge_icon="💬") for q in cd.get("quotes", [])])

    # Format actionable steps cards
    action_cards = []
    for i, act in enumerate(vr.get("actionable_steps", [])):
        action_cards.append(f"""<div style="background:#FFFFFF; border:1px solid #FDE68A; border-radius:10px; padding:11px 15px; margin-top:8px; display:flex; align-items:flex-start; gap:10px; box-shadow:0 1px 2px rgba(0,0,0,0.02);">
    <span style="background:#18181B; color:#FFFFFF; font-size:11px; font-weight:800; border-radius:50%; width:20px; height:20px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px;">{i+1}</span>
    <span style="font-size:13px; color:#1E293B; line-height:1.5;">{act}</span>
</div>""")
    action_items_html = "\n".join(action_cards)

    # Status banner
    if raw_summary.get("_gemini_success"):
        model_used = raw_summary.get("_model_used", "gemini-2.0-flash")
        status_banner = f"""<div style="background:#E2F6EE; border:1px solid #CCEFE2; color:#0F5132; padding:8px 14px; border-radius:10px; font-size:12px; font-weight:700; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
    <span>🟢</span> Live Google Gemini Active &bull; Engine: <code>{model_used}</code> &bull; Real Video-Grounded Synthesis
</div>"""
    elif raw_summary.get("_gemini_error"):
        raw_err = str(raw_summary.get("_gemini_error"))
        status_banner = f"""<div style="background:#FEF3D6; border:1px solid #FCE4AD; color:#713F12; padding:10px 14px; border-radius:12px; font-size:12px; margin-bottom:14px;">
    <b>⚠️ Notice:</b> Live Gemini endpoint returned (<code>{raw_err[:120]}</code>). Displaying video-grounded synthesis derived from real comment likes and topic clusters.
</div>"""
    else:
        status_banner = """<div style="background:#EDE9FE; border:1px solid #DDD6FE; color:#4C1D95; padding:8px 14px; border-radius:10px; font-size:12px; margin-bottom:14px; font-weight:600;">
    💡 <b>Grounded Heuristic Synthesis:</b> Derived from top-liked comments across discovered topics. Enter your Gemini API key below to activate live neural synthesis.
</div>"""

    md_output = f"""{status_banner}

<!-- CARD 1: AUDIENCE CONSENSUS & CULTURAL NUANCE -->
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:18px 20px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
        <div style="font-size:15px; font-weight:800; color:#18181B; display:flex; align-items:center; gap:8px;">
            <span>🎯</span> Audience Consensus & Performance Verdict
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="background:#E2F6EE; color:#065F46; font-size:11px; font-weight:700; padding:4px 10px; border-radius:20px; border:1px solid #CCEFE2;">{vc.get('overall_sentiment', 'Strongly Positive (Viral Hype)')}</span>
            <span style="background:#18181B; color:#FFFFFF; font-size:11px; font-weight:800; padding:4px 10px; border-radius:20px;">Pulse {vc.get('pulse_score', 8.8)} / 10</span>
        </div>
    </div>
    <div style="font-size:13.5px; color:#334155; line-height:1.6; margin-bottom:12px;">
        {vc.get('subject_summary', f'Audience engagement for {v_title} is exceptionally strong, driven by viral catchphrases and guest chemistry.')}
    </div>
    <div style="background:#FEF3D6; border:1px solid #FDE68A; border-radius:10px; padding:10px 14px; font-size:12.5px; color:#92400E; line-height:1.5;">
        <b>🗣️ Hinglish Nuance & Internet Culture:</b> {vc.get('cultural_nuance', 'Heavy code-mixing, creator shoutouts, and emotive expressions detected.')}
    </div>
</div>

<!-- CARD 2: WHAT VIEWERS LOVED MOST -->
<div style="background:#E2F6EE; border:1px solid #CCEFE2; border-radius:14px; padding:18px 20px; margin-bottom:14px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
        <div style="font-size:15px; font-weight:800; color:#065F46; display:flex; align-items:center; gap:8px;">
            <span>❤️</span> What Viewers Loved Most
        </div>
        <span style="background:#FFFFFF; color:#065F46; font-size:11px; font-weight:700; padding:4px 10px; border-radius:20px; border:1px solid #A7F3D0;">Top Acclaimed: {wvl.get('top_praised_moment', 'Key conversational segments')}</span>
    </div>
    <div style="font-size:13.5px; color:#0F5132; line-height:1.6; margin-bottom:10px;">
        {wvl.get('summary', 'Audience expressed strong appreciation for the guest chemistry, comedic pacing, and unscripted moments.')}
    </div>
    <div style="display:flex; flex-direction:column; gap:6px;">
        {loved_cards}
    </div>
</div>

<!-- CARD 3: CRITICISMS & DEBATES -->
<div style="background:#EDE9FE; border:1px solid #DDD6FE; border-radius:14px; padding:18px 20px; margin-bottom:14px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
        <div style="font-size:15px; font-weight:800; color:#4C1D95; display:flex; align-items:center; gap:8px;">
            <span>⚡</span> Real Viewer Criticisms, Debates & Friction
        </div>
        <span style="background:#FFFFFF; color:#4C1D95; font-size:11px; font-weight:700; padding:4px 10px; border-radius:20px; border:1px solid #C4B5FD;">Core Friction: {cd.get('key_controversy', 'Humorous banter & interpretations')}</span>
    </div>
    <div style="font-size:13.5px; color:#3B0764; line-height:1.6; margin-bottom:10px;">
        {cd.get('summary', 'Viewer debates centered on specific discussion points, banter, and humorous roasting rather than technical flaws.')}
    </div>
    <div style="display:flex; flex-direction:column; gap:6px;">
        {debate_cards}
    </div>
</div>

<!-- CARD 4: STRATEGIC CREATOR PLAYBOOK -->
<div style="background:#FEF3D6; border:1px solid #FDE68A; border-radius:14px; padding:18px 20px; margin-bottom:10px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
        <div style="font-size:15px; font-weight:800; color:#713F12; display:flex; align-items:center; gap:8px;">
            <span>🚀</span> Top Viewer Demands & Strategic Creator Steps
        </div>
        <span style="background:#FFFFFF; color:#713F12; font-size:11px; font-weight:700; padding:4px 10px; border-radius:20px; border:1px solid #FCD34D;">Priority: {vr.get('top_request', 'Part 2 & Uncut Content')}</span>
    </div>
    <div style="font-size:13.5px; color:#78350F; line-height:1.6; margin-bottom:10px;">
        {vr.get('summary', 'Viewers are actively asking for follow-up collaborations and extended uncut footage.')}
    </div>
    <div style="display:flex; flex-direction:column; gap:6px;">
        {action_items_html}
    </div>
</div>
"""
    return md_output
    return md_output
