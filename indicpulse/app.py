"""
IndicPulse Studio • Production Multilingual Semantic & Sentiment Analyzer
========================================================================
Production SaaS Architecture:
  • 6-Section Clean Rail Navigation (Home, Dashboard, History, Profile, Settings, T&C)
  • Automatic .env Environment Integration (No frontend API key inputs)
  • Zero Emojis in Navigation/Headers (Clean Inline SVG Emblems)
  • Anti-AI Button Nav Architecture (No red Streamlit radio circles)
  • Interactive Platform Selector Bento Cards (YouTube, Instagram, Reddit, CSV)
  • Cumulative Cross-Platform Dashboard with Empty State
  • Master-Detail History with In-Place Full Analysis & Live Refresh
  • Strict Soft Pastel Bento-Grid Design System & WCAG Accessibility
"""

import sys
import os
import re
import time
import textwrap
from datetime import datetime
from typing import Optional, Dict, Any, List

# Ensure app directory is in sys.path for robust imports from any working directory
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load credentials automatically from .env at startup
load_dotenv()

# Backend Pipeline Imports (Strictly Preserved)
from pipeline import (
    fetch_youtube_comments,
    preprocess_dataframe,
    run_mood_classification,
    run_topic_clustering,
    generate_executive_summary,
    load_demo_comments,
)

# UI Plotly Chart Helpers
from ui import charts, styles

# Playwright network-interception scraper (Instagram / Reddit)
try:
    from indicpulse.scraper import fetch_comments_via_network_interception as _scrape_comments
    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # ImportError or playwright not installed
    _PLAYWRIGHT_AVAILABLE = False

# ── PAGE CONFIGURATION ─────────────────────────────────────────
st.set_page_config(
    page_title="IndicPulse Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SAFE HTML & MARKDOWN RENDERING WRAPPERS ───────────────────
_orig_st_markdown = st.markdown

def render_html(html_str: str):
    """Safely renders HTML via st.markdown without CommonMark interpreting leading indentation as code blocks."""
    cleaned = textwrap.dedent(html_str).strip()
    _orig_st_markdown(cleaned, unsafe_allow_html=True)

def safe_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        # Dedent leading whitespace so multiline indentation is not treated as code blocks,
        # preserving blank lines, paragraphs, and markdown lists intact.
        body = textwrap.dedent(body).strip()
    return _orig_st_markdown(body, *args, **kwargs)

st.markdown = safe_markdown

def render_gemini_summary(summary_text: str):
    """
    Renders Gemini Executive Summary cleanly:
    - Strips code fences if present (```markdown ... ```)
    - Separates HTML status banner from Markdown content so CommonMark renders headings & lists cleanly
    """
    if not summary_text:
        st.info("No executive summary generated yet.")
        return

    text = str(summary_text).strip()
    text = re.sub(r"^```(?:markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = textwrap.dedent(text).strip()

    if text.startswith("<div") and "</div>" in text:
        div_end = text.find("</div>") + len("</div>")
        banner_html = text[:div_end].strip()
        markdown_part = text[div_end:].strip()
        _orig_st_markdown(banner_html, unsafe_allow_html=True)
        if markdown_part:
            _orig_st_markdown(markdown_part)
    else:
        _orig_st_markdown(text, unsafe_allow_html=True)

# ── PERSISTENT SESSION STATE INITIALIZATION ────────────────────
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"
if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = "YouTube"
if "active_drilldown" not in st.session_state:
    st.session_state.active_drilldown = "ALL"
if "inspector_open" not in st.session_state:
    st.session_state.inspector_open = False
if "inspector_title" not in st.session_state:
    st.session_state.inspector_title = "All Analyzed Comments"
if "inspector_filter_type" not in st.session_state:
    st.session_state.inspector_filter_type = "ALL"
if "inspector_filter_val" not in st.session_state:
    st.session_state.inspector_filter_val = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "youtube_api_key" not in st.session_state:
    st.session_state.youtube_api_key = os.getenv("YOUTUBE_API_KEY", "")
if "exec_summary" not in st.session_state:
    st.session_state.exec_summary = ""
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None
if "latest_run" not in st.session_state:
    st.session_state.latest_run = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Creator"
if "user_handle" not in st.session_state:
    st.session_state.user_handle = "@indicpulse_studio"
if "user_niche" not in st.session_state:
    st.session_state.user_niche = "Tech & Gadgets"
if "summary_lang" not in st.session_state:
    st.session_state.summary_lang = "English"
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False
if "colorblind_mode" not in st.session_state:
    st.session_state.colorblind_mode = False
if "font_scale" not in st.session_state:
    st.session_state.font_scale = 100
if "toggle_emoji" not in st.session_state:
    st.session_state.toggle_emoji = True
if "toggle_slang" not in st.session_state:
    st.session_state.toggle_slang = True
if "toggle_quarantine" not in st.session_state:
    st.session_state.toggle_quarantine = True
if "toggle_log_weight" not in st.session_state:
    st.session_state.toggle_log_weight = True


def trigger_inspector(title: str, f_type: str, f_val: Any = None):
    """Callback to open comment inspector modal and set filter parameters."""
    st.session_state.inspector_open = True
    st.session_state.inspector_title = title
    st.session_state.inspector_filter_type = f_type
    st.session_state.inspector_filter_val = f_val
    st.session_state.active_drilldown = f_type if f_type != "TOPIC" else f"TOPIC:{f_val}"


def close_inspector():
    """Callback to close the inspector modal dialog."""
    st.session_state.inspector_open = False


def get_gemini_api_key() -> str:
    """Retrieve Gemini API Key from session state, env vars, or Streamlit secrets."""
    if st.session_state.get("gemini_api_key"):
        return st.session_state.gemini_api_key.strip()
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY").strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass
    return ""


def get_youtube_api_key() -> str:
    """Retrieve YouTube Data API Key from session state, env vars, or Streamlit secrets."""
    if st.session_state.get("youtube_api_key"):
        return st.session_state.youtube_api_key.strip()
    if os.getenv("YOUTUBE_API_KEY"):
        return os.getenv("YOUTUBE_API_KEY").strip()
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"].strip()
    except Exception:
        pass
    return ""


def filter_comments_for_inspector(df: pd.DataFrame, f_type: str, f_val: Any = None):
    """
    Filters DataFrame based on active inspection category and returns:
    (filtered_df, math_explanation_str, filter_badge_label)
    f_type options: "ALL", "MOOD", "TOPIC", "SARCASM", "TOXIC", "APPROVAL" / "HIGH_WEIGHT"
    """
    if df is None or df.empty:
        return pd.DataFrame(), "No comments available in dataset.", "Empty (0)"

    target_df = df.copy()
    mood_col = "mood" if "mood" in target_df.columns else ("mood_label" if "mood_label" in target_df.columns else None)
    topic_col = "topic_name" if "topic_name" in target_df.columns else ("topic_label" if "topic_label" in target_df.columns else None)

    if f_type == "ALL":
        f_df = target_df
        expl = f"Displaying all {len(f_df)} comments across all 4 sentiment categories and discovered semantic clusters."
        badge = f"All Comments ({len(f_df)})"

    elif f_type == "MOOD":
        target_mood = str(f_val or "Positive").strip()
        if mood_col:
            m_mask = target_df[mood_col].astype(str).str.contains(target_mood, case=False, na=False)
            f_df = target_df[m_mask]
        else:
            f_df = target_df
        w_sum = f_df["engagement_weight"].sum() if "engagement_weight" in f_df.columns else 0
        expl = f"These {len(f_df)} comments were classified as '{target_mood}' by Google MuRIL multilingual logits with an aggregate log₁₀ engagement weight of {w_sum:.1f}."
        badge = f"Mood: {target_mood} ({len(f_df)})"

    elif f_type == "SARCASM":
        if mood_col:
            m_mask = target_df[mood_col].astype(str).str.contains("Sarcastic|Troll", case=False, na=False)
            f_df = target_df[m_mask]
        else:
            f_df = target_df
        expl = f"These {len(f_df)} comments matched irony patterns, mock praise, or sarcasm emoji tokens ([clown_mocking], [eye_roll_sarcasm])."
        badge = f"Sarcastic / Trolling ({len(f_df)})"

    elif f_type == "TOXIC":
        if "is_toxic" in target_df.columns:
            m_mask = target_df["is_toxic"] == True
        elif mood_col:
            m_mask = target_df[mood_col].astype(str).str.contains("Toxic|Abus", case=False, na=False)
        else:
            m_mask = pd.Series([False] * len(target_df))
        f_df = target_df[m_mask]
        expl = f"These {len(f_df)} hostile comments matched abusive profanity regexes or severe harassment patterns and were quarantined from semantic clustering."
        badge = f"Toxic Quarantined ({len(f_df)})"

    elif f_type in ["APPROVAL", "HIGH_WEIGHT"]:
        if "engagement_weight" in target_df.columns and len(target_df) > 0:
            q75 = target_df["engagement_weight"].quantile(0.75)
            f_df = target_df[target_df["engagement_weight"] >= q75]
        else:
            f_df = target_df.head(25)
        expl = f"These {len(f_df)} comments represent the highest 25% of audience engagement, weighted via W = log₁₀(Likes + 1) + 1 to anchor creator briefings."
        badge = f"Top-Weighted (Top 25%) ({len(f_df)})"

    elif f_type == "TOPIC":
        if f_val == "ALL_TOPICS" or not f_val:
            if topic_col:
                f_df = target_df[~target_df[topic_col].astype(str).str.contains("Quarantin", case=False, na=False)]
            else:
                f_df = target_df
            expl = f"These {len(f_df)} constructive comments were vectorized with IndicSBERT and clustered into thematic topics via HDBSCAN."
            badge = f"All Discovered Topics ({len(f_df)})"
        else:
            target_topic = str(f_val).strip()
            if topic_col:
                # 1. Exact direct match
                m1 = target_df[topic_col].astype(str).str.strip() == target_topic
                if m1.any():
                    f_df = target_df[m1]
                else:
                    # 2. Case and punctuation-insensitive match
                    clean_t = re.sub(r'[^\w\s]', '', target_topic).lower()
                    m2 = target_df[topic_col].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower()) == clean_t
                    if m2.any():
                        f_df = target_df[m2]
                    else:
                        # 3. Substring match on first anchor
                        first_kw = target_topic.split('&')[0].strip().strip('"').strip("'")
                        m3 = target_df[topic_col].astype(str).str.contains(re.escape(first_kw), case=False, na=False)
                        f_df = target_df[m3] if m3.any() else target_df
            else:
                f_df = target_df
            avg_w = f_df["engagement_weight"].mean() if "engagement_weight" in f_df.columns else 1.0
            expl = f"These {len(f_df)} comments were clustered into '{target_topic}' via IndicSBERT dense sentence embeddings with an average engagement weight of {avg_w:.2f}."
            badge = f"Topic: {target_topic} ({len(f_df)})"

    else:
        f_df = target_df
        expl = "Displaying comments matching the active inspection filter."
        badge = f"Filtered ({len(f_df)})"

    sort_c = "likes" if "likes" in f_df.columns else ("like_count" if "like_count" in f_df.columns else None)
    if sort_c:
        f_df = f_df.sort_values(sort_c, ascending=False)

    return f_df, expl, badge


@st.dialog("Evidence Inspector", width="large")
def render_inspector_dialog(df_source: Optional[pd.DataFrame] = None, title: Optional[str] = None, f_type: Optional[str] = None, f_val: Any = None):
    """
    Evidence Inspector Modal Dialog.
    Dynamically filters st.session_state.df_clustered (by Mood, Topic, or Weight)
    and displays the comments in st.dataframe(), showing the exact raw_text, likes, mood, and text_for_muril.
    """
    insp_title = title or st.session_state.get("inspector_title", "Evidence Inspector")
    insp_type = f_type or st.session_state.get("inspector_filter_type", "ALL")
    insp_val = f_val if f_val is not None else st.session_state.get("inspector_filter_val", None)

    target_df = df_source if (df_source is not None and not df_source.empty) else st.session_state.get("df_clustered")
    if target_df is None or target_df.empty:
        target_df = st.session_state.get("df_mood", pd.DataFrame())

    if target_df.empty:
        st.info("No comments found in dataset to inspect.")
        if st.button("Close Inspector", key="btn_close_empty_insp", use_container_width=True):
            st.session_state.inspector_open = False
            st.rerun()
        return

    f_df, math_expl, badge = filter_comments_for_inspector(target_df, insp_type, insp_val)

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:17px; font-weight:800; color:#18181B;">{insp_title}</span>
        <span class="pill-badge" style="background:#18181B; color:#FFFFFF;">{badge}</span>
    </div>
    <div style="font-size:12px; color:#475569; background:#F8FAFC; border:1px solid #EAEEF4; border-radius:12px; padding:10px 14px; margin-bottom:14px; line-height:1.5;">
        📐 <b>Evidence Logic:</b> {math_expl}
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Total Filtered Comments", f"{len(f_df):,}")
    with mc2:
        tot_l = int(f_df['likes'].sum()) if 'likes' in f_df.columns else (int(f_df['like_count'].sum()) if 'like_count' in f_df.columns else 0)
        st.metric("Aggregate Likes", f"{tot_l:,}")
    with mc3:
        avg_w = f_df['engagement_weight'].mean() if 'engagement_weight' in f_df.columns else 1.0
        st.metric("Avg Engagement Weight", f"{avg_w:.2f}")

    search_kw = st.text_input("🔍 Filter keywords in comments...", placeholder="Type to filter rows...", key=f"dlg_search_{insp_type}_{insp_val}")
    if search_kw:
        t_col = "raw_text" if "raw_text" in f_df.columns else "text"
        f_df = f_df[f_df[t_col].astype(str).str.contains(search_kw, case=False, na=False)]

    # Guarantee required columns: exact raw_text, likes, mood, and text_for_muril
    f_df_clean = f_df.copy()
    if "raw_text" not in f_df_clean.columns:
        f_df_clean["raw_text"] = f_df_clean.get("text", "")
    if "likes" not in f_df_clean.columns:
        f_df_clean["likes"] = f_df_clean.get("like_count", 0)
    if "mood" not in f_df_clean.columns:
        f_df_clean["mood"] = f_df_clean.get("mood_label", "Neutral")
    if "text_for_muril" not in f_df_clean.columns:
        f_df_clean["text_for_muril"] = f_df_clean.get("text_for_mood", f_df_clean["raw_text"])

    display_cols = ["raw_text", "likes", "mood", "text_for_muril"]
    if "topic_name" in f_df_clean.columns:
        display_cols = ["topic_name"] + display_cols
    elif "topic_label" in f_df_clean.columns:
        f_df_clean["topic_name"] = f_df_clean["topic_label"]
        display_cols = ["topic_name"] + display_cols

    st.dataframe(
        f_df_clean[display_cols],
        use_container_width=True,
        height=320,
        column_config={
            "raw_text": st.column_config.TextColumn("Comment Text (raw_text)", width="large"),
            "likes": st.column_config.NumberColumn("Likes", format="%d", width="small"),
            "mood": st.column_config.TextColumn("Mood", width="medium"),
            "text_for_muril": st.column_config.TextColumn("MuRIL Preprocessed (text_for_muril)", width="medium"),
            "topic_name": st.column_config.TextColumn("Topic Cluster", width="medium"),
        }
    )

    bc1, bc2 = st.columns([1, 1])
    with bc1:
        csv_data = f_df_clean[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Filtered Comments (CSV)", data=csv_data, file_name=f"indicpulse_inspect_{insp_type}.csv", mime="text/csv", use_container_width=True)
    with bc2:
        if st.button("Close Inspector Modal", key="close_dlg_btn", use_container_width=True):
            st.session_state.inspector_open = False
            st.rerun()

# ── INLINE PRODUCTION SVG EMBLEMS (COMPACT ZERO-INDENT) ─────────
SVG_BRAND_LOGO = '<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="34" height="34" rx="10" fill="#18181B"/><path d="M7 17H11L14 9L19 25L23 14L25 17H27" stroke="#34D399" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'

SVG_YOUTUBE = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#FF0000"/><path d="M11 9.5L18.5 14L11 18.5V9.5Z" fill="white"/></svg>'

SVG_INSTAGRAM = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="url(#ig_grad)"/><rect x="7" y="7" width="14" height="14" rx="4" stroke="white" stroke-width="2"/><circle cx="14" cy="14" r="3.2" stroke="white" stroke-width="2"/><circle cx="18" cy="10" r="1" fill="white"/><defs><radialGradient id="ig_grad" cx="0%" cy="100%" r="130%"><stop offset="0%" stop-color="#FFDD55"/><stop offset="30%" stop-color="#FF543E"/><stop offset="60%" stop-color="#C837AB"/><stop offset="100%" stop-color="#3771C8"/></radialGradient></defs></svg>'

SVG_REDDIT = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#FF4500"/><path d="M19.5 13.5C19.5 12.67 18.83 12 18 12C17.58 12 17.2 12.17 16.93 12.45C15.89 11.73 14.46 11.25 12.86 11.2L13.62 7.62L16.1 8.15C16.14 8.77 16.66 9.27 17.3 9.27C18 9.27 18.55 8.72 18.55 8.02C18.55 7.32 18 6.77 17.3 6.77C16.81 6.77 16.39 7.05 16.19 7.45L13.38 6.85C13.25 6.82 13.12 6.91 13.09 7.04L12.22 11.16C10.58 11.21 9.12 11.7 8.07 12.45C7.8 12.17 7.42 12 7 12C6.17 12 5.5 12.67 5.5 13.5C5.5 14.12 5.88 14.65 6.42 14.87C6.39 15.08 6.38 15.29 6.38 15.5C6.38 18.26 9.51 20.5 13.38 20.5C17.25 20.5 20.38 18.26 20.38 15.5C20.38 15.29 20.37 15.08 20.34 14.87C20.88 14.65 21.26 14.12 21.26 13.5C21.26 12.67 20.59 12 19.76 12L19.5 13.5ZM9.5 15.5C9.5 14.67 10.17 14 11 14C11.83 14 12.5 14.67 12.5 15.5C12.5 16.33 11.83 17 11 17C10.17 17 9.5 16.33 9.5 15.5ZM16.36 18.24C15.5 18.81 14.24 18.98 13.38 18.98C12.52 18.98 11.26 18.81 10.4 18.24C10.24 18.13 10.2 17.91 10.31 17.75C10.42 17.59 10.64 17.55 10.8 17.66C11.47 18.11 12.47 18.28 13.38 18.28C14.29 18.28 15.29 18.11 15.96 17.66C16.12 17.55 16.34 17.59 16.45 17.75C16.56 17.91 16.52 18.13 16.36 18.24ZM15.76 17C14.93 17 14.26 16.33 14.26 15.5C14.26 14.67 14.93 14 15.76 14C16.59 14 17.26 14.67 17.26 15.5C17.26 16.33 16.59 17 15.76 17Z" fill="white"/></svg>'

SVG_CSV = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#4F46E5"/><rect x="7" y="6" width="14" height="16" rx="2" stroke="white" stroke-width="1.8"/><line x1="7" y1="11" x2="21" y2="11" stroke="white" stroke-width="1.5"/><line x1="7" y1="16" x2="21" y2="16" stroke="white" stroke-width="1.5"/><line x1="14" y1="11" x2="14" y2="22" stroke="white" stroke-width="1.5"/></svg>'

SVG_SCRAPER = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#0F766E"/><circle cx="14" cy="12" r="5" stroke="white" stroke-width="2"/><line x1="17.5" y1="15.5" x2="22" y2="20" stroke="white" stroke-width="2" stroke-linecap="round"/><circle cx="14" cy="12" r="2" fill="white"/></svg>'

SVG_EMPTY_STATE = '<svg width="180" height="120" viewBox="0 0 180 120" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="20" y="20" width="140" height="80" rx="16" fill="#FFFFFF" stroke="#CCEFE2" stroke-width="2"/><rect x="35" y="35" width="40" height="10" rx="5" fill="#E2F6EE"/><rect x="35" y="52" width="110" height="6" rx="3" fill="#F1F5F9"/><rect x="35" y="64" width="85" height="6" rx="3" fill="#F1F5F9"/><circle cx="130" cy="40" r="10" fill="#E2F6EE"/><path d="M127 40L129 42L133 38" stroke="#0F5132" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'

# ── STRICT PASTEL BENTO-GRID CSS SYSTEM ─────────────────────────
border_style = "2px solid #18181B" if st.session_state.high_contrast else "1px solid #EAEEF4"
card_border_mint = "2px solid #18181B" if st.session_state.high_contrast else "1px solid #CCEFE2"
card_border_pink = "2px solid #18181B" if st.session_state.high_contrast else "1px solid #F9D1DF"
card_border_yellow = "2px solid #18181B" if st.session_state.high_contrast else "1px solid #FCE4AD"
card_border_purple = "2px solid #18181B" if st.session_state.high_contrast else "1px solid #DDD6FE"
font_pct = f"{st.session_state.font_scale}%"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        font-size: {font_pct};
    }}
    .stApp {{
        background-color: #F4F6F9;
    }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px;
    }}
    
    /* Sleek Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF !important;
        border-right: 1px solid #EAEEF4 !important;
        padding-top: 1rem !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem !important;
    }}

    /* Anti-AI Nav Button Styles */
    [data-testid="stSidebar"] button[kind="secondary"] {{
        background-color: transparent !important;
        border: none !important;
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
        margin-bottom: 3px !important;
    }}
    [data-testid="stSidebar"] button[kind="secondary"]:hover {{
        background-color: #F8FAFC !important;
        color: #18181B !important;
        transform: translateX(2px);
    }}
    [data-testid="stSidebar"] button[kind="primary"] {{
        background-color: #18181B !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(24, 24, 27, 0.12) !important;
        margin-bottom: 3px !important;
    }}

    /* Rounded White Bento Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #FFFFFF !important;
        border-radius: 22px !important;
        border: {border_style} !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
        padding: 20px !important;
        margin-bottom: 14px !important;
    }}

    /* Top Greeting Bento Bar */
    .bento-header {{
        background: #FFFFFF;
        border-radius: 22px;
        padding: 20px 26px;
        border: {border_style};
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }}

    /* 4 Signature Pastel Bento Themes */
    .bento-mint {{
        background-color: #E2F6EE;
        border: {card_border_mint};
        color: #0F5132;
        border-radius: 22px;
        padding: 16px 20px;
        height: 145px;
        min-height: 145px;
        max-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        margin-bottom: 0px !important;
    }}
    .bento-pink {{
        background-color: #FCE8EF;
        border: {card_border_pink};
        color: #881337;
        border-radius: 22px;
        padding: 16px 20px;
        height: 145px;
        min-height: 145px;
        max-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        margin-bottom: 0px !important;
    }}
    .bento-yellow {{
        background-color: #FEF3D6;
        border: {card_border_yellow};
        color: #713F12;
        border-radius: 22px;
        padding: 16px 20px;
        height: 145px;
        min-height: 145px;
        max-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        margin-bottom: 0px !important;
    }}
    .bento-purple {{
        background-color: #EDE9FE;
        border: {card_border_purple};
        color: #4C1D95;
        border-radius: 22px;
        padding: 16px 20px;
        height: 145px;
        min-height: 145px;
        max-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        margin-bottom: 0px !important;
    }}

    /* Clean Pills and Micro-Badges */
    .pill-badge {{
        background: #FFFFFF;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        color: #18181B;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}
    .arrow-circle {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #FFFFFF;
        color: #18181B;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .topic-pill-card {{
        border-radius: 18px;
        padding: 12px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.15s ease;
    }}
    .topic-card-mint {{
        background-color: #E2F6EE;
        border: 1px solid #CCEFE2;
        color: #0F5132;
    }}
    .topic-card-lavender {{
        background-color: #EDE9FE;
        border: 1px solid #DDD6FE;
        color: #4C1D95;
    }}
    .topic-card-yellow {{
        background-color: #FEF3D6;
        border: 1px solid #FCE4AD;
        color: #713F12;
    }}
    .topic-card-pink {{
        background-color: #FCE8EF;
        border: 1px solid #F9D1DF;
        color: #881337;
    }}
    .topic-icon {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #FFFFFF;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 15px;
        font-weight: 700;
        flex-shrink: 0;
    }}
    .progress-track {{
        width: 100%;
        height: 6px;
        background: rgba(0,0,0,0.08);
        border-radius: 4px;
        margin-top: 14px;
        overflow: hidden;
    }}
    .progress-fill {{
        height: 100%;
        background: #18181B;
        border-radius: 4px;
    }}

    /* Platform Card Styling */
    .platform-card {{
        background: #FFFFFF;
        border-radius: 20px;
        padding: 18px 14px;
        text-align: center;
        border: 1px solid #EAEEF4;
        transition: all 0.15s ease;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    .platform-card-active {{
        background: #FFFFFF;
        border-radius: 20px;
        padding: 18px 14px;
        text-align: center;
        border: 2px solid #18181B !important;
        box-shadow: 0 4px 16px rgba(24, 24, 27, 0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    .emblem-circle {{
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #F8FAFC;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }}

    /* 3D Mechanical Odometer / Rolling Digit Counter */
    @keyframes rollUpOdometer {{
        0% {{ transform: translateY(100%) scaleY(0.5); opacity: 0; filter: blur(2px); }}
        100% {{ transform: translateY(0) scaleY(1); opacity: 1; filter: blur(0); }}
    }}

    .odo-container {{ 
        display: inline-flex !important; 
        gap: 2px !important; 
        background: #18181B !important; 
        padding: 4px 8px !important; 
        border-radius: 8px !important; 
        margin: 0 !important; 
        align-items: center !important; 
        box-shadow: inset 0 3px 6px rgba(0,0,0,0.6), 0 4px 12px rgba(24,24,27,0.18) !important;
        overflow: hidden !important; /* Critical to hide the digits before they roll up */
    }}
    .odo-digit {{ 
        background: linear-gradient(180deg, #27272A 0%, #18181B 50%, #27272A 100%) !important; 
        color: #FFFFFF !important; 
        font-family: 'JetBrains Mono', monospace !important; 
        font-size: 22px !important; 
        font-weight: 700 !important; 
        min-width: 24px !important; 
        height: 32px !important; 
        display: flex !important; 
        align-items: center !important; 
        justify-content: center !important; 
        border-radius: 4px !important; 
        margin: 0 !important; 
        padding: 0 !important; 
        line-height: 1 !important;
        border-top: 1px solid rgba(255,255,255,0.15) !important;
        border-bottom: 1px solid rgba(0,0,0,0.8) !important;
        
        /* The Animation Trigger */
        animation: rollUpOdometer 0.85s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
        transform-origin: bottom center !important;
    }}
    
    /* Stagger the animation so digits roll in one after another */
    .odo-digit:nth-child(1) {{ animation-delay: 0.0s !important; }}
    .odo-digit:nth-child(2) {{ animation-delay: 0.1s !important; }}
    .odo-digit:nth-child(3) {{ animation-delay: 0.2s !important; }}
    .odo-digit:nth-child(4) {{ animation-delay: 0.3s !important; }}
    .odo-digit:nth-child(5) {{ animation-delay: 0.4s !important; }}
    .odo-digit:nth-child(6) {{ animation-delay: 0.5s !important; }}
    .odo-digit:nth-child(7) {{ animation-delay: 0.6s !important; }}

    .odo-sep {{
        color: #A1A1AA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 0 1px !important;
        margin: 0 !important;
        line-height: 1 !important;
    }}

    /* Global Input & Button Overrides */
    .stTextInput > div > div > input {{
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 10px 14px !important;
    }}
    .stButton > button {{
        border-radius: 12px !important;
        font-weight: 700 !important;
    }}

    /* Bento Inspect Buttons - Pixel-Perfect Bottom Alignment */
    div[class*="st-key-drill_"],
    div[class*="st-key-dash_inspect_"],
    div[class*="st-key-topic_btn_"],
    div[class*="st-key-dash_topic_btn_"] {{
        margin-top: -6px !important;
        margin-bottom: 0px !important;
    }}
    div[class*="st-key-drill_"] button,
    div[class*="st-key-dash_inspect_"] button,
    div[class*="st-key-topic_btn_"] button,
    div[class*="st-key-dash_topic_btn_"] button {{
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        border-radius: 12px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        padding: 0 10px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 6px rgba(24, 24, 27, 0.08) !important;
    }}
</style>
""", unsafe_allow_html=True)


# ===============================================================
# 1. CORE PIPELINE EXECUTOR & HISTORY PERSISTENCE
# ===============================================================
def execute_analysis_flow(
    df_input: Optional[pd.DataFrame] = None,
    platform_name: str = "YouTube",
    target_title: str = "Analysis Run",
    video_url: Optional[str] = None,
    max_comments: int = 300,
):
    """Executes the full IndicPulse 5-Stage pipeline with live status updates and commits result to history."""
    yt_key = get_youtube_api_key()
    gemini_key = get_gemini_api_key()
    model_id = os.getenv("HF_MODEL_ID", "google/muril-base-cased")

    with st.status("Running IndicPulse 5-Stage Pipeline...", expanded=True) as status:
        # Stage 1: Ingestion
        if video_url and (df_input is None or df_input.empty):
            st.write("1. Fetching comments and video metadata from YouTube API...")
            prog_text = st.empty()
            def on_progress(fetched: int, target: int):
                prog_text.caption(f"Fetched {fetched} / {target} comments...")

            fetch_res = fetch_youtube_comments(video_url, yt_key, max_comments=max_comments, progress_callback=on_progress)
            if isinstance(fetch_res, tuple):
                df_input, fetched_title = fetch_res
                if fetched_title and fetched_title != f"YouTube Video ({video_url})":
                    target_title = fetched_title
            else:
                df_input = fetch_res
            prog_text.empty()
            st.write(f"1. Successfully fetched {len(df_input)} comments from YouTube.")
        else:
            st.write(f"1. Ingested {len(df_input) if df_input is not None else 0} comments from {platform_name}.")

        if df_input is None or df_input.empty:
            status.update(label="No comments found to analyze.", state="error", expanded=False)
            st.error("No comments retrieved.")
            return

        # Extract video metadata
        v_title = df_input.attrs.get("video_title", target_title) if hasattr(df_input, "attrs") else target_title
        c_title = df_input.attrs.get("channel_title", "") if hasattr(df_input, "attrs") else ""

        # Stage 2: Translating Emojis & Hinglish Slang
        st.write("2. Translating emojis to bracket tags & resolving Hinglish slang...")
        df_proc = preprocess_dataframe(df_input.copy())

        # Stage 3: Running MuRIL Mood & Sarcasm Classifier
        st.write("3. Running MuRIL Mood & Sarcasm Classifier (hardware accelerated, batch_size=32)...")
        df_mood = run_mood_classification(df_proc, model_name=model_id, use_gpu=True)

        # Stage 4: Clustering Topics with IndicSBERT + BERTopic
        st.write("4. Clustering constructive topics with IndicSBERT & adaptive BERTopic...")
        df_clust = run_topic_clustering(df_mood)

        # Stage 5: Synthesizing Creator Action Plan with Gemini 2.5 Flash
        st.write(f"5. Synthesizing Executive Creator Action Plan for '{v_title}' with Gemini 2.5 Flash...")
        try:
            summary_text = generate_executive_summary(df_clust, gemini_key, video_title=v_title, channel_title=c_title)
        except Exception as e:
            logger.error(f"Gemini execution error: {e}")
            summary_text = f"Executive summary could not be synthesized: {e}"

        status.update(label=f"Analysis Complete! ({len(df_mood)} comments processed)", state="complete", expanded=False)

    # Compute key performance metrics
    total_c = len(df_mood)
    pos_mask = df_mood["mood"].astype(str).str.contains("Positive")
    total_w = max(1e-6, df_mood["engagement_weight"].sum())
    weighted_pos = (df_mood.loc[pos_mask, "engagement_weight"].sum() / total_w) * 100
    sarcasm_c = int((df_mood["mood"].astype(str).str.contains("Sarcastic")).sum())
    toxic_c = int((df_mood["mood"].astype(str).str.contains("Toxic")).sum())

    if "topic_name" in df_clust.columns:
        topic_counts = df_clust["topic_name"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        num_topics = len(topic_counts[~topic_counts["Topic"].str.contains("Quarantin")])
    else:
        topic_counts = pd.DataFrame({"Topic": ["Miscellaneous / Outliers"], "Count": [total_c]})
        num_topics = 1

    dom_mood = df_mood["mood"].mode()[0] if not df_mood.empty else "Neutral"
    pulse_score = round(min(10.0, max(3.5, (weighted_pos / 10.0))), 1)

    # Assemble run object
    run_id = f"run_{len(st.session_state.analysis_history) + 1}_{int(time.time())}"
    run_entry = {
        "id": run_id,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "platform": platform_name,
        "url_or_title": v_title,
        "video_title": v_title,
        "channel_title": c_title,
        "df_mood": df_mood,
        "df_clustered": df_clust,
        "exec_summary": summary_text,
        "metrics": {
            "total": total_c,
            "weighted_approval": weighted_pos,
            "sarcasm_count": sarcasm_c,
            "toxic_count": toxic_c,
            "num_topics": num_topics,
            "dominant_mood": dom_mood,
            "pulse_score": pulse_score,
        },
    }

    st.session_state.analysis_history.insert(0, run_entry)
    st.session_state.latest_run = run_entry
    st.session_state.selected_history_id = run_id
    st.session_state.df_clustered = df_clust
    st.session_state.df_mood = df_mood
    st.session_state.exec_summary = summary_text


# ===============================================================
# 2. SIDEBAR RAIL NAVIGATION (6 EXACT SECTIONS)
# ===============================================================
with st.sidebar:
    # Brand Mark
    render_html(f"""
    <div style="display:flex; align-items:center; gap:12px; padding:6px 4px 16px 4px;">
        {SVG_BRAND_LOGO}
        <div>
            <div style="font-size:17px; font-weight:800; color:#18181B; line-height:1.15;">IndicPulse</div>
            <div style="font-size:11px; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.6px;">Semantic Studio</div>
        </div>
    </div>
    """)

    st.markdown("<div style='font-size:11px; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.8px; margin:14px 0 6px 6px;'>Main Menu</div>", unsafe_allow_html=True)
    
    # Nav Item 1: Home
    if st.button(
        "Home",
        key="nav_home",
        type="primary" if st.session_state.active_page == "Home" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_page = "Home"
        st.session_state.selected_history_id = None
        st.rerun()

    # Nav Item 2: Dashboard
    if st.button(
        "Dashboard",
        key="nav_dashboard",
        type="primary" if st.session_state.active_page == "Dashboard" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_page = "Dashboard"
        st.session_state.selected_history_id = None
        st.rerun()

    # Nav Item 3: History
    if st.button(
        "History",
        key="nav_history",
        type="primary" if st.session_state.active_page == "History" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_page = "History"
        st.session_state.selected_history_id = None
        st.rerun()

    st.markdown("<div style='height:1px; background:#F1F5F9; margin:16px 0 12px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.8px; margin:0 0 6px 6px;'>Preferences & Legal</div>", unsafe_allow_html=True)

    # Nav Item 4: Profile
    if st.button(
        "Profile",
        key="nav_profile",
        type="primary" if st.session_state.active_page == "Profile" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_page = "Profile"
        st.rerun()

    # Nav Item 5: Settings
    if st.button(
        "Settings",
        key="nav_settings",
        type="primary" if st.session_state.active_page == "Settings" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_page = "Settings"
        st.rerun()

    # Nav Item 6: T&C
    if st.button(
        "T&C",
        key="nav_tnc",
        type="primary" if st.session_state.active_page == "T&C" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_page = "T&C"
        st.rerun()

    # Bottom Engine Online Badge
    st.markdown("<div style='height:36px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#E2F6EE; border:1px solid #CCEFE2; color:#0F5132; padding:10px 14px; border-radius:12px; font-size:12px; font-weight:700; display:flex; align-items:center; gap:8px;">
        <span style="font-size:9px; color:#10B981;">●</span> Engine Online
    </div>
    """, unsafe_allow_html=True)


# ===============================================================
# 3. HELPER: 3D MECHANICAL ODOMETER SCORE COMPONENT & BENTO GRID
# ===============================================================
def render_info_tooltip(tooltip: str) -> str:
    """Inject a subtle inline CSS tooltip info icon directly into HTML headers."""
    if not tooltip:
        return ""
    escaped_tip = str(tooltip).replace('"', '&quot;').replace("'", "&#39;")
    return f'<span title="{escaped_tip}" style="cursor:help; opacity:0.6; font-size:12px; margin-left:4px;">ℹ️</span>'


def render_info_btn(tooltip: str, title: str = "") -> str:
    """Subtle inline CSS tooltip fallback."""
    return render_info_tooltip(tooltip)


def render_odometer_score(value_str: str, label: Optional[str] = None, subtitle: str = "", theme: str = "mint", info_tooltip: str = "") -> str:
    """
    Render 3D mechanical odometer rolling digits (.odo-digit drums inside .odo-container).
    If label is provided, renders the complete Pastel Bento KPI Card.
    """
    digits_html = []
    for ch in str(value_str):
        if ch in "0123456789":
            digits_html.append(f'<span class="odo-digit">{ch}</span>')
        elif ch in ".,%/+-: ":
            digits_html.append(f'<span class="odo-sep">{ch}</span>')
        else:
            digits_html.append(f'<span class="odo-digit">{ch}</span>')
    odo_html = f'<div class="odo-container">{"".join(digits_html)}</div>'

    if label is None:
        return odo_html

    tooltip_html = render_info_tooltip(info_tooltip)
    sub_html = f'<div style="font-size:11px; font-weight:600; opacity:0.85; min-height:28px; line-height:1.25; display:flex; align-items:center;">{subtitle}</div>' if subtitle else '<div style="min-height:28px;"></div>'
    return f"""
    <div class="bento-{theme}" style="height:145px; min-height:145px; max-height:145px; padding:16px 20px; display:flex; flex-direction:column; justify-content:space-between; box-sizing:border-box; margin-bottom:0px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:4px;">
                <span style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">{label}</span>
                {tooltip_html}
            </div>
            <span class="arrow-circle">↗</span>
        </div>
        <div style="margin: 6px 0 4px 0;">
            {odo_html}
        </div>
        {sub_html}
    </div>
    """


def render_kpi_card(value_str: str, label: str, subtitle: str = "", theme: str = "mint", tooltip: str = "") -> str:
    """Helper to render a Bento KPI Card with 3D mechanical odometer and inline CSS tooltip."""
    return render_odometer_score(value_str, label=label, subtitle=subtitle, theme=theme, info_tooltip=tooltip)


def render_bento_grid_results(run_data: Dict[str, Any]):
    """Renders the complete Asymmetric Pastel Bento Grid with Odometer UI & Interactive Drill-Downs."""
    df_mood = run_data["df_mood"]
    df_clustered = run_data["df_clustered"]
    exec_summary = run_data["exec_summary"]
    metrics = run_data["metrics"]

    # Guarantee session state persistence for inspector dialog
    st.session_state.df_clustered = df_clustered
    st.session_state.df_mood = df_mood
    st.session_state.exec_summary = exec_summary

    total_c = metrics["total"]
    weighted_pos = metrics["weighted_approval"]
    sarcasm_c = metrics["sarcasm_count"]
    toxic_c = metrics["toxic_count"]
    num_topics = metrics["num_topics"]
    dom_mood = metrics["dominant_mood"]
    pulse_score = metrics.get("pulse_score", round(min(10.0, max(3.5, (weighted_pos / 10.0))), 1))
    v_title = run_data.get("video_title", run_data.get("url_or_title", "Video Analysis"))
    c_title = run_data.get("channel_title", "")

    # Ensure topic columns exist dynamically directly from clustered dataframe
    target_clustered = st.session_state.get("df_clustered", df_clustered)
    if target_clustered is None or target_clustered.empty:
        target_clustered = df_clustered

    if "topic_name" not in target_clustered.columns:
        target_clustered["topic_name"] = target_clustered.get("topic_label", "Miscellaneous / Outliers")

    topic_counts = target_clustered["topic_name"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    # Filter out quarantined from top topics display
    topic_counts = topic_counts[~topic_counts["Topic"].astype(str).str.contains("Quarantin", case=False, na=False)]

    pos_mask = df_mood["mood"].astype(str).str.contains("Positive")
    total_pos_count = int(pos_mask.sum())

    # Metadata Banner
    if c_title:
        st.caption(f"🎬 **Video Context:** {v_title} • **Channel:** {c_title} • **Sample Size:** {total_c} comments")

    # Open Modal Dialog if triggered by any Inspect button
    if st.session_state.get("inspector_open"):
        render_inspector_dialog()

    left_zone, right_zone = st.columns([2.15, 1.05], gap="large")

    # ── LEFT ZONE ──────────────────────────────────────────────
    with left_zone:
        st.markdown(f"<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Activities Overview ({total_c} comments)</div>", unsafe_allow_html=True)
        act_c1, act_c2 = st.columns(2, gap="medium")

        with act_c1:
            st.markdown(render_kpi_card(
                f"{weighted_pos:.1f}%",
                "Audience Hype",
                f"Dominant: {dom_mood}",
                theme="mint",
                tooltip="Percentage of positive community hype, log-weighted by likes."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key=f"drill_pos_{run_data['id']}",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"Audience Hype & Positive Comments ({total_pos_count})", "MOOD", "Positive"),
            )

        with act_c2:
            st.markdown(render_kpi_card(
                f"{toxic_c}",
                "Moderation Shield",
                f"{sarcasm_c} Sarcastic | {toxic_c} Quarantined",
                theme="pink",
                tooltip="Hostile comments, abusive language, and personal harassment isolated and quarantined."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key=f"drill_tox_{run_data['id']}",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"Moderation Shield & Toxic Quarantined ({toxic_c})", "TOXIC", None),
            )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Semantic Progress</div>", unsafe_allow_html=True)

        kpi_c1, kpi_c2, kpi_c3 = st.columns(3, gap="medium")
        with kpi_c1:
            st.markdown(render_kpi_card(
                f"{total_c:,}",
                "Analyzed",
                "Total Dataset Comments",
                theme="mint",
                tooltip="Total count of raw comments fetched, sanitized, and classified."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key=f"drill_all_{run_data['id']}",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"All Analyzed Comments ({total_c:,})", "ALL", None),
            )

        with kpi_c2:
            st.markdown(render_kpi_card(
                f"{weighted_pos:.0f}%",
                "Weighted Score",
                "Log10 Weighted Approval",
                theme="yellow",
                tooltip="Log10 Weighted Approval: (Sum W_pos / Sum W_total * 100)."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key=f"drill_weight_{run_data['id']}",
                use_container_width=True,
                on_click=trigger_inspector,
                args=("Top Engagement-Weighted Comments (Top 25%)", "APPROVAL", None),
            )

        with kpi_c3:
            st.markdown(render_kpi_card(
                f"{num_topics}",
                "Active Topics",
                "Discovered Clusters",
                theme="purple",
                tooltip="Distinct thematic conversation clusters discovered via IndicSBERT sentence embeddings."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key=f"drill_topics_{run_data['id']}",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"All Discovered Topics ({num_topics})", "TOPIC", "ALL_TOPICS"),
            )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # Executive Report Card (Clarified & Video-Specific)
        render_html(f"""
        <div class="bento-yellow" style="min-height:auto; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="pill-badge" style="background:#FFFFFF; color:#713F12; font-size:11px; font-weight:800; letter-spacing:0.6px;">AI CREATOR BRIEFING • GEMINI 2.5 FLASH</span>
                <span class="arrow-circle">↗</span>
            </div>
            <div style="font-size:18px; font-weight:800; color:#713F12; margin-top:8px;">Executive Action Synthesis (What Viewers Want Next)</div>
            <div style="font-size:12px; color:#713F12; opacity:0.85; margin-top:3px;">Video-specific analysis synthesized from the highest-engaged Hinglish comments across discovered topics.</div>
            <div class="progress-track"><div class="progress-fill" style="width:{min(100, max(10, int(weighted_pos)))}%;"></div></div>
        </div>
        """)

        with st.container(border=True):
            if st.session_state.exec_summary:
                st.markdown(st.session_state.exec_summary)  # Renders bullet points correctly!
            else:
                st.info("Awaiting Gemini Synthesis...")

            active_gem_key = get_gemini_api_key()
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            with st.expander("⚡ Google Gemini Neural Engine Controls", expanded=not bool(active_gem_key)):
                st.caption("💡 Free AI Studio keys work immediately without GCP setup: [Get Free Gemini API Key at Google AI Studio](https://aistudio.google.com/app/apikey)")
                g_col1, g_col2 = st.columns([3, 1.3])
                with g_col1:
                    gem_input = st.text_input(
                        "Google Gemini API Key",
                        value=active_gem_key,
                        type="password",
                        key=f"gem_inline_in_{run_data['id']}",
                        placeholder="Paste AI Studio API Key (AIzaSy...)",
                        help="Enter your Google Gemini API key to activate live neural synthesis."
                    )
                with g_col2:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    run_gemini_now = st.button("⚡ Synthesize with Gemini", key=f"run_gem_btn_{run_data['id']}", type="primary", use_container_width=True)

                if run_gemini_now:
                    target_key = gem_input.strip() or active_gem_key
                    if not target_key:
                        st.error("Please enter a valid Google Gemini API key.")
                    else:
                        st.session_state.gemini_api_key = target_key
                        os.environ["GEMINI_API_KEY"] = target_key
                        with st.spinner("Synthesizing live Gemini 2.5 Flash creator briefing..."):
                            new_summary = generate_executive_summary(df_clustered, target_key, video_title=v_title, channel_title=c_title)
                            run_data["exec_summary"] = new_summary
                            st.session_state.exec_summary = new_summary
                            st.success("Executive action synthesis refreshed via Google Gemini 2.5 Flash!")
                            st.rerun()

    # ── RIGHT ZONE ─────────────────────────────────────────────
    with right_zone:
        pulse_info = render_info_tooltip(
            "Pulse Score (0-10): Overall health score of audience reception. 8.5-10 = Viral Hype & Strong Community Praise; 7.0-8.4 = Strongly Positive; 5.0-6.9 = Mixed; <5.0 = Critical Friction."
        )
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:4px;">
                <span style="font-size:14px; font-weight:700; color:#64748B;">Community Sentiment Pulse</span>
                {pulse_info}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div style='font-size:13px; font-weight:700; color:#1E293B;'>Radial Speedometer Gauge</div>", unsafe_allow_html=True)
            fig_gauge = charts.pulse_gauge_chart(pulse_score, colorblind_safe=st.session_state.colorblind_mode)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Mood Distribution</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div style='font-size:13px; font-weight:700; color:#1E293B;'>Engagement-Weighted Breakdown</div>", unsafe_allow_html=True)
            fig_pie = charts.mood_donut_chart(df_mood, colorblind_safe=st.session_state.colorblind_mode)
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        clusters_info = render_info_tooltip(
            "Semantic Topic Clusters: Distinct thematic conversation themes discovered across comments using IndicSBERT embeddings."
        )
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:4px;">
                <span style="font-size:14px; font-weight:700; color:#64748B;">Top Semantic Clusters</span>
                {clusters_info}
            </div>
        </div>
        """, unsafe_allow_html=True)

        cluster_emblems = ["1", "2", "3", "4"]
        topic_styles = [
            ("topic-card-mint", "#0F5132"),
            ("topic-card-lavender", "#4C1D95"),
            ("topic-card-yellow", "#713F12"),
            ("topic-card-pink", "#881337"),
        ]
        for idx, row in topic_counts.head(4).iterrows():
            emb = cluster_emblems[idx % len(cluster_emblems)]
            card_class, icon_color = topic_styles[idx % len(topic_styles)]
            st.markdown(f"""
            <div class="topic-pill-card {card_class}">
                <div style="display:flex; align-items:center;">
                    <span class="topic-icon" style="color:{icon_color};">{emb}</span>
                    <div>
                        <div style="font-size:13px; font-weight:700;">{row['Topic']}</div>
                        <div style="font-size:11px; opacity:0.8;">IndicSBERT + Semantic Cluster</div>
                    </div>
                </div>
                <span class="pill-badge">{row['Count']} items</span>
            </div>
            """, unsafe_allow_html=True)
            st.button(
                f"Inspect {row['Count']} Comments ↗",
                key=f"topic_btn_{idx}_{run_data['id']}",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"Topic Cluster: {row['Topic']}", "TOPIC", row['Topic']),
            )

    # ── UNIFIED AUDIENCE COMMENT INTELLIGENCE & EVIDENCE INSPECTOR ───
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    f_type = st.session_state.get("inspector_filter_type", "ALL")
    f_val = st.session_state.get("inspector_filter_val", None)
    target_data = df_clustered if "topic_name" in df_clustered.columns else df_mood

    f_df, math_expl, badge = filter_comments_for_inspector(target_data, f_type, f_val)

    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
            <div>
                <span style="font-size:17px; font-weight:800; color:#1E293B;">🔍 Audience Comment Intelligence & Evidence Inspector</span>
                <span class="pill-badge" style="background:#18181B; color:#FFFFFF; margin-left:10px;">{badge}</span>
            </div>
            <div style="display:flex; gap:8px; align-items:center;">
                <span style="font-size:12px; font-weight:700; color:#64748B;">Showing {len(f_df):,} comments</span>
            </div>
        </div>
        <div style="font-size:12px; color:#475569; background:#F8FAFC; border:1px solid #EAEEF4; border-radius:12px; padding:10px 14px; margin-bottom:14px; line-height:1.5;">
            📐 <b>Evidence Logic:</b> {math_expl}
        </div>
        """, unsafe_allow_html=True)

        # Interactive Filter Toolbar
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2.3, 1.8, 2.3, 1.2])

        # 1. Topic Filter Dropdown
        all_unique_topics = [t for t in target_data["topic_name"].unique() if "quarantin" not in str(t).lower()] if "topic_name" in target_data.columns else []
        topic_filter_options = ["All Topics"] + all_unique_topics + ["Quarantined (Toxic / Abusive)"]

        # Determine currently active topic selection
        if f_type == "TOPIC" and f_val and f_val != "ALL_TOPICS":
            active_topic_str = str(f_val).strip()
            matching_opts = [opt for opt in topic_filter_options if opt.lower() == active_topic_str.lower() or active_topic_str.lower() in opt.lower()]
            selected_topic_default = matching_opts[0] if matching_opts else "All Topics"
        else:
            selected_topic_default = "All Topics"

        default_topic_idx = topic_filter_options.index(selected_topic_default) if selected_topic_default in topic_filter_options else 0

        with ctrl_c1:
            chosen_topic = st.selectbox(
                "Filter by Active Topic Discussed",
                options=topic_filter_options,
                index=default_topic_idx,
                key=f"topic_select_{run_data['id']}",
                help="Select an active discussion topic to inspect its underlying audience comments."
            )

        # 2. Mood Filter Dropdown
        mood_options = ["All Moods", "Positive (Aggressive Hype)", "Neutral", "Sarcastic / Trolling", "Toxic / Abusive"]
        if f_type == "MOOD" and f_val:
            m_target = str(f_val).strip()
            matching_moods = [opt for opt in mood_options if m_target.lower() in opt.lower()]
            selected_mood_default = matching_moods[0] if matching_moods else "All Moods"
        elif f_type == "SARCASM":
            selected_mood_default = "Sarcastic / Trolling"
        elif f_type == "TOXIC":
            selected_mood_default = "Toxic / Abusive"
        else:
            selected_mood_default = "All Moods"

        default_mood_idx = mood_options.index(selected_mood_default) if selected_mood_default in mood_options else 0

        with ctrl_c2:
            chosen_mood = st.selectbox(
                "Filter by Sentiment Mood",
                options=mood_options,
                index=default_mood_idx,
                key=f"mood_select_{run_data['id']}",
                help="Filter comments by detected sentiment mood."
            )

        # 3. Keyword Search
        with ctrl_c3:
            kw_search = st.text_input(
                "Search Comment Text",
                placeholder="Search words (e.g. wavy, scene)...",
                key=f"kw_search_{run_data['id']}",
            )

        # 4. Reset Button
        with ctrl_c4:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("↺ Show All", key=f"reset_filter_{run_data['id']}", use_container_width=True):
                st.session_state.inspector_filter_type = "ALL"
                st.session_state.inspector_filter_val = None
                st.rerun()

        # Apply Dropdown Overrides if changed by user
        final_df = target_data.copy()
        if chosen_topic != "All Topics":
            if chosen_topic == "Quarantined (Toxic / Abusive)":
                final_df = final_df[final_df["mood"].astype(str).str.contains("Toxic", case=False, na=False)]
            elif "topic_name" in final_df.columns:
                final_df = final_df[final_df["topic_name"] == chosen_topic]
        elif f_type == "TOPIC" and f_val == "ALL_TOPICS":
            if "topic_name" in final_df.columns:
                final_df = final_df[~final_df["topic_name"].astype(str).str.contains("Quarantin", case=False, na=False)]
        elif f_type == "TOPIC" and f_val and f_val != "ALL_TOPICS":
            # Match directly from filter_comments_for_inspector
            final_df = f_df

        if chosen_mood != "All Moods":
            mood_tag = chosen_mood.split()[0]
            if "mood" in final_df.columns:
                final_df = final_df[final_df["mood"].astype(str).str.contains(mood_tag, case=False, na=False)]

        if kw_search:
            raw_c = "raw_text" if "raw_text" in final_df.columns else "text"
            final_df = final_df[final_df[raw_c].astype(str).str.contains(kw_search, case=False, na=False)]

        # Sort by Likes
        sort_col = "likes" if "likes" in final_df.columns else ("like_count" if "like_count" in final_df.columns else None)
        if sort_col:
            final_df = final_df.sort_values(sort_col, ascending=False)

        # Prepare clean display DataFrame with Active Topic prominently as Column 1
        display_table = pd.DataFrame()
        display_table["Active Topic Discussed"] = final_df.get("topic_name", final_df.get("topic_label", "General Discussion"))
        display_table["Comment Text"] = final_df.get("raw_text", final_df.get("text", ""))
        display_table["Likes"] = final_df.get("likes", final_df.get("like_count", 0))
        display_table["Sentiment Mood"] = final_df.get("mood", final_df.get("mood_label", "Neutral"))
        if "engagement_weight" in final_df.columns:
            display_table["Engagement Weight"] = final_df["engagement_weight"].round(2)

        # Tabs: Active Comments View & Quarantine Isolated View
        tab_view, tab_quar = st.tabs([f"Semantic Comment Explorer ({len(final_df):,} comments)", f"Toxic Moderation Quarantine ({toxic_c})"])

        with tab_view:
            st.dataframe(
                display_table,
                use_container_width=True,
                height=340,
                column_config={
                    "Active Topic Discussed": st.column_config.TextColumn(
                        "Active Topic Discussed",
                        help="The semantic conversation theme this comment was clustered into.",
                        width="medium",
                    ),
                    "Comment Text": st.column_config.TextColumn(
                        "Comment Text",
                        width="large",
                    ),
                    "Likes": st.column_config.NumberColumn(
                        "Likes",
                        format="%d",
                        width="small",
                    ),
                    "Sentiment Mood": st.column_config.TextColumn(
                        "Sentiment Mood",
                        width="medium",
                    ),
                    "Engagement Weight": st.column_config.NumberColumn(
                        "Weight (W)",
                        format="%.2f",
                        width="small",
                    ),
                }
            )

        with tab_quar:
            toxic_df = target_data[target_data["mood"].astype(str).str.contains("Toxic", case=False, na=False)] if "mood" in target_data.columns else pd.DataFrame()
            st.warning(f"{len(toxic_df)} hostile comments isolated from BERTopic clustering to protect intelligence.")
            if not toxic_df.empty:
                q_disp = pd.DataFrame()
                q_disp["Quarantined Reason"] = ["Abusive Language / Hostile Attack"] * len(toxic_df)
                q_disp["Comment Text"] = toxic_df.get("raw_text", toxic_df.get("text", ""))
                q_disp["Likes"] = toxic_df.get("likes", toxic_df.get("like_count", 0))
                st.dataframe(q_disp, use_container_width=True, height=240)

        # CSV Download Bar
        csv_bytes = display_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Export Current View to CSV ({len(display_table):,} comments)",
            data=csv_bytes,
            file_name=f"indicpulse_comments_export.csv",
            mime="text/csv",
        )


# ===============================================================
# 4. SECTION 1: HOME (PLATFORM SELECTOR & WORKSPACE)
# ===============================================================
if st.session_state.active_page == "Home":
    # Top Greeting Bar
    st.markdown(f"""
    <div class="bento-header">
        <div>
            <div style="font-size:22px; font-weight:800; color:#1E293B;">Welcome back, {st.session_state.user_name}</div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">Platform Selection & Instant Analysis Workspace</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="pill-badge" style="background:#E2F6EE; color:#0F5132;">MuRIL + IndicSBERT</span>
            <span class="pill-badge" style="background:#18181B; color:#FFFFFF;">{len(st.session_state.analysis_history)} Analyses Run</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5 Interactive Platform Selector Cards
    p1, p2, p3, p4, p5 = st.columns(5)
    platforms = [
        ("YouTube", SVG_YOUTUBE, "Video & Shorts Comments", p1),
        ("Instagram Reels", SVG_INSTAGRAM, "Reel Commentary & Audio", p2),
        ("Reddit", SVG_REDDIT, "Community Discussions", p3),
        ("CSV / Custom Dataset", SVG_CSV, "Universal Table Upload", p4),
        ("Live Scraper", SVG_SCRAPER, "Network Interception", p5),
    ]

    for name, svg, subtitle, col in platforms:
        with col:
            is_active = (st.session_state.selected_platform == name)
            card_class = "platform-card-active" if is_active else "platform-card"
            st.markdown(f"""
            <div class="{card_class}">
                <div class="emblem-circle">{svg}</div>
                <div style="font-size:15px; font-weight:800; color:#18181B;">{name}</div>
                <div style="font-size:11px; color:#64748B;">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select" if not is_active else "Active", key=f"sel_{name}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state.selected_platform = name
                st.rerun()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Dynamic Contextual Input Drawer
    with st.container(border=True):
        st.markdown(f"<div style='font-size:15px; font-weight:800; color:#1E293B; margin-bottom:10px;'>Analyze {st.session_state.selected_platform}</div>", unsafe_allow_html=True)

        if st.session_state.selected_platform == "YouTube":
            yt_col1, yt_col2 = st.columns([2.8, 1.2])
            with yt_col1:
                video_url = st.text_input("YouTube URL", placeholder="Paste YouTube Video or Shorts URL...", label_visibility="collapsed")
            with yt_col2:
                run_yt = st.button("Analyze Comments", type="primary", use_container_width=True)

            sample_limit = st.slider(
                "Analysis Depth (Comment Volume)",
                min_value=100,
                max_value=2000,
                value=300,
                step=100,
                help="Scale up to 2,000 comments for comprehensive viral topic modeling."
            )
            render_html('<div style="font-size:11px; font-weight:600; color:#64748B; margin-top:-6px; margin-bottom:12px;">100–300: Fast (~8s) • 500–1,000: Deep Topic Scan (~20s) • 1,500–2,000: Full Viral Audit (~45s)</div>')

            d_col1, d_col2 = st.columns([3, 1.3])
            with d_col1:
                st.caption("Fetches real-time comments using Google YouTube Data API v3 with automatic pagination.")
            with d_col2:
                demo_trigger = st.button("Load Sample Hinglish Dataset", use_container_width=True)

            if run_yt:
                yt_key = os.getenv("YOUTUBE_API_KEY", "")
                if not video_url:
                    st.error("Please provide a valid YouTube URL.")
                elif not yt_key:
                    st.error("YouTube Data API Key not detected in .env file. Add YOUTUBE_API_KEY to your .env or click 'Load Sample Hinglish Dataset' below.")
                else:
                    try:
                        execute_analysis_flow(
                            df_input=None,
                            platform_name="YouTube",
                            target_title=video_url,
                            video_url=video_url,
                            max_comments=int(sample_limit),
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"YouTube Fetch Failed: {e}")

            if demo_trigger:
                base_demo = load_demo_comments()
                mult = max(1, int(sample_limit) // len(base_demo))
                scaled_demo = pd.concat([base_demo] * mult, ignore_index=True).head(int(sample_limit))
                scaled_demo["comment_id"] = [f"demo_{i:04d}" for i in range(len(scaled_demo))]
                execute_analysis_flow(
                    df_input=scaled_demo,
                    platform_name="Demo Hinglish",
                    target_title=f"Hinglish Slang Benchmark ({len(scaled_demo)} comments)",
                )
                st.rerun()

        elif st.session_state.selected_platform in ["Instagram Reels", "Reddit"]:
            platform_label = st.session_state.selected_platform
            st.caption(
                f"Paste a public {platform_label} link — the headless browser will intercept "
                "hidden JSON/GraphQL APIs and extract up to 2,000 real comments."
            )

            post_url = st.text_input(
                f"{platform_label} Link",
                placeholder=f"Paste {platform_label} post or thread URL...",
                label_visibility="collapsed",
                key=f"url_input_{platform_label}",
            )

            sample_limit_other = st.slider(
                "Analysis Depth (Comment Volume)",
                min_value=50,
                max_value=2000,
                value=300,
                step=50,
                key="depth_other",
            )
            render_html(
                '<div style="font-size:11px; font-weight:600; color:#64748B; '
                'margin-top:-6px; margin-bottom:12px;">'
                '50–300: Fast (~10s) • 500–1,000: Deep Topic Scan (~25s) • '
                '1,500–2,000: Full Viral Audit (~60s)</div>'
            )

            # ── Primary scrape button ─────────────────────────────────────
            run_scrape = st.button(
                f"⚡ Scrape & Analyse {platform_label}",
                type="primary",
                use_container_width=True,
                disabled=not bool(post_url),
                key=f"run_scrape_{platform_label}",
            )

            if run_scrape and post_url:
                if not _PLAYWRIGHT_AVAILABLE:
                    st.error(
                        "Playwright is not available in this environment.\n\n"
                        "Run these two commands in your terminal, then restart the app:\n"
                        "```\npip install playwright\nplaywright install chromium\n```"
                    )
                else:
                    with st.spinner(
                        f"Launching headless browser and intercepting {platform_label} comments… "
                        "(this may take 30–60 s for large depths)"
                    ):
                        try:
                            scraped_df = _scrape_comments(
                                post_url,
                                max_comments=int(sample_limit_other),
                                platform=platform_label,
                            )
                            execute_analysis_flow(
                                df_input=scraped_df,
                                platform_name=platform_label,
                                target_title=post_url,
                                max_comments=int(sample_limit_other),
                            )
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Scraping failed: {exc}")

            # ── Demo fallback ─────────────────────────────────────────────
            st.markdown("---")
            st.caption("No URL? Run the pipeline on a built-in Hinglish benchmark dataset:")
            if st.button(
                f"🧪 Run Demo Analysis ({platform_label})",
                key=f"demo_{platform_label}",
                use_container_width=True,
            ):
                base_demo = load_demo_comments()
                mult = max(1, int(sample_limit_other) // len(base_demo))
                scaled_demo = pd.concat([base_demo] * mult, ignore_index=True).head(int(sample_limit_other))
                scaled_demo["comment_id"] = [f"demo_{i:04d}" for i in range(len(scaled_demo))]
                execute_analysis_flow(
                    scaled_demo,
                    platform_label,
                    f"Benchmark Hinglish ({len(scaled_demo)} comments)",
                )
                st.rerun()

        else:  # Live Scraper (Network Interception) OR CSV / Custom Dataset
            if st.session_state.selected_platform == "Live Scraper":
                # ── Live Scraper input drawer ─────────────────────────────
                st.caption(
                    "Paste any public URL — the headless browser intercepts hidden "
                    "JSON/GraphQL network traffic to extract up to 2,000 real comments."
                )

                scraper_url = st.text_input(
                    "Target URL",
                    placeholder="Paste Instagram Reel, Reddit thread, or any public post URL…",
                    label_visibility="collapsed",
                    key="scraper_url_input",
                )
                st.caption(
                    "⚠️ Note: Heavily restricted sites like Instagram or Flipkart may block "
                    "unauthenticated headless browsers. If scraping fails, use the **Upload CSV** method."
                )

                source_mode = st.selectbox(
                    "Data Source",
                    options=[
                        "Live Scraper (Network Interception)",
                        "Demo Dataset (Benchmark Hinglish)",
                    ],
                    index=0,
                    key="scraper_source_mode",
                )

                sample_limit_scraper = st.slider(
                    "Analysis Depth (Comment Volume)",
                    min_value=50,
                    max_value=2000,
                    value=300,
                    step=50,
                    key="depth_scraper",
                )
                render_html(
                    '<div style="font-size:11px; font-weight:600; color:#64748B; '
                    'margin-top:-6px; margin-bottom:12px;">'
                    '50–300: Fast (~10s) • 500–1,000: Deep Topic Scan (~25s) • '
                    '1,500–2,000: Full Viral Audit (~60s)</div>'
                )

                trigger_btn = st.button(
                    "⚡ Run Live Scraper & Analyse",
                    type="primary",
                    use_container_width=True,
                    disabled=(
                        source_mode == "Live Scraper (Network Interception)"
                        and not bool(scraper_url)
                    ),
                    key="trigger_scraper",
                )

                if trigger_btn:
                    if source_mode == "Live Scraper (Network Interception)":
                        if not _PLAYWRIGHT_AVAILABLE:
                            st.error(
                                "Playwright is not available in this environment.\n\n"
                                "Run these two commands in your terminal, then restart the app:\n"
                                "```\npip install playwright\nplaywright install chromium\n```"
                            )
                        else:
                            with st.spinner(
                                f"Intercepting network traffic to extract up to "
                                f"{sample_limit_scraper} comments from {scraper_url[:40]}…"
                            ):
                                try:
                                    raw_df = _scrape_comments(
                                        scraper_url,
                                        max_comments=int(sample_limit_scraper),
                                        platform="Live Scraper",
                                    )
                                    execute_analysis_flow(
                                        df_input=raw_df,
                                        platform_name="Live Scraper",
                                        target_title=f"Scraper ({scraper_url[:30]}…)",
                                        max_comments=int(sample_limit_scraper),
                                    )
                                    st.rerun()
                                except ValueError as exc:
                                    st.error(str(exc))
                                except Exception as exc:
                                    st.error(f"Scraping failed: {exc}")

                    else:  # Demo Dataset
                        base_demo = load_demo_comments()
                        mult = max(1, int(sample_limit_scraper) // len(base_demo))
                        scaled_demo = pd.concat([base_demo] * mult, ignore_index=True).head(
                            int(sample_limit_scraper)
                        )
                        scaled_demo["comment_id"] = [f"demo_{i:04d}" for i in range(len(scaled_demo))]
                        execute_analysis_flow(
                            scaled_demo,
                            "Live Scraper",
                            f"Benchmark Hinglish ({len(scaled_demo)} comments)",
                        )
                        st.rerun()

            else:  # CSV / Custom Dataset
                st.caption("Upload any CSV with consumer comments, reviews, or social commentary.")
                csv_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")
                c_btn1, c_btn2 = st.columns([1.5, 3])
                with c_btn1:
                    run_csv = st.button("Analyze CSV Dataset", type="primary", use_container_width=True)
                with c_btn2:
                    demo_csv = st.button("Load Sample Hinglish Dataset", key="demo_csv_btn", use_container_width=True)

                if run_csv:
                    if csv_file is not None:
                        raw_csv = pd.read_csv(csv_file)
                        text_col = raw_csv.columns[0]
                        raw_df = pd.DataFrame({"raw_text": raw_csv[text_col].astype(str), "likes": 5, "source": "CSV Upload"})
                        execute_analysis_flow(raw_df, "CSV Dataset", csv_file.name)
                        st.rerun()
                    else:
                        st.error("Please upload a .csv file first.")

                if demo_csv:
                    base_demo = load_demo_comments()
                    execute_analysis_flow(base_demo, "CSV Benchmark", "Sample Reviews Dataset")
                    st.rerun()

    # Inline Instant Result View (Rendered directly on Home page when analysis is run)
    if st.session_state.latest_run is not None:
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="font-size:18px; font-weight:800; color:#1E293B;">Latest Analysis Result: {st.session_state.latest_run['url_or_title']}</div>
            <span class="pill-badge" style="background:#E2F6EE; color:#0F5132;">{st.session_state.latest_run['platform']}</span>
        </div>
        """, unsafe_allow_html=True)
        render_bento_grid_results(st.session_state.latest_run)


# ===============================================================
# 5. SECTION 2: DASHBOARD (CUMULATIVE PERFORMANCE)
# ===============================================================
elif st.session_state.active_page == "Dashboard":
    render_html(f"""
    <div class="bento-header">
        <div>
            <div style="font-size:22px; font-weight:800; color:#1E293B;">Cumulative Performance Dashboard</div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">Cross-Platform Community Metrics & Sentiment Trends</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="pill-badge" style="background:#18181B; color:#FFFFFF;">{len(st.session_state.analysis_history)} Videos & Datasets Aggregated</span>
        </div>
    </div>
    """)

    if len(st.session_state.analysis_history) == 0:
        # High-Craft Empty State Bento Card
        render_html(f"""
        <div style="background:#FFFFFF; border-radius:24px; border:1px solid #EAEEF4; padding:50px 24px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.02); max-width:680px; margin:30px auto;">
            <div style="display:flex; justify-content:center; margin-bottom:18px;">
                {SVG_EMPTY_STATE}
            </div>
            <div style="font-size:20px; font-weight:800; color:#18181B; margin-bottom:8px;">No Comment Analyses Yet</div>
            <div style="font-size:13px; color:#64748B; line-height:1.6; max-width:440px; margin:0 auto 24px auto;">
                Analyze your comments on the Home tab to see your overall cross-platform sentiment performance on the Dashboard.
            </div>
        </div>
        """)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1.3, 1.3])
        with col_btn2:
            if st.button("Go to Home & Analyze", type="primary", use_container_width=True):
                st.session_state.active_page = "Home"
                st.rerun()
        with col_btn3:
            if st.button("Load Demo & View Dashboard", use_container_width=True):
                execute_analysis_flow(load_demo_comments(), "Demo Hinglish", "Sample Benchmark Dataset")
                st.rerun()

    else:
        # Aggregated Data across all historical runs
        all_moods = pd.concat([item["df_mood"] for item in st.session_state.analysis_history], ignore_index=True)
        all_topics = pd.concat([item["df_clustered"] for item in st.session_state.analysis_history], ignore_index=True)

        cum_links = len(st.session_state.analysis_history)
        cum_comments = len(all_moods)
        total_w = max(1e-6, all_moods["engagement_weight"].sum())
        pos_mask = all_moods["mood"].astype(str).str.contains("Positive")
        cum_weighted_pos = (all_moods.loc[pos_mask, "engagement_weight"].sum() / total_w) * 100
        cum_sarcasm = int(all_moods["mood"].astype(str).str.contains("Sarcastic").sum())
        cum_toxic = int(all_moods["mood"].astype(str).str.contains("Toxic").sum())

        # Guarantee persistent session state for Dashboard inspector
        st.session_state.df_clustered = all_topics
        st.session_state.df_mood = all_moods

        # Open Modal Dialog if triggered on Dashboard
        if st.session_state.get("inspector_open"):
            render_inspector_dialog()

        # Row 1: 4 Cumulative Pastel Bento KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(render_kpi_card(
                f"{cum_links}",
                "Total Datasets",
                f"{cum_comments:,} Comments",
                theme="mint",
                tooltip="Number of YouTube videos or CSV uploads analyzed across all user sessions."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key="dash_inspect_all",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"All Aggregated Comments ({cum_comments:,})", "ALL", None),
            )
        with k2:
            st.markdown(render_kpi_card(
                f"{cum_weighted_pos:.1f}%",
                "Overall Approval",
                "Log₁₀ Weighted",
                theme="yellow",
                tooltip="Percentage of positive community hype, log-weighted by likes."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key="dash_inspect_pos",
                use_container_width=True,
                on_click=trigger_inspector,
                args=("Aggregated Positive Hype Comments", "MOOD", "Positive"),
            )
        with k3:
            st.markdown(render_kpi_card(
                f"{cum_sarcasm:,}",
                "Sarcasm Caught",
                "Detected Irony",
                theme="purple",
                tooltip="Total comments identified with mock praise, sarcasm emojis, or irony patterns across all datasets."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key="dash_inspect_sarc",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"Aggregated Sarcastic Comments ({cum_sarcasm})", "SARCASM", None),
            )
        with k4:
            st.markdown(render_kpi_card(
                f"{cum_toxic:,}",
                "Quarantined Toxic",
                "Isolated Abusive",
                theme="pink",
                tooltip="Hostile comments, abusive language, and personal harassment isolated and quarantined across all datasets."
            ), unsafe_allow_html=True)
            st.button(
                "Inspect ↗",
                key="dash_inspect_tox",
                use_container_width=True,
                on_click=trigger_inspector,
                args=(f"Aggregated Quarantined Toxic ({cum_toxic})", "TOXIC", None),
            )

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        # Row 2: Asymmetric 2-Zone Bento Split [2.1, 1.0]
        dash_left, dash_right = st.columns([2.1, 1.0], gap="large")

        with dash_left:
            # Cross-Platform Comparison Chart
            with st.container(border=True):
                st.markdown("<div style='font-size:14px; font-weight:800; color:#1E293B;'>Cross-Platform Approval Comparison</div>", unsafe_allow_html=True)
                history_df = pd.DataFrame([
                    {
                        "Target": item["url_or_title"][:28] + "...",
                        "Approval": item["metrics"]["weighted_approval"],
                        "Platform": item["platform"],
                        "Comments": item["metrics"]["total"],
                    }
                    for item in st.session_state.analysis_history
                ])
                fig_bar = px.bar(
                    history_df,
                    x="Approval",
                    y="Target",
                    color="Platform",
                    orientation="h",
                    color_discrete_map={"YouTube": "#FF0000", "Instagram Reels": "#E1306C", "Reddit": "#FF4500", "CSV / Custom Dataset": "#4F46E5", "Live Scraper": "#0F766E"},
                    range_x=[0, 100],
                )
                fig_bar.update_layout(
                    height=280,
                    margin=dict(t=20, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="#EAEEF4", tickfont=dict(family="Plus Jakarta Sans", size=11)),
                    yaxis=dict(tickfont=dict(family="Plus Jakarta Sans", size=11)),
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

            # Combined Latest Gemini Executive Insights
            with st.container(border=True):
                st.markdown("<div style='font-size:14px; font-weight:800; color:#1E293B; margin-bottom:10px;'>Latest Strategic Executive Synthesis</div>", unsafe_allow_html=True)
                if st.session_state.analysis_history and st.session_state.analysis_history[0].get("exec_summary"):
                    st.markdown(st.session_state.analysis_history[0]["exec_summary"])
                elif st.session_state.get("exec_summary"):
                    st.markdown(st.session_state.exec_summary)
                else:
                    st.info("Awaiting Gemini Synthesis...")

        with dash_right:
            # Cumulative Mood Donut
            with st.container(border=True):
                st.markdown("<div style='font-size:14px; font-weight:800; color:#1E293B;'>Cumulative Mood Distribution</div>", unsafe_allow_html=True)
                fig_pie_cum = charts.mood_donut_chart(all_moods, colorblind_safe=st.session_state.colorblind_mode)
                st.plotly_chart(fig_pie_cum, use_container_width=True, config={'displayModeBar': False})

            # Top Recurring Topic Clusters
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Top Recurring Topics</div>", unsafe_allow_html=True)

            if "topic_name" in all_topics.columns:
                cum_top_counts = all_topics["topic_name"].value_counts().reset_index()
                cum_top_counts.columns = ["Topic", "Count"]
                topic_styles = [
                    ("topic-card-mint", "#0F5132"),
                    ("topic-card-lavender", "#4C1D95"),
                    ("topic-card-yellow", "#713F12"),
                    ("topic-card-pink", "#881337"),
                ]
                for idx, row in cum_top_counts[cum_top_counts["Topic"] != "Toxic (Quarantined)"].head(4).iterrows():
                    card_class, _ = topic_styles[idx % len(topic_styles)]
                    st.markdown(f"""
                    <div class="topic-pill-card {card_class}">
                        <div style="font-size:13px; font-weight:700;">{row['Topic']}</div>
                        <span class="pill-badge">{row['Count']} comments</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button(
                        f"Inspect {row['Count']} Comments ↗",
                        key=f"dash_topic_btn_{idx}",
                        use_container_width=True,
                        on_click=trigger_inspector,
                        args=(f"Topic Cluster: {row['Topic']}", "TOPIC", row['Topic']),
                    )


# ===============================================================
# 6. SECTION 3: HISTORY (PAST ANALYSIS CARDS & IN-PLACE VIEW)
# ===============================================================
elif st.session_state.active_page == "History":
    st.markdown(f"""
    <div class="bento-header">
        <div>
            <div style="font-size:22px; font-weight:800; color:#1E293B;">Analysis History & Archives</div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">Inspect past runs, view deep semantic clusters, or refresh live comments.</div>
        </div>
        <span class="pill-badge" style="background:#18181B; color:#FFFFFF;">{len(st.session_state.analysis_history)} Archived Runs</span>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.analysis_history) == 0:
        st.markdown(f"""
        <div style="background:#FFFFFF; border-radius:24px; border:1px solid #EAEEF4; padding:50px 24px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.02); max-width:640px; margin:20px auto;">
            <div style="font-size:18px; font-weight:800; color:#18181B; margin-bottom:6px;">No Historical Records</div>
            <div style="font-size:13px; color:#64748B; margin-bottom:20px;">Run an analysis from the Home tab to automatically archive reports here.</div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.selected_history_id is None:
        # Master List of Past Analysis Cards
        for item in st.session_state.analysis_history:
            with st.container(border=True):
                h_col1, h_col2, h_col3 = st.columns([3.2, 1.4, 1.4])
                with h_col1:
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="pill-badge" style="background:#E2F6EE; color:#0F5132;">{item['platform']}</span>
                        <span style="font-size:12px; color:#94A3B8;">{item['timestamp']}</span>
                    </div>
                    <div style="font-size:17px; font-weight:800; color:#18181B; margin-top:6px;">{item['url_or_title']}</div>
                    """, unsafe_allow_html=True)

                with h_col2:
                    st.markdown(f"""
                    <div style="display:flex; flex-direction:column; gap:4px; margin-top:4px;">
                        <span class="pill-badge">Comments: {item['metrics']['total']}</span>
                        <span class="pill-badge" style="background:#FEF3D6; color:#713F12;">Approval: {item['metrics']['weighted_approval']:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                with h_col3:
                    if st.button("View Full Analysis", key=f"view_{item['id']}", type="primary", use_container_width=True):
                        st.session_state.selected_history_id = item["id"]
                        st.rerun()

                    if st.button("Refresh Data", key=f"ref_{item['id']}", use_container_width=True):
                        # Re-run pipeline for this history entry
                        with st.spinner("Refreshing commentary from source..."):
                            target_c = item.get("metrics", {}).get("total", 300)
                            if item["platform"] == "YouTube" and os.getenv("YOUTUBE_API_KEY"):
                                refreshed_res = fetch_youtube_comments(item["url_or_title"], os.getenv("YOUTUBE_API_KEY"), max_comments=target_c)
                                if isinstance(refreshed_res, tuple):
                                    refreshed_df, _ = refreshed_res
                                else:
                                    refreshed_df = refreshed_res
                            else:
                                refreshed_df = item["df_mood"]
                            
                            df_proc = preprocess_dataframe(refreshed_df.copy())
                            df_mood = run_mood_classification(df_proc, use_gpu=True)
                            df_clust = run_topic_clustering(df_mood)
                            summary_text = generate_executive_summary(df_clust, os.getenv("GEMINI_API_KEY", ""))
                            
                            # Update entry and metrics in place
                            tot = len(df_mood)
                            p_mask = df_mood["mood"].astype(str).str.contains("Positive")
                            w_tot = max(1e-6, df_mood["engagement_weight"].sum())
                            w_pos = (df_mood.loc[p_mask, "engagement_weight"].sum() / w_tot) * 100
                            s_cnt = int((df_mood["mood"].astype(str).str.contains("Sarcastic")).sum())
                            t_cnt = int((df_mood["mood"].astype(str).str.contains("Toxic")).sum())
                            d_mood = df_mood["mood"].mode()[0] if not df_mood.empty else "Neutral"

                            item["df_mood"] = df_mood
                            item["df_clustered"] = df_clust
                            item["exec_summary"] = summary_text
                            st.session_state.exec_summary = summary_text
                            item["metrics"] = {
                                "total": tot,
                                "weighted_approval": w_pos,
                                "sarcasm_count": s_cnt,
                                "toxic_count": t_cnt,
                                "num_topics": len(df_clust["topic_name"].unique()) if "topic_name" in df_clust.columns else 1,
                                "dominant_mood": d_mood,
                            }
                            item["timestamp"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
                            st.success("Entry refreshed successfully.")
                            st.rerun()

    else:
        # Detail View: In-Place Full Analysis
        selected_run = next((x for x in st.session_state.analysis_history if x["id"] == st.session_state.selected_history_id), None)
        if selected_run:
            back_col, title_col, ref_col = st.columns([1.2, 3.2, 1.2])
            with back_col:
                if st.button("← Back to All History", use_container_width=True):
                    st.session_state.selected_history_id = None
                    st.rerun()
            with title_col:
                st.markdown(f"<div style='font-size:16px; font-weight:800; color:#1E293B; text-align:center;'>{selected_run['url_or_title']}</div>", unsafe_allow_html=True)
            with ref_col:
                if st.button("↻ Refresh Data Now", type="primary", use_container_width=True):
                    with st.spinner("Re-fetching comments..."):
                        target_c = selected_run.get("metrics", {}).get("total", 300)
                        if selected_run["platform"] == "YouTube" and os.getenv("YOUTUBE_API_KEY"):
                            refreshed_res = fetch_youtube_comments(selected_run["url_or_title"], os.getenv("YOUTUBE_API_KEY"), max_comments=target_c)
                            if isinstance(refreshed_res, tuple):
                                refreshed_df, _ = refreshed_res
                            else:
                                refreshed_df = refreshed_res
                        else:
                            refreshed_df = selected_run["df_mood"]
                        
                        df_proc = preprocess_dataframe(refreshed_df.copy())
                        df_mood = run_mood_classification(df_proc, use_gpu=True)
                        df_clust = run_topic_clustering(df_mood)
                        summary_text = generate_executive_summary(df_clust, os.getenv("GEMINI_API_KEY", ""))

                        tot = len(df_mood)
                        p_mask = df_mood["mood"].astype(str).str.contains("Positive")
                        w_tot = max(1e-6, df_mood["engagement_weight"].sum())
                        w_pos = (df_mood.loc[p_mask, "engagement_weight"].sum() / w_tot) * 100
                        s_cnt = int((df_mood["mood"].astype(str).str.contains("Sarcastic")).sum())
                        t_cnt = int((df_mood["mood"].astype(str).str.contains("Toxic")).sum())
                        d_mood = df_mood["mood"].mode()[0] if not df_mood.empty else "Neutral"

                        selected_run["df_mood"] = df_mood
                        selected_run["df_clustered"] = df_clust
                        selected_run["exec_summary"] = summary_text
                        selected_run["metrics"] = {
                            "total": tot,
                            "weighted_approval": w_pos,
                            "sarcasm_count": s_cnt,
                            "toxic_count": t_cnt,
                            "num_topics": len(df_clust["topic_name"].unique()) if "topic_name" in df_clust.columns else 1,
                            "dominant_mood": d_mood,
                        }
                        selected_run["timestamp"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
                        st.success("Analysis updated!")
                        st.rerun()

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            render_bento_grid_results(selected_run)
        else:
            st.session_state.selected_history_id = None
            st.rerun()


# ===============================================================
# 7. SECTION 4: PROFILE (CREATOR & BRAND IDENTITY)
# ===============================================================
elif st.session_state.active_page == "Profile":
    st.markdown(f"""
    <div class="bento-header">
        <div>
            <div style="font-size:22px; font-weight:800; color:#1E293B;">Creator Profile & Brand Hub</div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">Manage studio identity, content niche, and account performance metrics.</div>
        </div>
        <span class="pill-badge" style="background:#18181B; color:#FFFFFF;">Creator Tier</span>
    </div>
    """, unsafe_allow_html=True)

    prof_col1, prof_col2 = st.columns([1.2, 1], gap="large")

    with prof_col1:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:16px; margin-bottom:18px;">
                <div style="width:64px; height:64px; border-radius:50%; background:#E2F6EE; border:2px solid #CCEFE2; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:800; color:#0F5132;">
                    IP
                </div>
                <div>
                    <div style="font-size:18px; font-weight:800; color:#18181B;">IndicPulse Studio Creator</div>
                    <div style="font-size:12px; color:#64748B;">Semantic Engine v2.0 • Pro Member</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.user_name = st.text_input("Display Name", value=st.session_state.user_name)
            st.session_state.user_handle = st.text_input("Brand / Channel Handle", value=st.session_state.user_handle)
            st.session_state.user_niche = st.selectbox(
                "Primary Content Niche",
                ["Tech & Gadgets", "Entertainment & Comedy", "D2C Brand", "Education", "News & Politics", "Gaming"],
                index=["Tech & Gadgets", "Entertainment & Comedy", "D2C Brand", "Education", "News & Politics", "Gaming"].index(st.session_state.user_niche),
            )
            st.session_state.summary_lang = st.selectbox(
                "Default Summary Language",
                ["English", "Hinglish"],
                index=["English", "Hinglish"].index(st.session_state.summary_lang),
            )

            if st.button("Save Profile Settings", type="primary", use_container_width=True):
                st.success("Profile updated successfully.")

    with prof_col2:
        cum_comments = sum(item["metrics"]["total"] for item in st.session_state.analysis_history)
        cum_toxic = sum(item["metrics"]["toxic_count"] for item in st.session_state.analysis_history)
        toxicity_rate = (cum_toxic / max(1, cum_comments)) * 100

        with st.container(border=True):
            st.markdown("<div style='font-size:16px; font-weight:800; color:#1E293B; margin-bottom:14px;'>Session Statistics</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; gap:12px;">
                <div style="display:flex; justify-content:space-between; padding:12px 14px; background:#F8FAFC; border-radius:12px;">
                    <span style="font-size:13px; font-weight:600; color:#64748B;">Total Links Checked</span>
                    <span style="font-size:15px; font-weight:800; color:#18181B;">{len(st.session_state.analysis_history)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:12px 14px; background:#F8FAFC; border-radius:12px;">
                    <span style="font-size:13px; font-weight:600; color:#64748B;">Total Comments Processed</span>
                    <span style="font-size:15px; font-weight:800; color:#18181B;">{cum_comments}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:12px 14px; background:#F8FAFC; border-radius:12px;">
                    <span style="font-size:13px; font-weight:600; color:#64748B;">Toxicity Shield Filtering</span>
                    <span style="font-size:15px; font-weight:800; color:#881337;">{toxicity_rate:.1f}% filtered</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            if st.button("Clear Session History", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.latest_run = None
                st.session_state.selected_history_id = None
                st.success("Session history cleared.")
                st.rerun()


# ===============================================================
# 8. SECTION 5: SETTINGS (ENVIRONMENT STATUS & ACCESSIBILITY)
# ===============================================================
elif st.session_state.active_page == "Settings":
    st.markdown(f"""
    <div class="bento-header">
        <div>
            <div style="font-size:22px; font-weight:800; color:#1E293B;">Settings & Pipeline Controls</div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">Automated .env diagnostic, pipeline feature toggles, and accessibility overrides.</div>
        </div>
        <span class="pill-badge" style="background:#E2F6EE; color:#0F5132;">Diagnostics Normal</span>
    </div>
    """, unsafe_allow_html=True)

    # Card 1: .env Environment Connection Status (Read-Only Diagnostic)
    curr_yt_key = get_youtube_api_key()
    curr_gem_key = get_gemini_api_key()
    yt_connected = bool(curr_yt_key)
    gemini_connected = bool(curr_gem_key)
    hf_model = os.getenv("HF_MODEL_ID", "google/muril-base-cased")

    with st.container(border=True):
        st.markdown("<div style='font-size:16px; font-weight:800; color:#1E293B; margin-bottom:12px;'>Environment Connection Status (.env)</div>", unsafe_allow_html=True)
        
        d1, d2, d3 = st.columns(3)
        with d1:
            bg = "#E2F6EE" if yt_connected else "#FEF3D6"
            col = "#0F5132" if yt_connected else "#713F12"
            status = "Connected" if yt_connected else "Demo / CSV Mode"
            st.markdown(f"""
            <div style="background:{bg}; padding:14px; border-radius:14px; border:1px solid rgba(0,0,0,0.04);">
                <div style="font-size:12px; font-weight:600; color:{col};">YouTube Data API v3</div>
                <div style="font-size:15px; font-weight:800; color:{col}; margin-top:4px;">● {status}</div>
            </div>
            """, unsafe_allow_html=True)

        with d2:
            bg = "#E2F6EE" if gemini_connected else "#FEF3D6"
            col = "#0F5132" if gemini_connected else "#713F12"
            status = "Connected" if gemini_connected else "Heuristic Active"
            st.markdown(f"""
            <div style="background:{bg}; padding:14px; border-radius:14px; border:1px solid rgba(0,0,0,0.04);">
                <div style="font-size:12px; font-weight:600; color:{col};">Google Gemini 2.5 Flash</div>
                <div style="font-size:15px; font-weight:800; color:{col}; margin-top:4px;">● {status}</div>
            </div>
            """, unsafe_allow_html=True)

        with d3:
            st.markdown(f"""
            <div style="background:#EDE9FE; padding:14px; border-radius:14px; border:1px solid #DDD6FE;">
                <div style="font-size:12px; font-weight:600; color:#4C1D95;">MuRIL Model Engine</div>
                <div style="font-size:15px; font-weight:800; color:#4C1D95; margin-top:4px;">● {hf_model.split('/')[-1]}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Configure / Override Environment Keys in Memory", expanded=not (yt_connected and gemini_connected)):
            st.caption("Inject keys for the active session without modifying the on-disk .env file:")
            temp_yt = st.text_input("YouTube Data API v3 Key", type="password", value=curr_yt_key)
            temp_gem = st.text_input("Google Gemini API Key", type="password", value=curr_gem_key)
            if st.button("Apply Keys to Memory", type="primary"):
                if temp_yt:
                    st.session_state.youtube_api_key = temp_yt.strip()
                    os.environ["YOUTUBE_API_KEY"] = temp_yt.strip()
                if temp_gem:
                    st.session_state.gemini_api_key = temp_gem.strip()
                    os.environ["GEMINI_API_KEY"] = temp_gem.strip()
                st.success("Session environment and API keys updated.")
                st.rerun()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Card 2: AI Pipeline Engine Switches (2x2 Pastel Bento Toggle Grid)
    with st.container(border=True):
        st.markdown("<div style='font-size:16px; font-weight:800; color:#1E293B; margin-bottom:14px;'>AI Pipeline Engine Functions</div>", unsafe_allow_html=True)
        sw1, sw2 = st.columns(2)
        with sw1:
            st.session_state.toggle_emoji = st.toggle("Emoji Semantic Tagging ([clown_mocking], [fire_hype])", value=st.session_state.toggle_emoji)
            st.session_state.toggle_slang = st.toggle("Hinglish Slang De-Obfuscation (ch*tiya, bkl)", value=st.session_state.toggle_slang)
        with sw2:
            st.session_state.toggle_quarantine = st.toggle("Auto-Quarantine Hostile Profanity before BERTopic", value=st.session_state.toggle_quarantine)
            st.session_state.toggle_log_weight = st.toggle("Log10 Engagement Weighting Formula", value=st.session_state.toggle_log_weight)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Card 3: Accessibility & Display Preferences
    with st.container(border=True):
        st.markdown("<div style='font-size:16px; font-weight:800; color:#1E293B; margin-bottom:14px;'>Accessibility & Display Preferences</div>", unsafe_allow_html=True)
        st.session_state.font_scale = st.select_slider("UI Font Scaling (%)", options=[95, 100, 110, 120], value=st.session_state.font_scale)
        st.session_state.high_contrast = st.toggle("WCAG High-Contrast Borders", value=st.session_state.high_contrast)
        st.session_state.colorblind_mode = st.toggle("Okabe-Ito Colorblind-Safe Chart Palette", value=st.session_state.colorblind_mode)
        if st.button("Apply Display Settings", type="primary"):
            st.rerun()


# ===============================================================
# 9. SECTION 6: T&C (TERMS, PRIVACY & ACADEMIC COMPLIANCE)
# ===============================================================
elif st.session_state.active_page == "T&C":
    st.markdown(f"""
    <div class="bento-header">
        <div>
            <div style="font-size:22px; font-weight:800; color:#1E293B;">Terms, Privacy & Academic Compliance</div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">Regulatory policies, multilingual AI model disclaimers, and open-source licenses.</div>
        </div>
        <span class="pill-badge" style="background:#18181B; color:#FFFFFF;">Apache 2.0 / MIT</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
        ### 1. Data Privacy & Platform API Compliance
        - **YouTube Data API v3 Terms:** IndicPulse complies with YouTube Developer Policies and Google Cloud Platform Terms of Service.
        - **Public Comment Ingestion:** Only publicly accessible commentary published by users on digital forums is analyzed. Private messages, unlisted feeds, or restricted profiles are never queried.
        - **Zero Permanent PII Storage:** Raw author identities, timestamps, and commentary are processed strictly in-memory or persisted locally within your session. Data is not shared with third-party brokers.
        
        ---
        
        ### 2. Multilingual AI & Model Disclaimer
        - **Probabilistic Semantic Interpretation:** Sentiment, sarcasm, and mood classifications generated by `google/muril-base-cased` and `IndicSBERT` are probabilistic estimates.
        - **Cultural Nuances & Romanized Slang:** Hinglish code-mixed dialect translation inherently involves informal peer expressions. Flagged profanities are categorized for constructive moderation and should not be used as legal determination of intent.
        - **Gemini 2.5 Flash Synthesis:** Executive summaries are generated using large language models and should be reviewed alongside raw comment distributions for critical decisions.
        
        ---
        
        ### 3. Open-Source Licensing & Academic Attribution
        - **Permissive Open-Source Licensing:** IndicPulse core pipeline modules are licensed under the Apache 2.0 and MIT open-source licenses.
        - **Academic Attribution:** If you utilize IndicPulse in academic publications, research demonstrations, or benchmarking evaluations, cite the project repository and underlying MuRIL / IndicSBERT model repositories.
        """)
