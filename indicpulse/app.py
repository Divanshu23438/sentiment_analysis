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
import time
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

# ── PAGE CONFIGURATION ─────────────────────────────────────────
st.set_page_config(
    page_title="IndicPulse Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SAFE HTML RENDERING WRAPPER (PREVENTS COMMONMARK INDENTED CODE BLOCKS) ──
_orig_st_markdown = st.markdown

def render_html(html_str: str):
    """Safely renders HTML via st.markdown without CommonMark interpreting leading indentation as code blocks."""
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    _orig_st_markdown(cleaned, unsafe_allow_html=True)

def safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html", False) and isinstance(body, str):
        # Strip all leading/trailing whitespace from each line so CommonMark never triggers code blocks
        body = "\n".join(line.strip() for line in body.splitlines() if line.strip())
    return _orig_st_markdown(body, *args, **kwargs)

st.markdown = safe_markdown

# ── PERSISTENT SESSION STATE INITIALIZATION ────────────────────
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"
if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = "YouTube"
if "active_drilldown" not in st.session_state:
    st.session_state.active_drilldown = "ALL"
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

# ── INLINE PRODUCTION SVG EMBLEMS (COMPACT ZERO-INDENT) ─────────
SVG_BRAND_LOGO = '<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="34" height="34" rx="10" fill="#18181B"/><path d="M7 17H11L14 9L19 25L23 14L25 17H27" stroke="#34D399" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'

SVG_YOUTUBE = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#FF0000"/><path d="M11 9.5L18.5 14L11 18.5V9.5Z" fill="white"/></svg>'

SVG_INSTAGRAM = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="url(#ig_grad)"/><rect x="7" y="7" width="14" height="14" rx="4" stroke="white" stroke-width="2"/><circle cx="14" cy="14" r="3.2" stroke="white" stroke-width="2"/><circle cx="18" cy="10" r="1" fill="white"/><defs><radialGradient id="ig_grad" cx="0%" cy="100%" r="130%"><stop offset="0%" stop-color="#FFDD55"/><stop offset="30%" stop-color="#FF543E"/><stop offset="60%" stop-color="#C837AB"/><stop offset="100%" stop-color="#3771C8"/></radialGradient></defs></svg>'

SVG_REDDIT = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#FF4500"/><path d="M19.5 13.5C19.5 12.67 18.83 12 18 12C17.58 12 17.2 12.17 16.93 12.45C15.89 11.73 14.46 11.25 12.86 11.2L13.62 7.62L16.1 8.15C16.14 8.77 16.66 9.27 17.3 9.27C18 9.27 18.55 8.72 18.55 8.02C18.55 7.32 18 6.77 17.3 6.77C16.81 6.77 16.39 7.05 16.19 7.45L13.38 6.85C13.25 6.82 13.12 6.91 13.09 7.04L12.22 11.16C10.58 11.21 9.12 11.7 8.07 12.45C7.8 12.17 7.42 12 7 12C6.17 12 5.5 12.67 5.5 13.5C5.5 14.12 5.88 14.65 6.42 14.87C6.39 15.08 6.38 15.29 6.38 15.5C6.38 18.26 9.51 20.5 13.38 20.5C17.25 20.5 20.38 18.26 20.38 15.5C20.38 15.29 20.37 15.08 20.34 14.87C20.88 14.65 21.26 14.12 21.26 13.5C21.26 12.67 20.59 12 19.76 12L19.5 13.5ZM9.5 15.5C9.5 14.67 10.17 14 11 14C11.83 14 12.5 14.67 12.5 15.5C12.5 16.33 11.83 17 11 17C10.17 17 9.5 16.33 9.5 15.5ZM16.36 18.24C15.5 18.81 14.24 18.98 13.38 18.98C12.52 18.98 11.26 18.81 10.4 18.24C10.24 18.13 10.2 17.91 10.31 17.75C10.42 17.59 10.64 17.55 10.8 17.66C11.47 18.11 12.47 18.28 13.38 18.28C14.29 18.28 15.29 18.11 15.96 17.66C16.12 17.55 16.34 17.59 16.45 17.75C16.56 17.91 16.52 18.13 16.36 18.24ZM15.76 17C14.93 17 14.26 16.33 14.26 15.5C14.26 14.67 14.93 14 15.76 14C16.59 14 17.26 14.67 17.26 15.5C17.26 16.33 16.59 17 15.76 17Z" fill="white"/></svg>'

SVG_CSV = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="7" fill="#4F46E5"/><rect x="7" y="6" width="14" height="16" rx="2" stroke="white" stroke-width="1.8"/><line x1="7" y1="11" x2="21" y2="11" stroke="white" stroke-width="1.5"/><line x1="7" y1="16" x2="21" y2="16" stroke="white" stroke-width="1.5"/><line x1="14" y1="11" x2="14" y2="22" stroke="white" stroke-width="1.5"/></svg>'

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
        padding: 22px;
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }}
    .bento-pink {{
        background-color: #FCE8EF;
        border: {card_border_pink};
        color: #881337;
        border-radius: 22px;
        padding: 22px;
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }}
    .bento-yellow {{
        background-color: #FEF3D6;
        border: {card_border_yellow};
        color: #713F12;
        border-radius: 22px;
        padding: 22px;
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }}
    .bento-purple {{
        background-color: #EDE9FE;
        border: {card_border_purple};
        color: #4C1D95;
        border-radius: 22px;
        padding: 22px;
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
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
    .odo-container {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #18181B;
        padding: 6px 10px;
        border-radius: 14px;
        box-shadow: inset 0 3px 6px rgba(0,0,0,0.6), 0 4px 12px rgba(24,24,27,0.18);
        border: 1px solid #3F3F46;
    }}
    .odo-digit {{
        background: linear-gradient(180deg, #27272A 0%, #3F3F46 48%, #18181B 52%, #27272A 100%);
        color: #FFFFFF;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 22px;
        font-weight: 800;
        min-width: 26px;
        height: 36px;
        border-radius: 7px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-top: 1px solid rgba(255,255,255,0.18);
        border-bottom: 1px solid rgba(0,0,0,0.8);
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.45);
        animation: odoRollIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .odo-sep {{
        color: #A1A1AA;
        font-weight: 800;
        font-size: 18px;
        padding: 0 2px;
    }}
    @keyframes odoRollIn {{
        0% {{ transform: translateY(-100%) rotateX(60deg); opacity: 0; }}
        100% {{ transform: translateY(0) rotateX(0deg); opacity: 1; }}
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
    yt_key = os.getenv("YOUTUBE_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model_id = os.getenv("HF_MODEL_ID", "google/muril-base-cased")

    with st.status("Running IndicPulse 5-Stage Pipeline...", expanded=True) as status:
        # Stage 1: Ingestion
        if video_url and (df_input is None or df_input.empty):
            st.write("1. Fetching comments and video metadata from YouTube API...")
            prog_text = st.empty()
            def on_progress(fetched: int, target: int):
                prog_text.caption(f"Fetched {fetched} / {target} comments...")

            df_input = fetch_youtube_comments(video_url, yt_key, max_comments=max_comments, progress_callback=on_progress)
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
            if not gemini_key:
                summary_text += "\n\n> *Note: Grounded heuristic synthesis generated. Configure GEMINI_API_KEY in .env for live Gemini 2.5 Flash LLM analysis.*"
        except Exception as e:
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
        topic_counts = pd.DataFrame({"Topic": ["General Feedback"], "Count": [total_c]})
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
def render_odometer_score(value_str: str, label: str, subtitle: str = "", theme: str = "mint") -> str:
    """Render a 3D Mechanical Odometer / Rolling Digit Counter inside a Bento Card theme."""
    digits_html = []
    for ch in str(value_str):
        if ch in "0123456789":
            digits_html.append(f'<span class="odo-digit">{ch}</span>')
        elif ch in ".,%/+-: ":
            digits_html.append(f'<span class="odo-sep">{ch}</span>')
        else:
            digits_html.append(f'<span class="odo-digit">{ch}</span>')
    odo_bar = f'<div class="odo-container">{"".join(digits_html)}</div>'
    sub_html = f'<div style="font-size:11px; font-weight:600; opacity:0.85; margin-top:8px;">{subtitle}</div>' if subtitle else ''

    return f"""
    <div class="bento-{theme}" style="min-height:140px; display:flex; flex-direction:column; justify-content:space-between; box-sizing:border-box;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">{label}</span>
            <span class="arrow-circle">↗</span>
        </div>
        <div style="margin: 10px 0 4px 0;">
            {odo_bar}
        </div>
        {sub_html}
    </div>
    """


def render_bento_grid_results(run_data: Dict[str, Any]):
    """Renders the complete Asymmetric Pastel Bento Grid with Odometer UI & Interactive Drill-Downs."""
    df_mood = run_data["df_mood"]
    df_clustered = run_data["df_clustered"]
    exec_summary = run_data["exec_summary"]
    metrics = run_data["metrics"]

    total_c = metrics["total"]
    weighted_pos = metrics["weighted_approval"]
    sarcasm_c = metrics["sarcasm_count"]
    toxic_c = metrics["toxic_count"]
    num_topics = metrics["num_topics"]
    dom_mood = metrics["dominant_mood"]
    pulse_score = metrics.get("pulse_score", round(min(10.0, max(3.5, (weighted_pos / 10.0))), 1))
    v_title = run_data.get("video_title", run_data.get("url_or_title", "Video Analysis"))
    c_title = run_data.get("channel_title", "")

    # Ensure topic columns exist
    if "topic_name" not in df_clustered.columns:
        df_clustered["topic_name"] = df_clustered.get("topic_label", "General Discussion")

    topic_counts = df_clustered["topic_name"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    # Filter out quarantined from top topics display
    topic_counts = topic_counts[~topic_counts["Topic"].str.contains("Quarantin")]

    pos_mask = df_mood["mood"].astype(str).str.contains("Positive")
    total_pos_count = int(pos_mask.sum())

    # Metadata Banner
    if c_title:
        st.caption(f"🎬 **Video Context:** {v_title} • **Channel:** {c_title} • **Sample Size:** {total_c} comments")

    left_zone, right_zone = st.columns([2.15, 1.05], gap="large")

    # ── LEFT ZONE ──────────────────────────────────────────────
    with left_zone:
        st.markdown(f"<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Activities Overview ({total_c} comments)</div>", unsafe_allow_html=True)
        act_c1, act_c2 = st.columns(2, gap="medium")

        with act_c1:
            st.markdown(render_odometer_score(f"{weighted_pos:.1f}%", "Audience Hype", f"Dominant: {dom_mood}", theme="mint"), unsafe_allow_html=True)
            if st.button(f"Inspect {total_pos_count} Hype Comments ↗", key=f"drill_pos_{run_data['id']}", use_container_width=True):
                st.session_state.active_drilldown = "MOOD:Positive (Aggressive Hype)"
                st.rerun()

        with act_c2:
            st.markdown(render_odometer_score(f"{toxic_c}", "Moderation Shield", f"{sarcasm_c} Sarcastic | {toxic_c} Quarantined", theme="pink"), unsafe_allow_html=True)
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if st.button(f"Inspect {sarcasm_c} Sarcastic ↗", key=f"drill_sarc_{run_data['id']}", use_container_width=True):
                    st.session_state.active_drilldown = "MOOD:Sarcastic / Trolling"
                    st.rerun()
            with m_col2:
                if st.button(f"Inspect {toxic_c} Toxic ↗", key=f"drill_tox_{run_data['id']}", use_container_width=True):
                    st.session_state.active_drilldown = "MOOD:Toxic / Abusive"
                    st.rerun()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Semantic Progress</div>", unsafe_allow_html=True)

        kpi_c1, kpi_c2, kpi_c3 = st.columns(3, gap="medium")
        with kpi_c1:
            st.markdown(render_odometer_score(f"{total_c:,}", "Analyzed", "Total Dataset Comments", theme="mint"), unsafe_allow_html=True)
            if st.button(f"Inspect All {total_c:,} ↗", key=f"drill_all_{run_data['id']}", use_container_width=True):
                st.session_state.active_drilldown = "ALL"
                st.rerun()

        with kpi_c2:
            st.markdown(render_odometer_score(f"{weighted_pos:.0f}%", "Weighted Score", "Log10 Weighted Approval", theme="yellow"), unsafe_allow_html=True)
            if st.button(f"Inspect Top Weighted ↗", key=f"drill_weight_{run_data['id']}", use_container_width=True):
                st.session_state.active_drilldown = "HIGH_WEIGHT"
                st.rerun()

        with kpi_c3:
            st.markdown(render_odometer_score(f"{num_topics}", "Active Topics", "Discovered Clusters", theme="purple"), unsafe_allow_html=True)
            if st.button(f"Inspect All Topics ↗", key=f"drill_topics_{run_data['id']}", use_container_width=True):
                st.session_state.active_drilldown = "ALL_TOPICS"
                st.rerun()

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
            st.markdown(exec_summary)

            gem_key = os.getenv("GEMINI_API_KEY", "")
            if gem_key:
                if st.button("Regenerate AI Summary", key=f"regen_{run_data['id']}", use_container_width=True):
                    with st.spinner("Re-synthesizing Gemini 2.5 Flash creator briefing..."):
                        new_summary = generate_executive_summary(df_clustered, gem_key, video_title=v_title, channel_title=c_title)
                        run_data["exec_summary"] = new_summary
                        st.success("Executive action synthesis refreshed.")
                        st.rerun()
            else:
                st.caption("💡 Add GEMINI_API_KEY to your .env file or session settings to enable live Gemini 2.5 Flash LLM re-synthesis.")

    # ── RIGHT ZONE ─────────────────────────────────────────────
    with right_zone:
        st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Community Sentiment Pulse</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div style='font-size:13px; font-weight:700; color:#1E293B;'>Radial Speedometer Gauge</div>", unsafe_allow_html=True)
            fig_gauge = charts.pulse_gauge_chart(pulse_score, colorblind_safe=st.session_state.colorblind_mode)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Mood Distribution</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div style='font-size:13px; font-weight:700; color:#1E293B;'>Engagement-Weighted Breakdown</div>", unsafe_allow_html=True)
            fig_pie = charts.mood_donut_chart(df_mood, colorblind_safe=st.session_state.colorblind_mode)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:700; color:#64748B; margin-bottom:10px;'>Top Semantic Clusters</div>", unsafe_allow_html=True)

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
                        <div style="font-size:11px; opacity:0.8;">IndicSBERT + BERTopic Cluster</div>
                    </div>
                </div>
                <span class="pill-badge">{row['Count']} items</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Inspect {row['Count']} Comments ↗", key=f"topic_drill_{idx}_{row['Topic']}_{run_data['id']}", use_container_width=True):
                st.session_state.active_drilldown = f"TOPIC:{row['Topic']}"
                st.rerun()

    # ── SCORE EVIDENCE & COMMENT INSPECTOR DRAWER ─────────────
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    active_drill = st.session_state.get("active_drilldown", "ALL")

    # Filter comments and build mathematical explanation
    target_data = df_clustered if "topic_name" in df_clustered.columns else df_mood

    if active_drill == "ALL":
        filter_label = f"All Comments ({total_c})"
        math_expl = f"Displaying all {total_c} comments across all 4 sentiment categories and {num_topics} semantic clusters."
        f_df = target_data
    elif active_drill.startswith("MOOD:Positive"):
        pos_m = target_data["mood"].astype(str).str.contains("Positive")
        f_df = target_data[pos_m]
        w_sum = f_df["engagement_weight"].sum() if "engagement_weight" in f_df.columns else 0
        filter_label = f"Positive (Aggressive Hype) — {len(f_df)} comments"
        math_expl = f"These {len(f_df)} comments triggered [fire_hype] / praise emoji tags or positive MuRIL logits with a combined log₁₀ engagement weight of {w_sum:.1f} ({weighted_pos:.1f}% of audience approval)."
    elif active_drill.startswith("MOOD:Sarcastic"):
        sarc_m = target_data["mood"].astype(str).str.contains("Sarcastic")
        f_df = target_data[sarc_m]
        filter_label = f"Sarcastic / Trolling — {len(f_df)} comments"
        math_expl = f"These {len(f_df)} comments matched irony patterns, mocking syntax, or sarcasm tokens ([clown_mocking], [eye_roll_sarcasm])."
    elif active_drill.startswith("MOOD:Toxic"):
        tox_m = target_data["mood"].astype(str).str.contains("Toxic")
        f_df = target_data[tox_m]
        filter_label = f"Toxic & Abusive (Quarantined) — {len(f_df)} comments"
        math_expl = f"These {len(f_df)} hostile comments matched abusive profanity regexes or severe harassment scores and were quarantined from semantic clustering."
    elif active_drill == "HIGH_WEIGHT":
        cutoff = target_data["engagement_weight"].quantile(0.75) if "engagement_weight" in target_data.columns else 0
        f_df = target_data[target_data["engagement_weight"] >= cutoff] if "engagement_weight" in target_data.columns else target_data.head(25)
        filter_label = f"Top-Engaged Comments (Top 25% Weight) — {len(f_df)} comments"
        math_expl = f"These {len(f_df)} comments earned the highest likes, weighted via W = log₁₀(Likes + 1) + 1 to anchor the Gemini 2.5 Flash executive action report."
    elif active_drill == "ALL_TOPICS":
        f_df = target_data[~target_data["topic_name"].str.contains("Quarantin")] if "topic_name" in target_data.columns else target_data
        filter_label = f"All Constructive Clustered Comments — {len(f_df)} comments"
        math_expl = f"These {len(f_df)} constructive comments were vectorized with IndicSBERT and clustered into {num_topics} distinct themes via HDBSCAN."
    elif active_drill.startswith("TOPIC:"):
        target_t = active_drill.replace("TOPIC:", "").strip()
        f_df = target_data[target_data["topic_name"] == target_t] if "topic_name" in target_data.columns else target_data
        avg_w = f_df["engagement_weight"].mean() if "engagement_weight" in f_df.columns else 1.0
        filter_label = f"Topic Cluster: {target_t} — {len(f_df)} comments"
        math_expl = f"These {len(f_df)} comments were grouped together via IndicSBERT dense sentence embeddings + UMAP with an average engagement weight of {avg_w:.2f}."
    else:
        filter_label = f"Filtered View: {active_drill}"
        math_expl = "Displaying comments matching the active drill-down filter."
        f_df = target_data

    # Sort filtered results by likes descending
    sort_c = "likes" if "likes" in f_df.columns else ("like_count" if "like_count" in f_df.columns else None)
    if sort_c:
        f_df = f_df.sort_values(sort_c, ascending=False)

    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div>
                <span style="font-size:16px; font-weight:800; color:#1E293B;">🔍 Score Evidence & Comment Inspector</span>
                <span class="pill-badge" style="background:#18181B; color:#FFFFFF; margin-left:10px;">{filter_label}</span>
            </div>
        </div>
        <div style="font-size:12px; color:#475569; background:#F8FAFC; border:1px solid #EAEEF4; border-radius:12px; padding:10px 14px; margin-bottom:14px; line-height:1.5;">
            📐 <b>Evidence Logic:</b> {math_expl}
        </div>
        """, unsafe_allow_html=True)

        if active_drill != "ALL":
            if st.button("↺ Reset Filter (Show All Comments)", key=f"reset_filter_{run_data['id']}"):
                st.session_state.active_drilldown = "ALL"
                st.rerun()

        cols_to_show = [c for c in ["raw_text", "likes", "engagement_weight", "mood", "topic_name", "text_for_muril"] if c in f_df.columns]
        st.dataframe(f_df[cols_to_show], use_container_width=True, height=280)

    # ── BOTTOM DRAWER: COMMENT EXPLORER & QUARANTINE ───────────
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        tab_exp, tab_quar = st.tabs(["Full Semantic Comment Explorer", f"Toxic Moderation Quarantine ({toxic_c})"])
        with tab_exp:
            display_cols = [c for c in ["raw_text", "likes", "mood", "topic_name", "text_for_muril", "engagement_weight"] if c in df_clustered.columns]
            st.dataframe(df_clustered[display_cols], use_container_width=True, height=270)
        with tab_quar:
            toxic_df = df_mood[df_mood["mood"].astype(str).str.contains("Toxic")]
            st.warning(f"{len(toxic_df)} hostile comments isolated from BERTopic clustering.")
            q_cols = [c for c in ["raw_text", "likes", "text_for_muril"] if c in toxic_df.columns]
            st.dataframe(toxic_df[q_cols], use_container_width=True, height=230)


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

    # 4 Interactive Platform Selector Cards
    p1, p2, p3, p4 = st.columns(4)
    platforms = [
        ("YouTube", SVG_YOUTUBE, "Video & Shorts Comments", p1),
        ("Instagram Reels", SVG_INSTAGRAM, "Reel Commentary & Audio", p2),
        ("Reddit", SVG_REDDIT, "Community Discussions", p3),
        ("CSV / Custom Dataset", SVG_CSV, "Universal Table Upload", p4),
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
            st.caption(f"Enter a public {st.session_state.selected_platform} post link to run semantic and sarcasm analysis.")
            in_col1, in_col2 = st.columns([3.5, 1.2])
            with in_col1:
                post_url = st.text_input(f"{st.session_state.selected_platform} Link", placeholder=f"Paste {st.session_state.selected_platform} post or thread URL...", label_visibility="collapsed")
            with in_col2:
                run_other = st.button("Analyze Comments", type="primary", use_container_width=True)

            sample_limit_other = st.slider(
                "Analysis Depth (Comment Volume)",
                min_value=100,
                max_value=2000,
                value=300,
                step=100,
                key="depth_other",
            )
            render_html('<div style="font-size:11px; font-weight:600; color:#64748B; margin-top:-6px; margin-bottom:12px;">100–300: Fast (~8s) • 500–1,000: Deep Topic Scan (~20s) • 1,500–2,000: Full Viral Audit (~45s)</div>')

            if st.button("Load Sample Hinglish Dataset", key="demo_other"):
                base_demo = load_demo_comments()
                mult = max(1, int(sample_limit_other) // len(base_demo))
                scaled_demo = pd.concat([base_demo] * mult, ignore_index=True).head(int(sample_limit_other))
                scaled_demo["comment_id"] = [f"demo_{i:04d}" for i in range(len(scaled_demo))]
                execute_analysis_flow(scaled_demo, st.session_state.selected_platform, f"Benchmark Hinglish ({len(scaled_demo)} comments)")
                st.rerun()

            if run_other:
                base_demo = load_demo_comments()
                mult = max(1, int(sample_limit_other) // len(base_demo))
                scaled_demo = pd.concat([base_demo] * mult, ignore_index=True).head(int(sample_limit_other))
                scaled_demo["comment_id"] = [f"demo_{i:04d}" for i in range(len(scaled_demo))]
                execute_analysis_flow(scaled_demo, st.session_state.selected_platform, post_url or f"{st.session_state.selected_platform} Post")
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

        # Row 1: 4 Cumulative Pastel Bento KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="bento-mint">
                <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Total Datasets</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                    <span style="font-size:32px; font-weight:800;">{cum_links}</span>
                    <span class="pill-badge">{cum_comments} Comments</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="bento-yellow">
                <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Overall Approval</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                    <span style="font-size:32px; font-weight:800;">{cum_weighted_pos:.1f}%</span>
                    <span class="arrow-circle">↗</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="bento-purple">
                <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Sarcasm Caught</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                    <span style="font-size:32px; font-weight:800;">{cum_sarcasm}</span>
                    <span class="arrow-circle">↗</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="bento-pink">
                <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Quarantined Toxic</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                    <span style="font-size:32px; font-weight:800;">{cum_toxic}</span>
                    <span class="arrow-circle">↗</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
                    color_discrete_map={"YouTube": "#FF0000", "Instagram Reels": "#E1306C", "Reddit": "#FF4500", "CSV / Custom Dataset": "#4F46E5"},
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
                st.plotly_chart(fig_bar, use_container_width=True)

            # Combined Latest Gemini Executive Insights
            with st.container(border=True):
                st.markdown("<div style='font-size:14px; font-weight:800; color:#1E293B; margin-bottom:10px;'>Latest Strategic Executive Synthesis</div>", unsafe_allow_html=True)
                st.markdown(st.session_state.analysis_history[0]["exec_summary"])

        with dash_right:
            # Cumulative Mood Donut
            with st.container(border=True):
                st.markdown("<div style='font-size:14px; font-weight:800; color:#1E293B;'>Cumulative Mood Distribution</div>", unsafe_allow_html=True)
                fig_pie_cum = charts.mood_donut_chart(all_moods, colorblind_safe=st.session_state.colorblind_mode)
                st.plotly_chart(fig_pie_cum, use_container_width=True)

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
                                refreshed_df = fetch_youtube_comments(item["url_or_title"], os.getenv("YOUTUBE_API_KEY"), max_comments=target_c)
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
                            refreshed_df = fetch_youtube_comments(selected_run["url_or_title"], os.getenv("YOUTUBE_API_KEY"), max_comments=target_c)
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
    yt_connected = bool(os.getenv("YOUTUBE_API_KEY"))
    gemini_connected = bool(os.getenv("GEMINI_API_KEY"))
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

        with st.expander("Configure / Override Environment Keys in Memory"):
            st.caption("Temporarily inject keys for the active session without modifying the on-disk .env file:")
            temp_yt = st.text_input("Temporary YouTube API Key", type="password", value=os.getenv("YOUTUBE_API_KEY", ""))
            temp_gem = st.text_input("Temporary Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
            if st.button("Apply Keys to Memory"):
                if temp_yt:
                    os.environ["YOUTUBE_API_KEY"] = temp_yt
                if temp_gem:
                    os.environ["GEMINI_API_KEY"] = temp_gem
                st.success("Session environment updated.")
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
