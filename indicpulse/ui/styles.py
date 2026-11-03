"""
IndicPulse – Soft Pastel Bento-Grid SaaS Design System v2.0
===========================================================
Complete User Profile Hub, Interactive Pipeline Settings, and
WCAG-Compliant Accessibility Controls (High-Contrast, Dyslexia, Motion, Scale).
"""

from typing import Optional, Dict


def get_custom_css(
    font_scale: float = 1.0,
    high_contrast: bool = False,
    dyslexia_font: bool = False,
    reduced_motion: bool = False,
) -> str:
    """
    Generate dynamic CSS with real-time accessibility overrides.

    Parameters
    ----------
    font_scale : float
        1.0 (100%), 1.15 (115%), 1.30 (130%)
    high_contrast : bool
        If True, replaces soft pastel borders with 2px solid #18181B borders and boosts text contrast (WCAG AAA 7:1+)
    dyslexia_font : bool
        If True, switches font to OpenDyslexic / Comic Sans with increased line height and letter spacing
    reduced_motion : bool
        If True, disables all CSS transitions, animations, and hover transforms
    """

    font_family = "'OpenDyslexic', 'Comic Sans MS', sans-serif" if dyslexia_font else "'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    letter_spacing = "0.06em" if dyslexia_font else "-0.01em"
    line_height = "1.68" if dyslexia_font else "1.5"

    # High Contrast Overrides
    if high_contrast:
        canvas_bg = "#FFFFFF"
        border_bento = "2px solid #18181B !important"
        text_mint = "#052e16 !important; font-weight: 800 !important"
        text_pink = "#4c0519 !important; font-weight: 800 !important"
        text_yellow = "#422006 !important; font-weight: 800 !important"
        text_purple = "#2e1065 !important; font-weight: 800 !important"
        focus_outline = "3px solid #18181B !important; outline-offset: 3px !important"
    else:
        canvas_bg = "#F4F6F9"
        border_bento = "initial"
        text_mint = "#0F5132"
        text_pink = "#881337"
        text_yellow = "#713F12"
        text_purple = "#4C1D95"
        focus_outline = "2px solid #18181B !important; outline-offset: 2px !important"

    # Reduced Motion Overrides
    motion_rules = """
    * {
        transition: none !important;
        animation: none !important;
        transform: none !important;
    }
    """ if reduced_motion else """
    .bento-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    .topic-pill-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(15, 81, 50, 0.06);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(24, 24, 27, 0.22);
    }
    """

    return f"""
<style>
/* ── GOOGLE FONTS ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* ── DESIGN SYSTEM TOKENS ─────────────────────────────── */
:root {{
    --bg-canvas: {canvas_bg};
    --bg-white: #FFFFFF;
    --border-subtle: #EAEEF4;
    --border-card: #E2E8F0;
    
    /* 4 Signature Pastel Themes */
    --mint-bg: #E2F6EE;
    --mint-border: #CCEFE2;
    --mint-text: {text_mint};

    --pink-bg: #FCE8EF;
    --pink-border: #F9D1DF;
    --pink-text: {text_pink};

    --yellow-bg: #FEF3D6;
    --yellow-border: #FCE4AD;
    --yellow-text: {text_yellow};

    --purple-bg: #EDE9FE;
    --purple-border: #DDD6FE;
    --purple-text: {text_purple};

    /* Primary Accent */
    --charcoal: #18181B;
    --charcoal-hover: #27272A;
    --slate-heading: #1E293B;
    --slate-muted: #64748B;
    --slate-light: #94A3B8;

    --radius-pill: 999px;
    --radius-bento: 22px;
    --radius-inner: 16px;
    --radius-sm: 10px;

    --font-scale: {font_scale};
}}

/* ── GLOBAL RESET & ROOT SCALING ──────────────────────── */
html, body, [class*="css"] {{
    font-size: calc(15px * var(--font-scale)) !important;
}}

.stApp {{
    background-color: var(--bg-canvas) !important;
    font-family: {font_family} !important;
    letter-spacing: {letter_spacing} !important;
    line-height: {line_height} !important;
    color: var(--slate-heading) !important;
}}

.main .block-container {{
    padding: 1.25rem 2.5rem 3rem 2.5rem !important;
    max-width: 1440px !important;
}}

/* ── KEYBOARD ACCESSIBILITY & FOCUS-VISIBLE ───────────── */
*:focus-visible {{
    outline: {focus_outline};
}}

/* ── MOTION RULES ─────────────────────────────────────── */
{motion_rules}

/* ── SIDEBAR NAVIGATION RAIL ──────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: var(--bg-white) !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.01) !important;
}}

section[data-testid="stSidebar"] > div {{
    padding: 1.25rem 1.1rem !important;
}}

/* Navigation Radio styling matching dark charcoal active pills */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    gap: 0.35rem;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
    background-color: #FAFAFC;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.55rem 0.85rem;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--slate-heading);
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
    background-color: #F1F5F9;
    border-color: var(--border-card);
}}

/* Active state for navigation rail item */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
    background-color: var(--charcoal) !important;
    color: #FFFFFF !important;
    border-color: var(--charcoal) !important;
    box-shadow: 0 4px 12px rgba(24, 24, 27, 0.18) !important;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}

/* ── CHARCOAL PRIMARY BUTTONS ─────────────────────────── */
.stButton > button {{
    background-color: var(--charcoal) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    font-family: {font_family} !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 12px rgba(24, 24, 27, 0.15) !important;
}}

/* ── SECONDARY WHITE / DOWNLOAD BUTTONS ───────────────── */
div[data-testid="stDownloadButton"] > button {{
    background-color: #FFFFFF !important;
    color: var(--charcoal) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}}

div[data-testid="stDownloadButton"] > button:hover {{
    background-color: #F8FAFC !important;
    border-color: var(--charcoal) !important;
    color: var(--charcoal) !important;
}}

/* ── TOP UTILITY & USER PROFILE HEADER BAR ────────────── */
.top-header-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.75rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--border-subtle);
}}

.creator-greeting {{
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--slate-heading);
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.2;
}}

.creator-role-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background-color: #EDE9FE;
    color: #4C1D95;
    border: 1px solid #DDD6FE;
    border-radius: var(--radius-pill);
    padding: 0.2rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-left: 0.5rem;
}}

.header-utility-cluster {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}

.notification-badge-btn {{
    background-color: #FFFFFF;
    border: 1px solid var(--border-subtle);
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    position: relative;
    cursor: pointer;
}}

.notification-dot {{
    position: absolute;
    top: 6px;
    right: 6px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #F43F5E;
    border: 1.5px solid #FFFFFF;
}}

.user-avatar-pill {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background-color: #FFFFFF;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    padding: 0.35rem 0.85rem 0.35rem 0.45rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}}

.avatar-circle {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
}}

/* ── 4 SIGNATURE PASTEL BENTO CARD CLASSES ────────────── */
.bento-card {{
    border-radius: var(--radius-bento);
    padding: 1.5rem 1.6rem;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
}}

/* 1. Mint Sage Green Card */
.bento-mint {{
    background-color: var(--mint-bg);
    border: {border_bento or "1px solid var(--mint-border)"};
    color: var(--mint-text);
    box-shadow: 0 4px 18px rgba(15, 81, 50, 0.04);
}}

/* 2. Blush Pastel Pink Card */
.bento-pink {{
    background-color: var(--pink-bg);
    border: {border_bento or "1px solid var(--pink-border)"};
    color: var(--pink-text);
    box-shadow: 0 4px 18px rgba(136, 19, 55, 0.04);
}}

/* 3. Warm Buttercream Yellow Card */
.bento-yellow {{
    background-color: var(--yellow-bg);
    border: {border_bento or "1px solid var(--yellow-border)"};
    color: var(--yellow-text);
    box-shadow: 0 4px 18px rgba(113, 63, 18, 0.04);
}}

/* 4. Soft Lavender Purple Card */
.bento-purple {{
    background-color: var(--purple-bg);
    border: {border_bento or "1px solid var(--purple-border)"};
    color: var(--purple-text);
    box-shadow: 0 4px 18px rgba(76, 29, 149, 0.04);
}}

/* 5. Pure White Container Card */
.bento-white {{
    background-color: var(--bg-white);
    border: {border_bento or "1px solid var(--border-subtle)"};
    border-radius: var(--radius-bento);
    padding: 1.6rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}}

/* ── MICRO-BADGES & ACCENT ICONS ──────────────────────── */
.bento-pill-white {{
    background-color: #FFFFFF;
    border-radius: var(--radius-pill);
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--charcoal);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}}

.bento-icon-btn {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #FFFFFF;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    color: var(--charcoal);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    flex-shrink: 0;
}}

.emoji-cluster-avatars {{
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 1.35rem;
    margin-top: 0.75rem;
}}

/* ── PROGRESS BAR STYLING ─────────────────────────────── */
.bento-progress {{
    width: 100%;
    height: 6px;
    background-color: rgba(0, 0, 0, 0.06);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 1rem;
}}

.bento-progress-fill {{
    height: 100%;
    border-radius: 4px;
    background-color: var(--charcoal);
}}

/* ── RIGHT COLUMN TOPIC PILLS ─────────────────────────── */
.topic-pill-card {{
    background-color: var(--mint-bg);
    border: 1px solid var(--mint-border);
    border-radius: 18px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.topic-pill-left {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

.topic-circle-icon {{
    width: 32px;
    height: 32px;
    background-color: #FFFFFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
}}

.topic-name {{
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--mint-text);
}}

.topic-count-pill {{
    background-color: #FFFFFF;
    border-radius: var(--radius-pill);
    padding: 0.25rem 0.65rem;
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--mint-text);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
}}

/* ── SECTION LABELS ───────────────────────────────────── */
.section-kicker {{
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--slate-muted);
    margin-bottom: 0.85rem;
}}

/* ── TAB & COMMENT ROW STYLING ────────────────────────── */
div[data-testid="stTabs"] button {{
    background: transparent !important;
    color: var(--slate-muted) !important;
    border: none !important;
    font-family: {font_family} !important;
    font-weight: 600 !important;
    font-size: 0.94rem !important;
    padding: 0.6rem 1.2rem !important;
}}

div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: var(--charcoal) !important;
    border-bottom: 2px solid var(--charcoal) !important;
}}

.quarantine-container {{
    background-color: var(--pink-bg);
    border: 1px solid var(--pink-border);
    border-radius: 20px;
    padding: 1.5rem;
    color: var(--pink-text);
}}

.comment-row-card {{
    background-color: #FFFFFF;
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 0.6rem;
}}

/* Mood Badges */
.mood-badge-mint {{
    background-color: var(--mint-bg);
    color: var(--mint-text);
    border: 1px solid var(--mint-border);
    border-radius: var(--radius-pill);
    padding: 0.2rem 0.65rem;
    font-size: 0.74rem;
    font-weight: 700;
}}

.mood-badge-pink {{
    background-color: var(--pink-bg);
    color: var(--pink-text);
    border: 1px solid var(--pink-border);
    border-radius: var(--radius-pill);
    padding: 0.2rem 0.65rem;
    font-size: 0.74rem;
    font-weight: 700;
}}

.mood-badge-purple {{
    background-color: var(--purple-bg);
    color: var(--purple-text);
    border: 1px solid var(--purple-border);
    border-radius: var(--radius-pill);
    padding: 0.2rem 0.65rem;
    font-size: 0.74rem;
    font-weight: 700;
}}

.mood-badge-gray {{
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
    border-radius: var(--radius-pill);
    padding: 0.2rem 0.65rem;
    font-size: 0.74rem;
    font-weight: 700;
}}

/* Pastel Switch Cards in System Functions */
.switch-bento-card {{
    border-radius: 18px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
</style>
"""


def render_header_bar(user_name: str = "Creator", channel_handle: str = "IndicPulse Studio"):
    """Render top header bar with user profile details, status, and alerts."""
    import streamlit as st
    st.markdown(f"""
    <div class="top-header-bar" role="region" aria-label="Dashboard Top Navigation and User Profile">
        <div>
            <div style="display:flex; align-items:center;">
                <h1 class="creator-greeting">Welcome back, {user_name} 👋</h1>
                <span class="creator-role-badge">Creator Tier • {channel_handle}</span>
            </div>
            <div style="font-size:0.86rem; color:#64748B; margin-top:0.3rem;">
                Hinglish Neural Pipeline Active • Real-Time Multilingual Analytics
            </div>
        </div>
        <div class="header-utility-cluster">
            <div style="background:#FFFFFF; border:1px solid #EAEEF4; border-radius:999px; padding:0.4rem 0.9rem; font-size:0.82rem; color:#0F5132; font-weight:700; display:flex; align-items:center; gap:0.4rem; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
                <span>🟢</span>
                <span>MuRIL & IndicSBERT Ready</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def mood_badge_html(label: str) -> str:
    """Return HTML for a pastel mood badge pill."""
    l_lower = label.lower()
    if "pos" in l_lower:
        return f'<span class="mood-badge-mint">🔥 Positive</span>'
    elif "sarc" in l_lower:
        return f'<span class="mood-badge-purple">🤡 Sarcastic</span>'
    elif "tox" in l_lower:
        return f'<span class="mood-badge-pink">🚫 Toxic</span>'
    else:
        return f'<span class="mood-badge-gray">💬 Neutral</span>'
