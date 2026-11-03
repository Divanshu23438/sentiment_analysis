"""
IndicPulse – Pastel Plotly Chart Components with Colorblind-Safe Overrides
===========================================================================
Interactive Plotly charts styled for the Soft Pastel Bento-Grid SaaS layout.
Includes scientific Okabe-Ito colorblind-safe palette support.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

# ── STANDARD PASTEL PALETTE ──────────────────────────────────
PASTEL_MOOD_COLORS = {
    # Full aliases from pipeline and MuRIL
    "Positive (Aggressive Hype)": "#10B981",          # Mint Emerald Green
    "Positive (Genuine Praise / Hype)": "#10B981",
    "Positive": "#10B981",
    "Sarcastic / Trolling": "#8B5CF6",                # Soft Lavender
    "Sarcastic": "#8B5CF6",
    "Neutral": "#94A3B8",                            # Cool Slate Gray
    "Neutral / Informational": "#94A3B8",
    "Toxic / Abusive": "#F43F5E",                     # Blush Rose / Red
    "Toxic": "#F43F5E",
}

# ── OKABE-ITO COLORBLIND-SAFE PALETTE (WCAG Compliant) ───────
# Validated for Protanopia, Deuteranopia, and Tritanopia
OKABE_ITO_MOOD_COLORS = {
    # Full aliases
    "Positive (Aggressive Hype)": "#0072B2",          # Blue
    "Positive (Genuine Praise / Hype)": "#0072B2",
    "Positive": "#0072B2",
    "Sarcastic / Trolling": "#E69F00",                # Orange
    "Sarcastic": "#E69F00",
    "Neutral": "#94A3B8",                            # Slate Gray
    "Neutral / Informational": "#94A3B8",
    "Toxic / Abusive": "#D55E00",                     # Vermilion
    "Toxic": "#D55E00",
}

# ── TOPIC CLUSTER COLOR PALETTES ──────────────────────────────
PASTEL_TOPIC_COLORS = [
    "#10B981",  # Mint Emerald
    "#8B5CF6",  # Soft Lavender
    "#F59E0B",  # Warm Amber
    "#3B82F6",  # Sky Blue
    "#EC4899",  # Soft Rose Pink
    "#14B8A6",  # Teal Cyan
    "#6366F1",  # Indigo
]

OKABE_ITO_TOPIC_COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
]

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#64748B", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)


def get_mood_color_map(colorblind_safe: bool = False) -> Dict[str, str]:
    return OKABE_ITO_MOOD_COLORS if colorblind_safe else PASTEL_MOOD_COLORS


def resolve_mood_color(label: Any, colorblind_safe: bool = False) -> str:
    """Resolve distinct mood colors supporting full, partial, and lowercase strings."""
    label_str = str(label).strip()
    color_map = OKABE_ITO_MOOD_COLORS if colorblind_safe else PASTEL_MOOD_COLORS

    if label_str in color_map:
        return color_map[label_str]

    lbl_lower = label_str.lower()
    if any(k in lbl_lower for k in ["pos", "praise", "hype", "good", "love"]):
        return color_map["Positive"]
    elif any(k in lbl_lower for k in ["sarcas", "troll", "irony"]):
        return color_map["Sarcastic"]
    elif any(k in lbl_lower for k in ["tox", "abus", "hate", "hostil", "insult"]):
        return color_map["Toxic"]
    elif any(k in lbl_lower for k in ["neut", "info"]):
        return color_map["Neutral"]

    # Deterministic fallback cycle if label is novel
    fallback_palette = OKABE_ITO_TOPIC_COLORS if colorblind_safe else PASTEL_TOPIC_COLORS
    return fallback_palette[abs(hash(label_str)) % len(fallback_palette)]


def mood_donut_chart(mood_counts: Any, colorblind_safe: bool = False) -> go.Figure:
    """Engagement-Weighted Mood Donut Chart with sleek modern ring styling."""
    if isinstance(mood_counts, pd.DataFrame):
        col = "mood" if "mood" in mood_counts.columns else "mood_label"
        if col in mood_counts.columns:
            mood_counts = mood_counts[col].value_counts().to_dict()
        else:
            mood_counts = {}
    elif isinstance(mood_counts, pd.Series):
        mood_counts = mood_counts.to_dict()
    elif not isinstance(mood_counts, dict):
        mood_counts = dict(mood_counts)

    labels = list(mood_counts.keys())
    values = list(mood_counts.values())
    colors = [resolve_mood_color(l, colorblind_safe=colorblind_safe) for l in labels]

    total = sum(values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.72,
                marker=dict(
                    colors=colors,
                    line=dict(color="#FFFFFF", width=3),
                ),
                textinfo="percent",
                textfont=dict(size=11, color="#1E293B", family="Plus Jakarta Sans"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        **CHART_THEME,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
            font=dict(color="#475569", size=11, family="Plus Jakarta Sans"),
            bgcolor="rgba(0,0,0,0)",
        ),
        annotations=[
            dict(
                text=f"<span style='font-size:24px; font-weight:800; color:#1E293B;'>{total}</span><br><span style='font-size:11px; font-weight:600; color:#94A3B8;'>COMMENTS</span>",
                x=0.5,
                y=0.5,
                font=dict(family="Plus Jakarta Sans"),
                showarrow=False,
            )
        ],
        height=260,
    )
    return fig


def engagement_bar_chart(topic_data: Any, colorblind_safe: bool = False) -> go.Figure:
    """Horizontal bar chart: topic clusters by avg engagement weight."""
    if isinstance(topic_data, pd.DataFrame):
        topic_summary = {}
        if "topic_id" in topic_data.columns:
            for tid in topic_data["topic_id"].unique():
                cluster = topic_data[topic_data["topic_id"] == tid]
                lbl = cluster["topic_name"].iloc[0] if "topic_name" in cluster.columns else (cluster["topic_label"].iloc[0] if "topic_label" in cluster.columns else f"Topic {tid}")
                topic_summary[int(tid)] = {
                    "label": lbl,
                    "size": len(cluster),
                    "avg_engagement": cluster["engagement_weight"].mean() if "engagement_weight" in cluster.columns else 1.0,
                }
        else:
            topic_summary = {}
    elif isinstance(topic_data, dict):
        topic_summary = topic_data
    else:
        topic_summary = {}

    if not topic_summary:
        fig = go.Figure()
        fig.update_layout(**CHART_THEME, title="No topics found")
        return fig

    data = [
        (info["label"], info["size"], info.get("avg_engagement", 1))
        for tid, info in topic_summary.items()
        if tid != -1
    ]
    if not data:
        fig = go.Figure()
        fig.update_layout(**CHART_THEME)
        return fig

    data.sort(key=lambda x: x[2], reverse=True)
    labels, sizes, weights = zip(*data)

    topic_palette = OKABE_ITO_TOPIC_COLORS if colorblind_safe else PASTEL_TOPIC_COLORS
    bar_colors = [topic_palette[i % len(topic_palette)] for i in range(len(labels))]

    fig = go.Figure(
        data=[
            go.Bar(
                y=list(labels),
                x=list(weights),
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    opacity=0.9,
                    line=dict(color="#FFFFFF", width=1.5),
                    cornerradius=8,
                ),
                text=[f"{s} comments" for s in sizes],
                textposition="inside",
                textfont=dict(color="#FFFFFF", size=11, family="Plus Jakarta Sans"),
                hovertemplate="<b>%{y}</b><br>Avg Weight: %{x:.2f}<br>%{text}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        **CHART_THEME,
        xaxis=dict(
            showgrid=True,
            gridcolor="#EAEEF4",
            title="Avg Engagement Weight (log₁₀ scale)",
            color="#64748B",
            tickfont=dict(family="Plus Jakarta Sans", size=11),
        ),
        yaxis=dict(color="#1E293B", autorange="reversed", tickfont=dict(family="Plus Jakarta Sans", size=11)),
        height=max(180, len(labels) * 44 + 60),
    )
    return fig


def mood_timeline_chart(df: pd.DataFrame, colorblind_safe: bool = False) -> go.Figure:
    """Line chart: mood distribution over time with pastel fills."""
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        fig = go.Figure()
        fig.update_layout(**CHART_THEME, title="No timestamp data available")
        return fig

    df_t = df.copy()
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], errors="coerce")
    df_t = df_t.dropna(subset=["timestamp"])
    if df_t.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_THEME, title="No timestamp data available")
        return fig

    df_t["date"] = df_t["timestamp"].dt.date
    col = "mood" if "mood" in df_t.columns else ("mood_label" if "mood_label" in df_t.columns else None)
    if not col:
        fig = go.Figure()
        fig.update_layout(**CHART_THEME, title="No mood data available")
        return fig

    timeline = df_t.groupby(["date", col]).size().reset_index(name="count")

    fig = go.Figure()
    for mood in timeline[col].unique():
        color = resolve_mood_color(mood, colorblind_safe=colorblind_safe)
        mood_data = timeline[timeline[col] == mood]
        if mood_data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=mood_data["date"],
                y=mood_data["count"],
                name=str(mood),
                line=dict(color=color, width=2.5),
                mode="lines+markers",
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{mood}</b><br>Date: %{{x}}<br>Comments: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        **CHART_THEME,
        xaxis=dict(showgrid=False, color="#64748B", tickfont=dict(family="Plus Jakarta Sans")),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EAEEF4",
            color="#64748B",
            tickfont=dict(family="Plus Jakarta Sans"),
        ),
        legend=dict(orientation="h", y=-0.2, font=dict(color="#475569", family="Plus Jakarta Sans")),
        height=240,
    )
    return fig


def pulse_gauge_chart(score: float, colorblind_safe: bool = False) -> go.Figure:
    """Gauge indicator for sentiment pulse score."""
    if colorblind_safe:
        color = "#0072B2" if score >= 7.0 else ("#E69F00" if score >= 5.0 else "#D55E00")
        step_low = "#FCE8EF"
        step_mid = "#FEF3D6"
        step_high = "#E0F2FE"
    else:
        color = "#10B981" if score >= 7.0 else ("#F59E0B" if score >= 5.0 else "#F43F5E")
        step_low = "#FCE8EF"
        step_mid = "#FEF3D6"
        step_high = "#E2F6EE"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(
                font=dict(size=36, color="#1E293B", family="Plus Jakarta Sans"),
                suffix="/10",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 10],
                    tickwidth=1,
                    tickcolor="#CBD5E1",
                    tickfont=dict(color="#64748B", size=10, family="Plus Jakarta Sans"),
                ),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 4.5], color=step_low),
                    dict(range=[4.5, 7.0], color=step_mid),
                    dict(range=[7.0, 10], color=step_high),
                ],
            ),
        )
    )

    fig.update_layout(
        **CHART_THEME,
        height=220,
    )
    return fig


def topic_scatter_chart(df: pd.DataFrame, topic_summary: Optional[Dict] = None, colorblind_safe: bool = False) -> go.Figure:
    """Scatter plot of topics: Size vs Engagement."""
    if topic_summary is None:
        topic_summary = {}
        if "topic_id" in df.columns:
            for tid in df["topic_id"].unique():
                cluster = df[df["topic_id"] == tid]
                lbl = cluster["topic_name"].iloc[0] if "topic_name" in cluster.columns else (cluster["topic_label"].iloc[0] if "topic_label" in cluster.columns else f"Topic {tid}")
                topic_summary[int(tid)] = {
                    "label": lbl,
                    "size": len(cluster),
                    "avg_engagement": cluster["engagement_weight"].mean() if "engagement_weight" in cluster.columns else 1.0,
                }
    if not topic_summary:
        return go.Figure()

    rows = []
    for tid, info in topic_summary.items():
        if tid == -1:
            continue
        rows.append(
            {
                "label": info["label"],
                "size": info["size"],
                "engagement": info.get("avg_engagement", 1),
                "topic_id": tid,
            }
        )

    if not rows:
        return go.Figure()

    tdf = pd.DataFrame(rows)
    color_scale = ["#BAE6FD", "#38BDF8", "#0284C7", "#0369A1"] if colorblind_safe else ["#CCEFE2", "#34D399", "#10B981", "#0F5132"]

    fig = px.scatter(
        tdf,
        x="engagement",
        y="size",
        size="size",
        color="engagement",
        text="label",
        color_continuous_scale=color_scale,
        size_max=45,
        hover_data={"label": True, "size": True, "engagement": ":.2f"},
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=11, color="#1E293B", family="Plus Jakarta Sans"),
        marker=dict(line=dict(width=1, color="#FFFFFF")),
    )

    fig.update_layout(
        **CHART_THEME,
        xaxis=dict(
            title="Avg Engagement Weight",
            showgrid=True,
            gridcolor="#EAEEF4",
            color="#64748B",
        ),
        yaxis=dict(
            title="Comment Count",
            showgrid=True,
            gridcolor="#EAEEF4",
            color="#64748B",
        ),
        coloraxis_showscale=False,
        height=320,
    )
    return fig


def profanity_severity_chart(df: pd.DataFrame, colorblind_safe: bool = False) -> go.Figure:
    """Bar chart of profanity severity distribution."""
    if "profanity_severity" not in df.columns:
        return go.Figure()

    sev_counts = df[df["has_profanity"] == True]["profanity_severity"].value_counts()
    order = ["low", "medium", "high"]
    
    if colorblind_safe:
        colors = {"low": "#FEF3D6", "medium": "#FED7AA", "high": "#FFEDD5"}
        border_colors = {"low": "#E69F00", "medium": "#D55E00", "high": "#9A3412"}
    else:
        colors = {"low": "#FEF3D6", "medium": "#FDE68A", "high": "#FCE8EF"}
        border_colors = {"low": "#FCE4AD", "medium": "#F59E0B", "high": "#F9D1DF"}

    vals = [sev_counts.get(s, 0) for s in order]
    cols = [colors[s] for s in order]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[s.capitalize() for s in order],
                y=vals,
                marker=dict(color=cols, line=dict(color=[border_colors[s] for s in order], width=1.5), cornerradius=6),
                text=vals,
                textposition="auto",
                textfont=dict(color="#1E293B", family="Plus Jakarta Sans", size=12),
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        **CHART_THEME,
        xaxis=dict(color="#64748B", tickfont=dict(family="Plus Jakarta Sans")),
        yaxis=dict(showgrid=True, gridcolor="#EAEEF4", color="#64748B", tickfont=dict(family="Plus Jakarta Sans")),
        height=200,
        showlegend=False,
    )
    return fig
