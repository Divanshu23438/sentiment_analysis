"""
IndicPulse – Step 1: Universal Ingestion Engine
================================================
Supports two ingestion modes:
  1. YouTube Data API v3  – fetches comments, author, timestamp, like_count
  2. Universal CSV Parser – accepts any CSV with flexible column mapping

Output schema (unified DataFrame):
  comment_id | author | text | timestamp | like_count | source
"""

import os
import re
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Union, Callable
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# YOUTUBE INGESTION
# ──────────────────────────────────────────────

def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return raw ID."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pat in patterns:
        match = re.search(pat, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract video ID from: {url_or_id}")


def fetch_youtube_comments(
    video_url_or_id: str,
    api_key: Optional[str] = None,
    max_comments: int = 500,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Fetch top-level comments from a YouTube video via the Data API v3.
    Supports scaling up to 2,000 comments via automatic pagination.

    Parameters
    ----------
    video_url_or_id : str
        Full YouTube URL or bare video ID.
    api_key : str, optional
        YouTube Data API v3 key. Falls back to YOUTUBE_API_KEY env var.
    max_comments : int
        Maximum number of comments to retrieve (default 500, up to 2000).
    progress_callback : Callable[[int, int], None], optional
        Callback invoked after each page: progress_callback(fetched_so_far, total_target).

    Returns
    -------
    pd.DataFrame
        Unified schema: comment_id, author, text, timestamp, like_count, source
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        raise ImportError(
            "google-api-python-client not installed. "
            "Run: pip install google-api-python-client"
        )

    key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise ValueError(
            "YouTube API key required. Set YOUTUBE_API_KEY in .env or pass api_key="
        )

    video_id = _extract_video_id(video_url_or_id)
    youtube = build("youtube", "v3", developerKey=key)

    # Fetch Real YouTube Video Metadata (Title, Channel, Description)
    video_title = f"YouTube Video ({video_id})"
    channel_title = "YouTube Channel"
    video_description = ""
    try:
        vid_req = youtube.videos().list(part="snippet,statistics", id=video_id)
        vid_resp = vid_req.execute()
        v_items = vid_resp.get("items", [])
        if v_items:
            snip = v_items[0].get("snippet", {})
            video_title = snip.get("title", video_title)
            channel_title = snip.get("channelTitle", channel_title)
            video_description = snip.get("description", "")[:300]
            logger.info(f"Retrieved video metadata: '{video_title}' by {channel_title}")
    except Exception as e:
        logger.warning(f"Could not fetch video metadata for {video_id}: {e}")

    rows, next_page = [], None
    fetched = 0

    logger.info(f"Fetching up to {max_comments} comments for video: {video_id} ('{video_title}')")

    while fetched < max_comments:
        batch = min(100, max_comments - fetched)
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch,
                pageToken=next_page,
                textFormat="plainText",
                order="relevance",
            )
            resp = req.execute()
        except HttpError as e:
            if e.resp.status == 403 and "commentsDisabled" in str(e):
                logger.warning(f"Comments are disabled for video: {video_id}")
                return _empty_schema()
            logger.error(f"YouTube API HttpError: {e}")
            raise
        except Exception as e:
            logger.error(f"YouTube API error: {e}")
            raise

        items = resp.get("items", [])
        if not items:
            break

        for item in items:
            snip = item["snippet"]["topLevelComment"]["snippet"]
            rows.append(
                {
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author": snip.get("authorDisplayName", "Anonymous"),
                    "text": snip.get("textDisplay", ""),
                    "timestamp": snip.get("publishedAt", ""),
                    "like_count": snip.get("likeCount", 0),
                    "source": "youtube",
                }
            )

        fetched += len(items)
        if progress_callback:
            try:
                progress_callback(fetched, max_comments)
            except Exception:
                pass

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No comments retrieved from YouTube.")
        return _empty_schema()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    
    # Attach video metadata to dataframe attributes
    df.attrs["video_title"] = video_title
    df.attrs["channel_title"] = channel_title
    df.attrs["video_description"] = video_description

    logger.info(f"Successfully fetched {len(df)} YouTube comments for '{video_title}'.")
    out = df[["comment_id", "author", "text", "timestamp", "like_count", "source"]]
    out.attrs["video_title"] = video_title
    out.attrs["channel_title"] = channel_title
    out.attrs["video_description"] = video_description
    return out


# ──────────────────────────────────────────────
# CSV INGESTION
# ──────────────────────────────────────────────

_CSV_COLUMN_ALIASES = {
    "text": ["text", "comment", "review", "body", "content", "message", "Comment", "Review", "Text"],
    "author": ["author", "user", "username", "name", "reviewer", "Author", "User"],
    "timestamp": ["timestamp", "date", "created_at", "time", "published_at", "Date"],
    "like_count": ["like_count", "likes", "thumbs_up", "upvotes", "helpful", "Likes"],
    "comment_id": ["comment_id", "id", "review_id", "ID"],
}


def _resolve_column(df_cols: list, aliases: list) -> Optional[str]:
    """Return the first alias that exists in df_cols (case-insensitive)."""
    col_map = {c.lower(): c for c in df_cols}
    for alias in aliases:
        if alias.lower() in col_map:
            return col_map[alias.lower()]
    return None


def load_csv_comments(
    filepath: Union[str, Path],
    text_col: Optional[str] = None,
    author_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    like_col: Optional[str] = None,
    id_col: Optional[str] = None,
    source_label: str = "csv",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load comments from any CSV file into the unified IndicPulse schema.

    Auto-detects column names using common aliases. Override with explicit params.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file.
    text_col, author_col, etc. : str, optional
        Explicit column name overrides.
    source_label : str
        Tag for the 'source' column (e.g. 'instagram', 'reddit').
    encoding : str
        File encoding (default utf-8, falls back to latin-1).

    Returns
    -------
    pd.DataFrame  – unified schema
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    cols = list(df.columns)

    # Resolve columns
    t_col = text_col or _resolve_column(cols, _CSV_COLUMN_ALIASES["text"])
    a_col = author_col or _resolve_column(cols, _CSV_COLUMN_ALIASES["author"])
    ts_col = timestamp_col or _resolve_column(cols, _CSV_COLUMN_ALIASES["timestamp"])
    l_col = like_col or _resolve_column(cols, _CSV_COLUMN_ALIASES["like_count"])
    id_c = id_col or _resolve_column(cols, _CSV_COLUMN_ALIASES["comment_id"])

    if t_col is None:
        raise ValueError(
            f"Cannot detect text column. Available columns: {cols}. "
            "Pass text_col='your_column_name'."
        )

    out = pd.DataFrame()
    out["comment_id"] = df[id_c].astype(str) if id_c else [f"csv_{i}" for i in range(len(df))]
    out["author"] = df[a_col].fillna("Anonymous") if a_col else "Anonymous"
    out["text"] = df[t_col].fillna("").astype(str)
    out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.NaT
    out["like_count"] = (
        pd.to_numeric(df[l_col], errors="coerce").fillna(0).astype(int) if l_col else 0
    )
    out["source"] = source_label

    # Drop empty text rows
    out = out[out["text"].str.strip().astype(bool)].reset_index(drop=True)
    out.attrs["video_title"] = f"CSV Dataset: {path.name}"
    out.attrs["channel_title"] = source_label
    out.attrs["video_description"] = f"Imported from {path.name}"
    logger.info(f"Loaded {len(out)} comments from CSV: {path.name}")
    return out


def _empty_schema() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["comment_id", "author", "text", "timestamp", "like_count", "source"]
    )


def load_demo_comments() -> pd.DataFrame:
    """
    Return a built-in demo dataset of 30 realistic Hinglish / code-mixed comments
    for testing the full pipeline without requiring API keys.
    """
    demo_data = [
        {"text": "Bhai yeh video ekdum fire hai 🔥🔥 audio quality bahut badhiya", "like_count": 342},
        {"text": "kya bakwaas content hai yaar, time waste kar diya mera 😡", "like_count": 12},
        {"text": "Haha sahi keh raha hai, jaise yeh koi expert hai 😂😂 zyaada mat socho", "like_count": 89},
        {"text": "Bro yeh pricing toh bilkul galat hai, itna mehenga kyun", "like_count": 156},
        {"text": "Editing kaafi smooth hai, transitions ekdum clean 👌", "like_count": 201},
        {"text": "ch*tiya content creator hai yeh toh, bekaar hai sab", "like_count": 5},
        {"text": "Baaaahut badiya video bhai ❤️ keep it up", "like_count": 432},
        {"text": "Audio thoda low hai kuch kuch jagah par, baaki sab theek", "like_count": 78},
        {"text": "Haan haan bilkul, jaise yeh khud billionaire hai 🤡🤡", "like_count": 234},
        {"text": "Product quality se bahut khush hoon, packaging bhi acha tha", "like_count": 189},
        {"text": "Yeh thumbnail se content match nahi karta, clickbait lagta hai", "like_count": 93},
        {"text": "g**nd marao aise videos ki, total ghatiya", "like_count": 3},
        {"text": "Subscriber ho gaya bhai, agar aisa content milta raha toh 🙌", "like_count": 521},
        {"text": "Speed thodi zyaada hai explanation ki, notes lena mushkil ho raha", "like_count": 67},
        {"text": "Ha bilkul, yahi sunna tha mujhe 🙄 ek dum helpful video", "like_count": 145},
        {"text": "Camera work outstanding hai is video mein genuinely", "like_count": 287},
        {"text": "Delivery time 2 hafte laga, unacceptable for premium product", "like_count": 112},
        {"text": "bkl kuch bhi bolega kya, faltu information spread kar raha hai", "like_count": 8},
        {"text": "Color grading waah bhai 😍 kaunsa software use karte ho", "like_count": 394},
        {"text": "Price reasonable hai lekin quality aur improve ho sakti hai", "like_count": 73},
        {"text": "Yeh toh genius hai 😏 bahut hi enlightening video", "like_count": 167},
        {"text": "Refund process itna complicated kyun hai, 1 hafte se waiting", "like_count": 203},
        {"text": "Yaar ekdum mast video, late rat ko dekh ke neend khul gayi 😂❤️", "like_count": 456},
        {"text": "Script writing bahut weak hai, random information daal raha hai", "like_count": 44},
        {"text": "Oh wow kitni helpful video hai 🙄 aise hi 5 saal waste karo", "like_count": 178},
        {"text": "Customer service team bahut helpful thi, issue resolve ho gaya jaldi", "like_count": 134},
        {"text": "Background music bahut loud hai, voice dab rahi hai uske neeche", "like_count": 88},
        {"text": "Superb research quality, ek ek point backed by data 🔥", "like_count": 312},
        {"text": "Yeh koi naya concept nahi, purani baatein naye packaging mein", "like_count": 56},
        {"text": "Bhai teri editing skills dekhke mujhe khud seekhna hai genuinely", "like_count": 267},
    ]
    rows = []
    for i, d in enumerate(demo_data):
        rows.append(
            {
                "comment_id": f"demo_{i:03d}",
                "author": f"User_{i+1:03d}",
                "text": d["text"],
                "timestamp": pd.Timestamp("2024-11-01") + pd.Timedelta(hours=i * 3),
                "like_count": d["like_count"],
                "source": "demo",
            }
        )
    demo_df = pd.DataFrame(rows)
    demo_df.attrs["video_title"] = "Demo Indian Social Media Dataset"
    demo_df.attrs["channel_title"] = "IndicPulse Demo Studio"
    demo_df.attrs["video_description"] = "Curated demo reviews containing Indian code-mixed Hinglish and slang."
    return demo_df
