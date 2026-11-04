"""
IndicPulse – Step 2: Smart Preprocessing Engine
================================================
Transforms raw Hinglish/code-mixed comments into two parallel forms:

  • text_for_mood   – emoji tags included (MuRIL-ready)
  • text_for_topic  – emoji stripped, clean for BERTopic

Pipeline operations:
  1. Unicode normalization & whitespace cleanup
  2. Emoji → semantic bracket tag translation
  3. Elongated word normalization  (baaaahut → baahut)
  4. Indian profanity un-censoring + flagging
  5. Hinglish slang de-obfuscation
  6. URL / handle / noise removal
"""

import re
import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# EMOJI → SEMANTIC TAG MAP
# Used for MuRIL input (preserves emotional signal)
# ──────────────────────────────────────────────
EMOJI_TAG_MAP = {
    "🔥": "[fire_hype]",
    "❤️": "[heart_love]",
    "💔": "[broken_heart]",
    "😂": "[laughing_hard]",
    "🤣": "[laughing_hard]",
    "😭": "[crying_sad]",
    "😡": "[angry_frustrated]",
    "🤬": "[angry_frustrated]",
    "😍": "[love_amazed]",
    "🙌": "[celebration_praise]",
    "👌": "[ok_perfect]",
    "👍": "[thumbs_up]",
    "👎": "[thumbs_down]",
    "🤡": "[clown_mocking]",
    "🙄": "[eye_roll_sarcasm]",
    "😏": "[smirk_sarcasm]",
    "💀": "[dead_hilarious]",
    "☠️": "[dead_hilarious]",
    "☠": "[dead_hilarious]",
    "💃": "[dance_celebration]",
    "🕺": "[dance_celebration]",
    "🗿": "[stone_gigachad]",
    "😅": "[laughing_nervous]",
    "😆": "[laughing_hard]",
    "✅": "[verified_agree]",
    "❌": "[cross_disagree]",
    "⚡": "[electric_energy]",
    "💯": "[hundred_perfect]",
    "🚀": "[rocket_amazing]",
    "😎": "[cool_confident]",
    "🥺": "[pleading_sad]",
    "😢": "[crying_sad]",
    "🎉": "[party_celebration]",
    "🎊": "[party_celebration]",
    "💪": "[strong_motivation]",
    "🤔": "[thinking_confused]",
    "👏": "[clapping_praise]",
    "😊": "[smile_happy]",
    "🥰": "[love_adore]",
    "😤": "[frustrated_snort]",
    "😴": "[bored_sleepy]",
    "🤯": "[mind_blown]",
    "👀": "[watching_alert]",
    "💩": "[poop_bad]",
    "🗑️": "[trash_bad]",
    "⭐": "[star_rating]",
    "🌟": "[star_rating]",
}

# ──────────────────────────────────────────────
# PROFANITY UN-CENSOR MAP
# Maps censored/abbreviated forms → explicit flags
# ──────────────────────────────────────────────
PROFANITY_PATTERNS = [
    # Pattern (regex), replacement display, severity
    (r"\bch\*tiy[ae]?\b", "chutiya", "high"),
    (r"\bch[u]+tiy[ae]?\b", "chutiya", "high"),
    (r"\bb[k]+l\b", "bhenchod", "high"),
    (r"\bg\*\*nd\b", "gaand", "medium"),
    (r"\bg[a@]and\b", "gaand", "medium"),
    (r"\bm[a@]d[e3]rch[o0]d\b", "madarchod", "high"),
    (r"\bbc\b(?!\s*(?:rap|battle|bhai))\b", "bhenchod", "high"),
    (r"\bmc\b(?!\s*(?:stan|stanny|square|insaan|rap|hiphop|artist|bhai))\b", "madarchod", "high"),
    (r"\bl[o0]d[u]*\b", "lodu", "medium"),
    (r"\bhar[a@]mi\b", "harami", "medium"),
    (r"\bkamine\b", "kamine", "low"),
    (r"\bbewak[o0]of\b", "bewakoof", "low"),
    (r"\bb[e3]w[a@]k[o0][o0]f\b", "bewakoof", "low"),
    (r"\bsaala?\b", "saala", "low"),
    (r"\bbhad\s*mein\b", "bhad mein", "low"),
    (r"\bg[a@]ndu\b", "gandu", "high"),
    (r"\bb[a@]st[a@]rd\b", "bastard", "medium"),
]

# ──────────────────────────────────────────────
# HINGLISH SLANG NORMALIZATION
# ──────────────────────────────────────────────
SLANG_MAP = {
    r"\bkys\b": "karo yaar samjho",  # context re-map
    r"\bbtw\b": "by the way",
    r"\blol\b": "laughing out loud",
    r"\bomg\b": "oh my god",
    r"\bidk\b": "i don't know",
    r"\bngl\b": "not gonna lie",
    r"\bfr\b": "for real",
    r"\bbro\b": "bhai",
    r"\blit\b": "amazing",
    r"\bfam\b": "dost",
    r"\bsmh\b": "shaking my head",
    r"\btbh\b": "to be honest",
    r"\baf\b": "bahut zyaada",
    r"\bnah\b": "nahi",
    r"\byaar\b": "yaar",  # keep
    r"\bbhai\b": "bhai",  # keep
    r"\bkya\b": "kya",   # keep
}


# ──────────────────────────────────────────────
# CORE PREPROCESSING FUNCTIONS
# ──────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """Normalize unicode, fix encoding artifacts."""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    # Replace common encoding garbage
    text = text.replace("\u200b", " ").replace("\u200c", "").replace("\u200d", "")
    return text


def _remove_urls_handles(text: str) -> str:
    """Strip URLs, @mentions, and #hashtags."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    return text


def _translate_emojis_to_tags(text: str) -> Tuple[str, list]:
    """
    Replace emojis with semantic bracket tags.
    Returns (tagged_text, list_of_emoji_tags_found)
    """
    found_tags = []
    # First pass: known emoji map
    for emoji_char, tag in EMOJI_TAG_MAP.items():
        if emoji_char in text:
            text = text.replace(emoji_char, f" {tag} ")
            found_tags.append(tag)
    # Second pass: catch any remaining emojis with emoji library
    try:
        import emoji as emoji_lib
        text = emoji_lib.replace_emoji(text, replace=" [emoji] ")
    except ImportError:
        # Fallback: regex strip remaining unicode emoji ranges
        text = re.sub(
            r"[\U0001F300-\U0001FFFF\U00002600-\U000027FF]",
            " [emoji] ",
            text,
        )
    return text, found_tags


def _strip_all_emojis(text: str) -> str:
    """Remove all emoji characters and bracket tags for BERTopic."""
    # Remove bracket tags added in previous step
    text = re.sub(r"\[[\w_]+\]", " ", text)
    # Remove residual unicode emojis
    try:
        import emoji as emoji_lib
        text = emoji_lib.replace_emoji(text, replace=" ")
    except ImportError:
        text = re.sub(
            r"[\U0001F300-\U0001FFFF\U00002600-\U000027FF]",
            " ",
            text,
        )
    return text


def _normalize_elongated(text: str) -> str:
    """
    Collapse repeated characters (≥3) to 2.
    baaaahut → baahut, yaaaaar → yaaar
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _detect_uncensor_profanity(text: str) -> Tuple[str, bool, str]:
    """
    Un-censor masked profanity; return (cleaned_text, has_profanity, severity).
    Severity: 'none' | 'low' | 'medium' | 'high'
    """
    has_profanity = False
    max_severity = "none"
    severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}

    for pattern, replacement, severity in PROFANITY_PATTERNS:
        new_text, count = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if count > 0:
            text = new_text
            has_profanity = True
            if severity_rank[severity] > severity_rank[max_severity]:
                max_severity = severity

    return text, has_profanity, max_severity


def _normalize_slang(text: str) -> str:
    """Expand common Hinglish / internet slang abbreviations."""
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _clean_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing."""
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────
# PRIMARY PREPROCESSING FUNCTION
# ──────────────────────────────────────────────

def preprocess_comment(raw_text: str) -> dict:
    """
    Full preprocessing pipeline for a single comment.

    Returns
    -------
    dict with keys:
      text_for_mood   : str  – cleaned text with emoji tags (→ MuRIL)
      text_for_topic  : str  – cleaned text without emojis (→ BERTopic)
      has_profanity   : bool
      profanity_severity : str  ('none'|'low'|'medium'|'high')
      emoji_tags      : list[str]
    """
    text = str(raw_text)

    # Step 1: Unicode normalization
    text = _normalize_unicode(text)

    # Step 2: Remove URLs, handles, hashtags
    text = _remove_urls_handles(text)

    # Step 3: Normalize elongated words (before emoji processing)
    text = _normalize_elongated(text)

    # Step 4: Un-censor profanity
    text, has_profanity, severity = _detect_uncensor_profanity(text)

    # Step 5: Normalize slang
    text = _normalize_slang(text)

    # Step 6: Translate emojis → tags (for MuRIL)
    text_with_tags, emoji_tags = _translate_emojis_to_tags(text)
    text_for_mood = _clean_whitespace(text_with_tags)

    # Step 7: Strip all emoji tags (for BERTopic)
    text_no_emoji = _strip_all_emojis(text_with_tags)
    text_for_topic = _clean_whitespace(text_no_emoji)

    return {
        "text_for_mood": text_for_mood,
        "text_for_topic": text_for_topic,
        "has_profanity": has_profanity,
        "profanity_severity": severity,
        "emoji_tags": emoji_tags,
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocess_comment to all rows of the ingestion DataFrame.

    Adds columns:
      text_for_mood, text_for_topic, has_profanity,
      profanity_severity, emoji_tags
    """
    logger.info(f"Preprocessing {len(df)} comments...")
    results = df["text"].apply(preprocess_comment)
    enriched = pd.DataFrame(list(results))
    out = pd.concat([df.reset_index(drop=True), enriched], axis=1)
    if hasattr(df, "attrs"):
        out.attrs = df.attrs.copy()
    logger.info("Preprocessing complete.")
    return out
