"""
IndicPulse Studio • Playwright Network Interception Scraper
===========================================================
Headless browser scraper using Playwright's network response interception
to extract hidden JSON/GraphQL comment data from non-API platforms such as
Instagram Reels and Reddit threads.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    from indicpulse.scraper import fetch_comments_via_network_interception
    df = fetch_comments_via_network_interception(url, max_comments=500, platform="Instagram Reels")
"""

import time
import random
import logging

import pandas as pd

try:
    from playwright.sync_api import Error as PlaywrightError
except ImportError:
    class PlaywrightError(Exception):  # type: ignore
        """Fallback exception when playwright is not installed."""
        pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recursive JSON extraction helpers
# ---------------------------------------------------------------------------

# Keys that commonly hold comment/post text across platforms
# Includes e-commerce review keys (Flipkart, Amazon, etc.) and generic web structures
_TEXT_KEYS = [
    "text", "body", "content", "message", "comment", "caption",
    "description", "snippet", "reply_text", "commentText",
    "reviewText", "review_text", "title", "summary", "reviewBody",
    "reviewDescription", "feedbackText", "value", "review",
]

# Known API error / authentication failure strings to blacklist (case-insensitive).
# If any extracted text contains one of these substrings, it is discarded.
_ERROR_BLACKLIST = [
    "unauthorized logged out query",
    "login required",
    "rate limit exceeded",
    "please wait a few minutes before you try again",
    "page not found",
    "checkpoint required",
    "access denied",
    "not authorized",
    "authentication required",
    "session expired",
    "forbidden",
]

# Keys that commonly hold engagement / like counts
_LIKE_KEYS = [
    "likes", "like_count", "likeCount", "score", "ups", "thumbsUpCount",
    "reactionCount", "num_likes", "favoriteCount", "voteCount",
]

# URL substrings that identify network responses likely to carry comments
_COMMENT_URL_SIGNALS = [
    "graphql", "comment", "api", "feed", "thread", "post", "query",
    "timeline", "conversation", "replies", "discussion",
]


def _recursive_extract(obj, results: list, depth: int = 0) -> None:
    """
    Walk any JSON structure (dict / list) recursively and append dicts of
    {"text": str, "likes": int} to *results* whenever a text field is found.

    Extracted strings are checked against ``_ERROR_BLACKLIST`` before being
    accepted — this discards API error messages, login-wall responses, and
    rate-limit notices that would otherwise pollute the pipeline.

    Args:
        obj:     Parsed JSON value (dict, list, or scalar).
        results: Accumulator list — appended in-place.
        depth:   Current recursion depth (guard against deeply nested payloads).
    """
    if depth > 12 or len(results) >= 2000:
        return

    if isinstance(obj, dict):
        text_val: str | None = None
        like_val: int = 0

        # Try to extract a text value from known keys
        for k in _TEXT_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and len(v.strip()) > 5:
                candidate = v.strip()
                # ── API error blacklist filter ───────────────────────────
                candidate_lower = candidate.lower()
                if any(err in candidate_lower for err in _ERROR_BLACKLIST):
                    # This string is an API error — skip it entirely
                    break
                text_val = candidate
                break

        # Try to extract a like/score count from known keys
        for k in _LIKE_KEYS:
            if k in obj:
                try:
                    like_val = int(obj[k])
                except (TypeError, ValueError):
                    like_val = 0
                break

        if text_val:
            results.append({"text": text_val, "likes": like_val})

        # Recurse into all child values
        for v in obj.values():
            _recursive_extract(v, results, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            _recursive_extract(item, results, depth + 1)


# ---------------------------------------------------------------------------
# Public scraper function
# ---------------------------------------------------------------------------


def fetch_comments_via_network_interception(
    url: str,
    max_comments: int = 2000,
    platform: str = "web",
) -> pd.DataFrame:
    """
    Launch a headless Chromium browser, intercept all JSON/GraphQL network
    responses, and recursively extract comment text + like counts.

    Implements an anti-bot auto-scroll loop with randomised human-like delays
    (1.5–3.5 s) to trigger lazy-loaded comment batches without triggering
    rate-limit bans.

    Args:
        url:          Fully-qualified public post/thread URL.
        max_comments: Maximum number of comments to collect (tied to the
                      Analysis Depth slider in the UI).
        platform:     Human-readable platform label (e.g. "Instagram Reels",
                      "Reddit") — stored in the ``source`` column.

    Returns:
        A :class:`pandas.DataFrame` with columns matching the IndicPulse
        pipeline schema::

            comment_id | author | text | timestamp | like_count | source

        DataFrame.attrs are populated with ``video_title``,
        ``channel_title``, and ``video_description`` so that
        ``execute_analysis_flow`` can display them without a separate fetch.

    Raises:
        ImportError:  If playwright is not installed.
        ValueError:   If no comments could be extracted (login wall, wrong URL
                      structure, or Chromium not installed).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Playwright is not installed. Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        ) from exc

    comments_data: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            # Realistic desktop Chrome UA to reduce bot-detection probability
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        # ── Network interception handler ─────────────────────────────────
        def handle_response(response):  # noqa: ANN001
            if len(comments_data) >= max_comments:
                return
            url_lower = response.url.lower()
            if not any(sig in url_lower for sig in _COMMENT_URL_SIGNALS):
                return
            try:
                payload = response.json()
                _recursive_extract(payload, comments_data)
            except Exception:
                # Response body is not JSON — silently skip
                pass

        page.on("response", handle_response)

        # ── Navigate ─────────────────────────────────────────────────────
        try:
            page.goto(url, wait_until="networkidle", timeout=60_000)
        except Exception as exc:
            # Timeout is common on slow pages; we still attempt scraping
            logger.warning("Navigation note (%s): %s", url, exc)

        # ── Anti-bot auto-scroll loop ─────────────────────────────────────
        try:
            last_height = page.evaluate("document.body.scrollHeight")
        except (PlaywrightError, Exception) as exc:
            logger.warning("Could not read initial document height: %s", exc)
            last_height = 0

        scroll_attempts = 0

        while len(comments_data) < max_comments and scroll_attempts < 50:
            try:
                page.evaluate("window.scrollBy(0, 1500)")
                time.sleep(random.uniform(1.5, 3.5)) 

                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                    last_height = new_height

            except PlaywrightError as e:
                # If the platform forces a redirect (like a login wall), stop scrolling cleanly.
                if "Execution context was destroyed" in str(e) or "Target closed" in str(e):
                    print("Page redirected or closed (likely an anti-bot login wall). Stopping scroll.")
                    break 
                else:
                    print(f"Non-fatal Playwright error during scroll: {e}")
                    break
            except Exception as e:
                print(f"Unexpected error during scroll: {e}")
                break

        try:
            browser.close()
        except Exception as exc:
            logger.warning("Browser close warning: %s", exc)

    # ── Post-processing ───────────────────────────────────────────────────
    comments_data = comments_data[:max_comments]

    # Keep only entries that have non-empty text
    extracted_texts = [c.get("text", "").strip() for c in comments_data if c.get("text", "").strip()]
    extracted_likes = [c.get("likes", 0) for c in comments_data if c.get("text", "").strip()]

    if not extracted_texts:
        raise ValueError(
            "No valid comments found. The platform blocked the request, required a "
            "logged-in user session, or used an unrecognized data structure.\n\n"
            "Possible causes:\n"
            "  • Instagram / Flipkart require authentication — headless scraping is blocked.\n"
            "  • Rate-limit or CAPTCHA triggered after repeated requests.\n"
            "  • Chromium browser not installed — run: playwright install chromium\n\n"
            "Tip: Export comments to a CSV from the platform's web UI and use "
            "'Upload CSV' instead."
        )

    df = pd.DataFrame(
        {
            "comment_id": [f"scraped_{i:04d}" for i in range(len(extracted_texts))],
            "author": ["Anonymous"] * len(extracted_texts),
            "text": extracted_texts,
            "timestamp": pd.NaT,
            "like_count": extracted_likes,
            "source": [platform.lower()] * len(extracted_texts),
        }
    )

    # Attach metadata consumed by execute_analysis_flow display logic
    df.attrs["video_title"] = f"{platform} Post"
    df.attrs["channel_title"] = platform
    df.attrs["video_description"] = (
        f"Comments scraped via Playwright network interception from {url}"
    )

    logger.info("Playwright scraper captured %d comments from %s", len(df), url)
    return df
