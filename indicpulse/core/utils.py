"""
IndicPulse – Utility Functions
================================
Shared helpers used across the pipeline.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the IndicPulse pipeline."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    import re
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:100]


def get_project_root() -> str:
    """Return absolute path to indicpulse project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
