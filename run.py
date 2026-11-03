#!/usr/bin/env python
"""
Launcher script for IndicPulse Studio
====================================
Usage:
    python run.py
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "indicpulse" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
