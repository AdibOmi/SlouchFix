"""Thin entry point for the system-tray app: python scripts/run_tray.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slouchfix.tray import main

if __name__ == "__main__":
    main()
