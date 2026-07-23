"""Thin entry point for the console app: python scripts/run_app.py --preview"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slouchfix.app import main

if __name__ == "__main__":
    main()
