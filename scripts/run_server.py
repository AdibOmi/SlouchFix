"""Thin entry point for the local backend the Flutter app talks to:
python scripts\\run_server.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slouchfix.server import main

if __name__ == "__main__":
    main()
