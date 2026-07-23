"""Thin entry point: python scripts/collect_data.py --person p01 --distance 50"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slouchfix.data_collection import main

if __name__ == "__main__":
    main()
