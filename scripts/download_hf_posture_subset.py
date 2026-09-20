"""
Download a small labeled subset of SeiyaCM/KandenAiHackathonPosture2 without
pulling the full ~52,500-image dataset.

How this avoids downloading the whole dataset
-----------------------------------------------
The dataset's rows are stored in contiguous blocks per label (verified by
probing the Hugging Face datasets-server "rows" API):
    rows      0 -  ~13,124  -> 01_good
    rows ~13,125 - ~26,249  -> 02_slouch
    rows ~26,250 - ~39,374  -> 03_chin_rest
    rows ~39,375 -  52,499  -> 04_stretch

Because of that, we don't need to scan or download the underlying parquet
shards at all: the public datasets-server "rows" endpoint can return any
row range directly (image URL + label + prompt) in pages of up to 100 rows.
We just request two known-safe row windows, one from inside the "good"
block and one from inside the "slouch" block, and download the images
those rows point to. Total network traffic is ~2,000 images, not 52,500.

Usage
-----
    python scripts/download_hf_posture_subset.py --n 1000
"""
import argparse
import time
from pathlib import Path

import requests

DATASET = "SeiyaCM/KandenAiHackathonPosture2"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # max allowed by the datasets-server API

# (label, output folder name, a row offset already confirmed to sit safely
# inside that label's contiguous block)
TARGETS = [
    ("01_good", "good", 0),
    ("02_slouch", "slouched", 15000),
]


def fetch_page(offset: int, length: int, retries: int = 5) -> list[dict]:
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }
    for attempt in range(retries):
        resp = requests.get(ROWS_API, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()["rows"]
        if resp.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch rows at offset {offset} after {retries} retries")


def download_image(url: str, dest: Path, retries: int = 5) -> None:
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            return
        if resp.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to download {url} after {retries} retries")


def collect(label: str, folder: Path, start_offset: int, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    offset = start_offset
    while saved < n:
        length = min(PAGE_SIZE, n - saved)
        rows = fetch_page(offset, length)
        if not rows:
            raise RuntimeError(
                f"Ran out of rows for label {label!r} before reaching {n} images "
                f"(stopped at offset {offset}). Adjust the start offset for this label."
            )
        for entry in rows:
            row = entry["row"]
            if row["label"] != label:
                raise RuntimeError(
                    f"Unexpected label {row['label']!r} at offset {entry['row_idx']}, "
                    f"expected {label!r}. The dataset's row ordering may differ from "
                    f"what this script assumes -- adjust the start offset for this label."
                )
            img_url = row["image"]["src"]
            out_path = folder / f"{entry['row_idx']:06d}.jpg"
            download_image(img_url, out_path)
            saved += 1
        offset += length
        print(f"  {label}: {saved}/{n} saved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="images per class")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "hf_posture_images",
        help="output directory (a subfolder per class is created inside it)",
    )
    args = parser.parse_args()

    for label, folder_name, start_offset in TARGETS:
        print(f"Fetching {args.n} images for label {label!r} starting at row {start_offset} ...")
        collect(label, args.out / folder_name, start_offset, args.n)

    print(f"Done. Images saved under: {args.out}")


if __name__ == "__main__":
    main()
