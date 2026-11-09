from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_download import download_and_extract  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data"
    print("Downloading the official Recipe Box archive...")
    paths = download_and_extract(data_dir, lambda done, total: print(
        f"Downloaded {done / (1024 * 1024):,.1f} MB" +
        (f" / {total / (1024 * 1024):,.1f} MB" if total else ""), end="\r"
    ))
    print("\nDataset files ready:")
    for path in paths:
        print(f" - {path.name} ({path.stat().st_size / (1024 * 1024):,.1f} MB)")


if __name__ == "__main__":
    main()
