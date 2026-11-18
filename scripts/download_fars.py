"""NHTSA FARS National CSV downloader (FARS{year}NationalCSV.zip into data/).

Files are hosted at
    https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip
(listed on https://www.nhtsa.gov/file-downloads). The analysis uses 2021-2023.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_DIR / "data"
BASE_URL = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FARS National CSV zips from NHTSA.")
    parser.add_argument("--years", nargs="*", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing zips.")
    return parser.parse_args()


def download(year: int, save_dir: Path, force: bool) -> bool:
    url = BASE_URL.format(year=year)
    filename = save_dir / f"FARS{year}NationalCSV.zip"
    if filename.exists() and not force:
        print(f"[skip] {year} exists: {filename}")
        return True
    print(f"[try ] {year} -> {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, filename.open("wb") as out_file:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"  [done] {filename} ({filename.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  [fail] {url} -> {exc}")
        if filename.exists():
            filename.unlink()
        return False


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {args.save_dir.resolve()}")
    results = {}
    for year in args.years:
        results[year] = download(year, args.save_dir, args.force)
        time.sleep(0.5)
    print("\nSummary:")
    for year, ok in results.items():
        print(f"  FARS {year}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
