"""NHTSA CRSS National CSV batch downloader.

CRSS (Crash Report Sampling System) replaced GES in 2016. Data are hosted at:
    https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/National/CRSS{year}NationalCSV.zip

Years available: 2016 onward (CRSS started in 2016).
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_DIR / "data"

PRIMARY_URL = "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip"
AUXILIARY_URL = "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}AuxiliaryCSV.zip"
FALLBACK_URLS = [
    "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/National/CRSS{year}NationalCSV.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CRSS National CSV zips from NHTSA.")
    parser.add_argument("--years", nargs="*", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing zips.")
    return parser.parse_args()


def try_url(url: str, filename: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, filename.open("wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block = 65536
            while True:
                chunk = response.read(block)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:5.1f}% ({downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB)", end="\r")
        print(f"\n  [done] {filename} ({filename.stat().st_size/1024/1024:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  [fail] {url} -> {exc}")
        if filename.exists() and filename.stat().st_size == 0:
            filename.unlink()
        return False


def download_year(year: int, save_dir: Path, force: bool, also_auxiliary: bool = True) -> bool:
    # Main release named to match the FARS convention (CRSS{year}NationalCSV.zip) for downstream code compatibility
    filename = save_dir / f"CRSS{year}NationalCSV.zip"
    if filename.exists() and not force:
        print(f"[skip] {year} exists: {filename}")
    else:
        candidates = [PRIMARY_URL.format(year=year)] + [t.format(year=year) for t in FALLBACK_URLS]
        success = False
        for url in candidates:
            print(f"[try ] {year} -> {url}")
            if try_url(url, filename):
                success = True
                break
            if filename.exists() and filename.stat().st_size == 0:
                filename.unlink()
        if not success:
            print(f"[FAIL] {year}: all candidate URLs failed.")
            return False

    if also_auxiliary:
        aux_path = save_dir / f"CRSS{year}AuxiliaryCSV.zip"
        if not aux_path.exists() or force:
            aux_url = AUXILIARY_URL.format(year=year)
            print(f"[try ] {year} auxiliary -> {aux_url}")
            try_url(aux_url, aux_path)
    return True


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {args.save_dir.resolve()}")
    print(f"Years: {args.years}\n")

    results = {}
    for year in args.years:
        results[year] = download_year(year, args.save_dir, args.force)
        time.sleep(0.5)

    print("\nSummary:")
    for year, ok in results.items():
        print(f"  CRSS {year}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
