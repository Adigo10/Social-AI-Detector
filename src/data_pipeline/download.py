"""Step 1: Download datasets (MultiSocial, HC3, RAID) to data/raw/."""

import os
import json
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

RAW_DIR = os.path.join("data", "raw")


def download_file(url, dest_path, description="Downloading"):
    """Stream-download a file with progress bar."""
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=description
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")


def download_multisocial():
    """Download MultiSocial dataset CSVs from Zenodo.

    This dataset has restricted access. You must either:
    1. Request access at https://zenodo.org/records/13846152 and set ZENODO_TOKEN in .env
    2. Or manually download the CSV files and place them in data/raw/multisocial/
    """
    print("\n=== Downloading MultiSocial dataset from Zenodo ===")

    dest_dir = os.path.join(RAW_DIR, "multisocial")
    os.makedirs(dest_dir, exist_ok=True)

    # Check if files were already placed manually
    existing = [f for f in os.listdir(dest_dir)
                if f.endswith(".csv") or f.endswith(".csv.gz")] if os.path.isdir(dest_dir) else []
    if existing:
        print(f"  Found {len(existing)} existing CSV file(s) — skipping download:")
        for f in existing:
            size_mb = os.path.getsize(os.path.join(dest_dir, f)) / (1024 * 1024)
            print(f"    - {f} ({size_mb:.1f} MB)")
        return

    # Try API with token (restricted dataset requires authentication)
    token = os.environ.get("ZENODO_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    api_url = "https://zenodo.org/api/records/13846152/files"
    resp = requests.get(api_url, headers=headers, timeout=30)

    if resp.status_code == 403 or resp.status_code == 401:
        print("  ⚠ MultiSocial is a RESTRICTED dataset on Zenodo.")
        print("  To download it, choose one of these options:")
        print()
        print("  Option A: Manual download")
        print("    1. Go to https://zenodo.org/records/13846152")
        print("    2. Request access and wait for approval")
        print("    3. Download the CSV file(s)")
        print("    4. Place them in data/raw/multisocial/")
        print("    5. Rerun this script")
        print()
        print("  Option B: API token (after access is granted)")
        print("    1. Create a token at https://zenodo.org/account/settings/applications/")
        print("    2. Add ZENODO_TOKEN=your_token to .env")
        print("    3. Rerun this script")
        print()
        print("  Skipping MultiSocial for now. Continuing with other datasets...")
        return

    resp.raise_for_status()
    data = resp.json()
    entries = data.get("entries", [])

    if not entries:
        print("  WARNING: No files found in Zenodo record. Skipping.")
        return

    print(f"  Found {len(entries)} file(s):")
    for entry in entries:
        key = entry["key"]
        size_mb = entry.get("size", 0) / (1024 * 1024)
        print(f"    - {key} ({size_mb:.1f} MB)")

    for entry in entries:
        key = entry["key"]
        if key.endswith(".csv") or key.endswith(".csv.gz"):
            url = entry["links"]["self"]
            dest = os.path.join(dest_dir, key)
            if os.path.exists(dest):
                print(f"  Skipping {key} (already exists)")
                continue
            download_file(url, dest, description=key)


def download_hc3():
    """Download HC3 dataset from HuggingFace."""
    print("\n=== Downloading HC3 dataset from HuggingFace ===")
    dest_dir = os.path.join(RAW_DIR, "hc3")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "hc3_all.jsonl")

    if os.path.exists(dest_path):
        print(f"  Skipping (already exists): {dest_path}")
        return

    ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)
    print(f"  Loaded splits: {list(ds.keys())}")

    count = 0
    with open(dest_path, "w", encoding="utf-8") as f:
        for split_name, split_data in ds.items():
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  Saved: {dest_path} ({size_mb:.1f} MB, {count} records)")


def download_raid():
    """Download RAID test set CSV."""
    print("\n=== Downloading RAID test set ===")
    dest_path = os.path.join(RAW_DIR, "raid_test.csv")

    if os.path.exists(dest_path):
        print(f"  Skipping (already exists): {dest_path}")
        return

    url = "https://dataset.raid-bench.xyz/test.csv"
    download_file(url, dest_path, description="RAID test.csv")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Starting dataset downloads...")

    download_multisocial()
    download_hc3()
    download_raid()

    print("\n=== Download complete ===")
    print(f"Raw data directory: {RAW_DIR}")
    for root, dirs, files in os.walk(RAW_DIR):
        for name in files:
            path = os.path.join(root, name)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
