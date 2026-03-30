"""Step 1: Download datasets (MultiSocial, HC3, RAID) to data/raw/."""

import os
import json
import requests
from tqdm import tqdm
from datasets import load_dataset

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
    """Download MultiSocial dataset CSVs from Zenodo."""
    print("\n=== Downloading MultiSocial dataset from Zenodo ===")
    api_url = "https://zenodo.org/api/records/13846152"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    record = resp.json()

    files = record.get("files", [])
    if not files:
        raise RuntimeError("No files found in Zenodo record 13846152")

    print(f"  Found {len(files)} file(s) in Zenodo record:")
    for f in files:
        print(f"    - {f['key']} ({f['size'] / (1024*1024):.1f} MB)")

    dest_dir = os.path.join(RAW_DIR, "multisocial")
    os.makedirs(dest_dir, exist_ok=True)

    for f in files:
        name = f["key"]
        if name.endswith(".csv") or name.endswith(".csv.gz"):
            url = f["links"]["self"]
            dest = os.path.join(dest_dir, name)
            if os.path.exists(dest):
                print(f"  Skipping {name} (already exists)")
                continue
            download_file(url, dest, description=name)


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
