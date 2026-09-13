#!/usr/bin/env python3
"""
Download iNaturalist 2021 mini dataset.

Dataset info:
- 500K images (train_mini)
- 100K images (validation)
- 10K species
- Long-tail distribution
- Official train/val splits

AWS S3 Links:
- train_mini: s3://ml-inat-competition-datasets/2021/train_mini.tar.gz (42GB)
- train_mini json: s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz (45MB)
- validation: s3://ml-inat-competition-datasets/2021/val.tar.gz (8.4GB)
- validation json: s3://ml-inat-competition-datasets/2021/val.json.tar.gz (9.4MB)

Download from: https://github.com/visipedia/inat_comp/tree/main/2021
"""

import argparse
import subprocess
import os
from pathlib import Path
import json
from tqdm import tqdm


def download_from_s3(s3_url: str, dest: str):
    """Download from AWS S3 using AWS CLI or curl."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading: {s3_url}")
    print(f"📍 Destination: {dest}")

    # Try using AWS CLI if available
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", s3_url, str(dest)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Downloaded successfully!")
            return True
    except FileNotFoundError:
        print("⚠️ AWS CLI not found. Trying curl...")

    # Fallback to curl
    try:
        subprocess.run(
            ["curl", "-o", str(dest), s3_url],
            check=True
        )
        print(f"✅ Downloaded successfully!")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ curl not found either. Please download manually.")
        return False


def extract_tar_gz(tar_path: str, extract_to: str):
    """Extract tar.gz file."""
    tar_path = Path(tar_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Extracting: {tar_path.name}")
    try:
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            # Show progress
            members = tar.getmembers()
            for i, member in enumerate(tqdm(members, desc="Extracting")):
                tar.extract(member, extract_to)
        print(f"✅ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Error extracting: {e}")
        return False


def main(args):
    """Download iNaturalist 2021 mini dataset."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("iNaturalist 2021 Mini Dataset Download")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nDataset info:")
    print("  - Training images: 500K (train_mini)")
    print("  - Validation images: 100K")
    print("  - Total size: ~50GB (compressed)")
    print("  - Species: 10K")
    print("  - Distribution: Long-tail")

    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)

    print("""
Option 1: Download using AWS CLI (Fastest)
────────────────────────────────────────────
Prerequisites:
  pip install awscli

Commands:
  aws s3 cp s3://ml-inat-competition-datasets/2021/train_mini.tar.gz data/
  aws s3 cp s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz data/
  aws s3 cp s3://ml-inat-competition-datasets/2021/val.tar.gz data/
  aws s3 cp s3://ml-inat-competition-datasets/2021/val.json.tar.gz data/

Option 2: Download using curl
────────────────────────────────────────────
Commands:
  curl -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
  curl -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
  curl -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
  curl -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz

Option 3: Manual Download from GitHub
────────────────────────────────────────────
1. Visit: https://github.com/visipedia/inat_comp/tree/main/2021
2. Find S3 links in "Data" section
3. Download to data/inat2021_mini/

Expected structure after download:
────────────────────────────────────────────
data/inat2021_mini/
├── train_mini.tar.gz         (extract → train_mini/)
├── train_mini.json.tar.gz    (extract → train_mini.json)
├── val.tar.gz                (extract → val/)
├── val.json.tar.gz           (extract → val.json)

After extraction:
data/inat2021_mini/
├── train_mini2021.json
├── val2021.json
├── train_mini/
│   └── images/
│       ├── (category_ids)/
│       └── image files
└── val/
    └── images/
        ├── (category_ids)/
        └── image files

    """)

    if args.download:
        print("="*70)
        print("AUTOMATIC DOWNLOAD (if AWS CLI available)")
        print("="*70)

        files = [
            ("s3://ml-inat-competition-datasets/2021/train_mini.tar.gz", "train_mini.tar.gz"),
            ("s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz", "train_mini.json.tar.gz"),
            ("s3://ml-inat-competition-datasets/2021/val.tar.gz", "val.tar.gz"),
            ("s3://ml-inat-competition-datasets/2021/val.json.tar.gz", "val.json.tar.gz"),
        ]

        for s3_url, filename in files:
            dest = output_dir / filename
            if dest.exists():
                print(f"✅ Already downloaded: {filename}")
                continue

            success = download_from_s3(s3_url, str(dest))
            if success and args.extract:
                extract_tar_gz(str(dest), str(output_dir))

    # Verify download
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    json_files = list(output_dir.glob("*2021.json"))
    if json_files:
        for json_file in json_files:
            print(f"\n✅ Found: {json_file.name}")
            with open(json_file) as f:
                meta = json.load(f)
            print(f"   - Images: {len(meta.get('images', []))}")
            print(f"   - Categories: {len(meta.get('categories', []))}")

    image_dirs = list(output_dir.glob("*/images"))
    for img_dir in image_dirs:
        num_images = sum(1 for _ in img_dir.rglob("*.jpg"))
        print(f"✅ Found images: {img_dir.parent.name} - {num_images} JPG files")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Ensure all files are extracted properly:
   - train_mini2021.json
   - val2021.json
   - train_mini/images/ (contains subdirectories with JPG files)
   - val/images/ (contains subdirectories with JPG files)

2. Verify JSON file names match what data_loader.py expects:
   - train2021_mini.json (or rename train_mini2021.json)
   - val2021_mini.json (or rename val2021.json)

3. Run training:
   python scripts/train.py --data-dir data/inat2021_mini

Need help? Check the GitHub page:
https://github.com/visipedia/inat_comp/tree/main/2021
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download iNaturalist 2021 mini dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python scripts/download_dataset.py --output-dir data/inat2021_mini

  # For automatic downloads, see GitHub repo:
  https://github.com/visipedia/inat_comp
        """
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/inat2021_mini",
        help="Output directory for dataset"
    )

    args = parser.parse_args()
    main(args)
