#!/usr/bin/env python3
"""
Download ISIC dataset from Hugging Face Hub (for RunPod/cloud training).
"""

from pathlib import Path
from huggingface_hub import snapshot_download
import sys


def download_dataset(repo_id="oladri-Renuka/isic-2019-long-tail", output_dir="data"):
    """Download dataset from HF Hub."""

    print("="*70)
    print("Downloading ISIC 2019 Dataset from Hugging Face Hub")
    print("="*70)
    print(f"\nRepository: {repo_id}")
    print(f"Output directory: {output_dir}\n")

    try:
        print("📥 Downloading dataset...")
        print("(This may take 5-15 minutes depending on internet speed)\n")

        # Download from HF Hub
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            cache_dir=".cache",
        )

        print(f"\n✅ Download complete!")

        # Verify
        output_path = Path(output_dir)

        # Count files
        train_images = len(list((output_path / "ISIC_2019_Training_Input").glob("*.jpg")))
        test_images = len(list((output_path / "ISIC_2019_Test_Input").glob("*.jpg")))

        print(f"\n📊 Dataset Statistics:")
        print(f"   - Training images: {train_images}")
        print(f"   - Test images: {test_images}")

        # Check CSVs
        csv_files = [
            "ISIC_2019_Training_GroundTruth.csv",
            "ISIC_2019_Test_GroundTruth.csv",
            "ISIC_2019_Training_Metadata.csv",
            "ISIC_2019_Test_Metadata.csv",
        ]

        print(f"\n   CSV Files:")
        for csv in csv_files:
            csv_path = output_path / csv
            if csv_path.exists():
                size_mb = csv_path.stat().st_size / (1024*1024)
                print(f"   ✅ {csv} ({size_mb:.1f}MB)")
            else:
                print(f"   ❌ {csv} (missing)")

        print(f"\n🚀 Ready to train!")
        print(f"\n   Run: python scripts/train_isic.py --data-dir {output_dir}")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Verify repository is public: https://huggingface.co/datasets/{repo_id}")
        print(f"3. Try again with: python scripts/download_from_hf.py")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ISIC dataset from HF Hub")
    parser.add_argument("--repo-id", type=str, default="oladri-Renuka/isic-2019-long-tail",
                       help="HF Hub repository ID")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for dataset")

    args = parser.parse_args()

    success = download_dataset(args.repo_id, args.output_dir)
    sys.exit(0 if success else 1)
