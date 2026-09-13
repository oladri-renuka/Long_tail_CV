#!/usr/bin/env python3
"""
Upload ISIC 2019 dataset to Hugging Face Hub.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
import json

def upload_dataset():
    """Upload ISIC dataset to HF Hub."""

    # Configuration
    repo_id = "oladri-Renuka/isic-2019-long-tail"
    dataset_dir = Path("data")

    print("="*70)
    print("Uploading ISIC 2019 Dataset to Hugging Face Hub")
    print("="*70)
    print(f"\nRepository: {repo_id}")
    print(f"Dataset directory: {dataset_dir.absolute()}\n")

    # Initialize API
    api = HfApi()

    # Check if logged in
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face!")
        print("Run: huggingface-cli login")
        return False

    # Verify data exists
    files_to_upload = []

    # Check CSV files
    csv_files = [
        "ISIC_2019_Training_GroundTruth.csv",
        "ISIC_2019_Test_GroundTruth.csv",
        "ISIC_2019_Training_Metadata.csv",
        "ISIC_2019_Test_Metadata.csv",
    ]

    for csv_file in csv_files:
        csv_path = dataset_dir / csv_file
        if csv_path.exists():
            files_to_upload.append(csv_path)
            print(f"✅ Found: {csv_file}")
        else:
            print(f"❌ Missing: {csv_file}")

    # Check image directories
    train_dir = dataset_dir / "ISIC_2019_Training_Input"
    test_dir = dataset_dir / "ISIC_2019_Test_Input"

    if train_dir.exists():
        train_count = len(list(train_dir.glob("*.jpg")))
        print(f"✅ Found: ISIC_2019_Training_Input/ ({train_count} images)")
    else:
        print(f"❌ Missing: ISIC_2019_Training_Input/")

    if test_dir.exists():
        test_count = len(list(test_dir.glob("*.jpg")))
        print(f"✅ Found: ISIC_2019_Test_Input/ ({test_count} images)")
    else:
        print(f"❌ Missing: ISIC_2019_Test_Input/")

    # Create README for dataset
    readme_content = """---
task_categories:
- image-classification
language:
- en
license: cc-by-nc-4.0
size_categories:
- 10K<n<100K
---

# ISIC 2019 Skin Lesion Dataset - Long-Tail Version

This is the **ISIC 2019 Challenge** dataset prepared for long-tail learning research.

## Dataset Info

- **25,333 training images**
- **8,240 test images**
- **8 disease types** (melanoma, nevus, BCC, AK, BKL, DF, VASC, SCC)
- **Long-tail distribution**: Most images are Nevus (~51%), rare classes have <1%

## Dataset Structure

```
data/
├── ISIC_2019_Training_Input/          (25,333 training images)
├── ISIC_2019_Test_Input/              (8,240 test images)
├── ISIC_2019_Training_GroundTruth.csv (training labels - one-hot encoded)
├── ISIC_2019_Test_GroundTruth.csv     (test labels)
├── ISIC_2019_Training_Metadata.csv    (training image metadata)
└── ISIC_2019_Test_Metadata.csv        (test image metadata)
```

## Class Distribution

| Disease | Count | % | Type |
|---------|-------|---|------|
| Nevus (NV) | 12,875 | 51% | Head (common) |
| Melanoma (MEL) | 1,113 | 4% | Mid |
| BCC | 514 | 2% | Mid |
| AK | 450 | 2% | Mid |
| BKL | 327 | 1% | Mid |
| VASC | 142 | <1% | Tail (rare) |
| DF | 115 | <1% | Tail |
| SCC | 79 | <1% | Tail |

## Usage

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset('oladri-Renuka/isic-2019-long-tail')

# Access training data
train_images = dataset['train']
test_images = dataset['test']

# Get metadata
print(train_images.features)
```

Or for this project:

```bash
# Download to local machine
python scripts/download_from_hf.py

# Will download to: data/ directory
```

## Citation

If you use this dataset, please cite the original ISIC dataset:

```bibtex
@inproceedings{combalia2019isic,
  title={ISIC 2019 Challenge Dataset},
  author={Combalia, Marc and others},
  booktitle={Proceedings of the CVPR 2019 Workshops},
  year={2019}
}
```

## License

CC-BY-NC (Creative Commons Attribution Non-Commercial)

## References

- Official ISIC: https://challenge.isic-archive.com/
- Challenge Paper: https://arxiv.org/abs/1902.08497

---

Dataset prepared for long-tail learning research on medical imaging.
"""

    readme_path = dataset_dir / "README.md"
    print(f"\n📝 Creating README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    # Upload to HF
    print(f"\n{'='*70}")
    print("UPLOADING TO HUGGING FACE HUB")
    print(f"{'='*70}\n")

    try:
        # Upload entire folder
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload ISIC 2019 dataset for long-tail learning",
            ignore_patterns=[
                "*.pyc",
                "__pycache__",
                ".git",
                ".gitignore",
                "*.zip",
            ],
        )

        print(f"\n✅ Upload complete!")
        print(f"\n📊 Dataset available at:")
        print(f"   https://huggingface.co/datasets/{repo_id}")
        print(f"\n🚀 On RunPod, pull with:")
        print(f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="data"
)
        """)

        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload ISIC dataset to HF Hub")
    parser.add_argument("--repo-id", type=str, default="oladri-Renuka/isic-2019-long-tail",
                       help="HF Hub repository ID")

    args = parser.parse_args()

    success = upload_dataset()
    exit(0 if success else 1)
