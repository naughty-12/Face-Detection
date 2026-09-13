"""WIDER Face dataset setup helper.

This script does **not** download anything: the dataset is ~3 GB and the upstream
URLs are unreliable, so it only creates the expected directories and prints manual
download instructions. Once the archives are extracted, run the conversion and
split scripts to produce what training needs.
"""
import os

from src.paths import ANNO_DIR, RAW_DIR


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ANNO_DIR, exist_ok=True)
    print("WIDER Face dataset requires manual download (~3 GB).")
    print("1. Download WIDER_train.zip and WIDER_val.zip from http://shuoyang1213.me/WIDERFACE/")
    print("2. Extract WIDER_train/images/ -> data/raw/WIDER_train/images/")
    print("3. Extract WIDER_val/images/   -> data/raw/WIDER_val/images/")
    print("4. Download wider_face_split.zip and extract to data/annotations/")
    print("5. Convert annotations to YOLO labels:  python -m src.data.convert")
    print("6. Build train/val image lists:         python -m src.data.split")


if __name__ == "__main__":
    main()
