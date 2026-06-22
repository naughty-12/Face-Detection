"""Download WIDER Face dataset. If URLs timeout, manually download from:
   http://shuoyang1213.me/WIDERFACE/
   and place under data/raw/ and data/annotations/"""
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
ANNO_DIR = os.path.join(os.path.dirname(__file__), "annotations")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ANNO_DIR, exist_ok=True)
    print("WIDER Face dataset requires manual download (~3GB).")
    print("1. Download WIDER_train.zip and WIDER_val.zip from http://shuoyang1213.me/WIDERFACE/")
    print("2. Extract WIDER_train/images/ → data/raw/WIDER_train/images/")
    print("3. Extract WIDER_val/images/   → data/raw/WIDER_val/images/")
    print("4. Download wider_face_split.zip and extract to data/annotations/")
    print("5. After extraction, run: python data/generate_list.py")


if __name__ == "__main__":
    main()
