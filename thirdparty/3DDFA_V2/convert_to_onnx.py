"""Convert local 3DDFA_V2 PyTorch assets to ONNX."""
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bfm.bfm_onnx import convert_bfm_to_onnx
from utils.onnx import convert_to_onnx


def main():
    config_path = ROOT / "configs" / "mb1_120x120.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bfm_onnx = ROOT / "configs" / "bfm_noneck_v3.onnx"
    if not bfm_onnx.exists():
        convert_bfm_to_onnx(str(bfm_onnx))
    else:
        print(f"BFM ONNX already exists: {bfm_onnx}")

    checkpoint = ROOT / cfg["checkpoint_fp"]
    model_onnx = checkpoint.with_suffix(".onnx")
    if not model_onnx.exists():
        cfg = dict(cfg)
        cfg["checkpoint_fp"] = str(checkpoint)
        convert_to_onnx(**cfg)
    else:
        print(f"3DDFA ONNX already exists: {model_onnx}")


if __name__ == "__main__":
    main()
