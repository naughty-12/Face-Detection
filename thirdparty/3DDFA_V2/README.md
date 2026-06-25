# 3DDFA_V2 Head Pose Integration

This folder contains a trimmed local copy of the pieces needed from `cleardusk/3DDFA_V2`.

- Source repo: https://github.com/cleardusk/3DDFA_V2
- Config: `configs/mb1_120x120.yml`
- PyTorch weight: `weights/mb1_120x120.pth`
- Converted model: `weights/mb1_120x120.onnx`
- BFM asset: `configs/bfm_noneck_v3.pkl`
- Converted BFM decoder: `configs/bfm_noneck_v3.onnx`
- Head-pose wrapper: `head_pose.py`

The ONNX files and large model assets are ignored by this folder's `.gitignore`. Recreate ONNX files with:

```powershell
.\.venv\Scripts\python.exe thirdparty\3DDFA_V2\convert_to_onnx.py
```

The conversion script keeps paths local to this folder and uses the patched exporter calls in:

- `utils/onnx.py`
- `bfm/bfm_onnx.py`

Those patches pass `dynamo=False` to `torch.onnx.export` for compatibility with newer PyTorch versions.

`head_pose.py` estimates VTube Studio-compatible angle keys from a face bounding box:

- `FaceAngleX`
- `FaceAngleY`
- `FaceAngleZ`
