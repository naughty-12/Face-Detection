# PFLD Landmark Integration

This folder contains local integration code for `yakhyo/face-landmark-detection`.

- Source repo: https://github.com/yakhyo/face-landmark-detection
- ONNX weight: `weights/pfld_landmark.onnx`
- Input shape: `[1, 3, 112, 112]`
- Landmark output shape: `[1, 196]`, reshaped to `98 x 2`
- Default face crop scale: `1.3`

The ONNX weight is ignored by Git via the project-level `*.onnx` rule. Re-download it with:

```powershell
New-Item -ItemType Directory -Force thirdparty\PFLD\weights
Invoke-WebRequest -Uri https://github.com/yakhyo/face-landmark-detection/raw/main/weights/pfld_landmark.onnx -OutFile thirdparty\PFLD\weights\pfld_landmark.onnx
```
