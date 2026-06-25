# MediaPipe Face Landmarker Integration

This folder contains local integration code for MediaPipe Face Landmarker expression tracking.

- Official docs: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- Model file: `models/face_landmarker.task`
- Wrapper: `face_landmarker.py`

The `.task` model file is ignored by this folder's `.gitignore`. Download it with:

```powershell
New-Item -ItemType Directory -Force thirdparty\MediaPipe\models
Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task -OutFile thirdparty\MediaPipe\models\face_landmarker.task
```

The wrapper maps MediaPipe blendshape scores to VTube Studio input parameters such as:

- `EyeOpenLeft`
- `EyeOpenRight`
- `MouthOpen`
- `MouthSmile`
- `BrowLeftY`
- `BrowRightY`
