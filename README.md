# Liveness Detection (OpenCV + MediaPipe)

This small script captures the default webcam on macOS, uses MediaPipe Face Mesh to track facial landmarks, computes Eye Aspect Ratio (EAR) to detect blinks, and displays a clean overlay that shows "SCANNING LIVENESS" (yellow) until the user blinks once, after which it shows "LIVENESS VERIFIED" (green).

Files:
- `liveness_detection.py` — main script
- `requirements.txt` — Python dependencies

Quick start (macOS):

1. Create a virtual environment and activate it (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python3 liveness_detection.py
```

Press `q` to quit. The script uses the AVFoundation backend on macOS for better camera compatibility.

Notes and tuning:
- EAR threshold is set to 0.21; adjust if needed for your camera/subject.
- The script uses the default webcam index 0. If you have multiple cameras, change the index in the code.

Troubleshooting:
- If the camera fails to start, ensure the app (terminal or Python) has camera permission in macOS System Settings.
- If MediaPipe installation fails on Apple Silicon, ensure you have a compatible Python version and follow the MediaPipe installation docs.
