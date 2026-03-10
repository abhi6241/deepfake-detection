# 🛡️ Real-Time Deepfake Detection & Liveness Verification

A real-time computer vision system that combines **liveness detection** (blink-based anti-spoofing) with **AI-powered deepfake risk scoring** using a Vision Transformer (ViT) model. Optimized for macOS with Apple Silicon GPU (MPS) acceleration.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Liveness Detection** | Tracks facial landmarks via MediaPipe and detects real blinks using Eye Aspect Ratio (EAR) |
| **Deepfake Risk Scoring** | Uses a fine-tuned ViT model ([Deep-Fake-Detector-v2](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model)) to classify face crops as real or fake |
| **MPS GPU Acceleration** | Automatically uses Apple Silicon GPU (Metal Performance Shaders) when available, falls back to CPU |
| **Temporal Smoothing** | 25-frame moving average on risk scores to prevent jitter, with trend indicators (⬆️/⬇️/➖) |
| **Proximity Gate** | Skips inference when the face is too far (< 120px bounding box) to prevent false positives from low-res crops |
| **Virtual Camera Detection** | Detects virtual cameras (OBS, ManyCam, Snap Camera, etc.) on macOS and raises a security alert |
| **Non-Blocking Inference** | Deepfake model runs on a background thread every 15 frames — the UI never freezes |

---

## 🏗️ How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Webcam      │────▶│  MediaPipe Face   │────▶│  EAR Blink Detection  │
│  (AVFoundation)│   │  Mesh / Landmarker│     │  (Liveness Check)     │
└─────────────┘     └──────────────────┘     └───────────────────────┘
                            │
                            ▼
                    ┌──────────────────┐     ┌───────────────────────┐
                    │  Face Crop       │────▶│  ViT Deepfake Model   │
                    │  (every 15 frames)│    │  (Background Thread)  │
                    │  + Proximity Gate │     │  + MPS GPU            │
                    └──────────────────┘     └───────────────────────┘
                                                      │
                                                      ▼
                                              ┌───────────────────┐
                                              │  25-Frame Moving  │
                                              │  Average + Trend  │
                                              └───────────────────┘
                                                      │
                                                      ▼
                                              ┌───────────────────┐
                                              │  OpenCV UI Overlay │
                                              │  (Risk %, Status) │
                                              └───────────────────┘
```

### Detection Pipeline

1. **Camera Capture** — Opens the default webcam using the AVFoundation backend (macOS-optimized).
2. **Security Check** — Enumerates system cameras via `system_profiler` and flags virtual cameras.
3. **Face Landmark Tracking** — MediaPipe detects 468+ facial landmarks in real time.
4. **Blink Detection** — Computes the Eye Aspect Ratio (EAR) from 6 eye landmarks per eye. EAR < 0.21 = eyes closed. A blink is detected on the rising edge (closed → open). After 1 blink, liveness is verified.
5. **Proximity Gate** — If the face bounding box is smaller than 120px, inference is skipped and a warning is shown.
6. **Deepfake Inference** — Every 15 frames, the face crop is sent to a background thread where the ViT model classifies it. Softmax is applied to the logits, and the "Fake" class probability is extracted.
7. **Temporal Smoothing** — The raw probability is added to a 25-frame rolling buffer. The displayed score is the buffer mean, with a trend arrow showing direction of change.

---

## 🚀 Quick Start

### Prerequisites

- **macOS** (optimized for Apple Silicon M1/M2/M3/M4)
- **Python 3.10+**
- Camera permissions enabled for your terminal app

### 1. Clone the repository

```bash
git clone https://github.com/abhi6241/deepfake-detection.git
cd deepfake-detection
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python3 liveness_detection.py
```

> **First run:** The ViT deepfake model (~340 MB) will be automatically downloaded from HuggingFace Hub and cached locally. Subsequent runs load instantly.

### 5. Usage

- **Look at the camera** — The system will start scanning for liveness.
- **Blink once** — Status changes from `SCANNING LIVENESS` (yellow) to `LIVENESS VERIFIED` (green).
- **Deepfake risk score** — Shown at the bottom-left with a progress bar (green < 70%, red > 70%).
- **Move closer** if you see the `WARNING: MOVE CLOSER TO CAMERA` banner.
- Press **`q`** to quit.

---

## 📁 Project Structure

```
deepfake-detection/
├── liveness_detection.py    # Main application script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .gitignore               # Git exclusions (venv/, models/, etc.)
└── models/                  # Auto-downloaded model cache (gitignored)
```

---

## ⚙️ Configuration

These constants can be adjusted at the top of `main()` in `liveness_detection.py`:

| Constant | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.21` | EAR value below which eyes are considered closed |
| `DEEPFAKE_INTERVAL` | `15` | Run deepfake inference every N frames |
| `MIN_FACE_PX` | `120` | Minimum face bounding box size (px) to run inference |

### Environment Variables

| Variable | Description |
|---|---|
| `VIDEO_DEVICE_INDEX` | Camera index to use (default: `0`) |
| `MP_FACE_LANDMARKER_MODEL` | Path to a custom MediaPipe FaceLandmarker TFLite model |

---

## 🖥️ UI Overlays

| Overlay | Location | Description |
|---|---|---|
| **Liveness Status** | Top center | `SCANNING LIVENESS` (yellow) → `LIVENESS VERIFIED` (green) |
| **Blink Counter** | Top right | Shows total blink count |
| **Deepfake Risk Score** | Bottom left | `DEEPFAKE RISK: XX.X% ⬆️/⬇️/➖` with color-coded progress bar |
| **Proximity Warning** | Center | `WARNING: MOVE CLOSER TO CAMERA` (yellow banner) |
| **Security Alert** | Top center | `CRITICAL SECURITY ALERT: VIRTUAL CAMERA DETECTED` (red) |

---

## 🛠️ Dependencies

```
mediapipe>=0.10.0          # Face landmark detection
opencv-python>=4.5         # Camera capture & UI rendering
numpy>=1.21                # Array operations
torch>=2.0                 # Neural network inference + MPS support
torchvision>=0.15          # Image transforms
transformers>=4.30         # HuggingFace ViT model loading
huggingface-hub>=0.17      # Model downloading
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| **Camera won't start** | Grant camera permission: *System Settings → Privacy & Security → Camera* → enable your terminal app |
| **MediaPipe import error** | Ensure Python 3.10+ and run `pip install mediapipe --upgrade` |
| **Model download fails** | Check internet connection. Set `HF_TOKEN` env variable if you hit rate limits |
| **MPS not detected** | Requires macOS 12.3+ and PyTorch 2.0+. Falls back to CPU automatically |
| **High CPU usage** | Increase `DEEPFAKE_INTERVAL` to run inference less frequently |
| **EAR not detecting blinks** | Adjust `EAR_THRESHOLD` — try `0.19` for glasses or `0.23` for larger eyes |

---

## 📄 License

MIT License
