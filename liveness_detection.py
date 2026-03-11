#!/usr/bin/env python3
"""
Real-time liveness detection + deepfake risk scoring.

Features:
- Capture default webcam (optimized for macOS using AVFoundation backend)
- Use MediaPipe Face Mesh to obtain facial landmarks
- Compute Eye Aspect Ratio (EAR) to detect blinks
- UI overlay: "SCANNING LIVENESS" (yellow) -> "LIVENESS VERIFIED" (green) after 1 blink
- Blink counter displayed
- Deepfake risk score via EfficientNet-B0 (runs on background thread every 15 frames)
- Clean camera release on 'q'

Usage:
    python3 liveness_detection.py

Dependencies:
    mediapipe, opencv-python, numpy, torch, torchvision

"""
import collections
import math
import queue
import random
import threading
import time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import json
import os
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Face Mesh landmark indices for eyes (MediaPipe Face Mesh)
# Left eye: [33, 160, 158, 133, 153, 144]
# Right eye: [263, 387, 385, 362, 380, 373]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# --- Unified Fraud Risk Engine weights ---
W_DEEPFAKE = 0.45
W_LIVENESS = 0.30
W_CHALLENGE = 0.15
W_CAMERA = 0.10


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def eye_aspect_ratio(landmarks: List[Tuple[float, float]], eye_indices: List[int]) -> float:
    """Compute EAR for a single eye.

    landmarks: list of (x, y) pixel coordinates for all face landmarks
    eye_indices: six indices for the eye in the order [p1, p2, p3, p4, p5, p6]
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    # vertical distances
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    # horizontal distance
    C = euclidean(p1, p4)

    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear


def landmarks_to_pixel_list(face_landmarks, image_w: int, image_h: int) -> List[Tuple[float, float]]:
    pts = []
    for lm in face_landmarks.landmark:
        x_px = int(lm.x * image_w)
        y_px = int(lm.y * image_h)
        pts.append((x_px, y_px))
    return pts


# ---------------------------------------------------------------------------
# Virtual Camera Detection (Zero Trust Allowlist)
# ---------------------------------------------------------------------------
CAMERA_ALLOWLIST = ['facetime', 'apple', 'built-in', 'macbook']


def check_virtual_camera() -> Tuple[bool, str]:
    """Query macOS hardware registry for cameras and verify against allowlist.

    Returns (is_untrusted, camera_name).  A camera is trusted only if its
    name contains a keyword from CAMERA_ALLOWLIST (case-insensitive).
    Runs once at startup and does not block the real-time pipeline.
    """
    if os.name != 'posix':
        return True, 'Unknown'

    cameras: List[str] = []
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for entry in data.get('SPCameraDataType', []):
                name = entry.get('_name') or entry.get('camera') or ''
                if not name:
                    for v in entry.values():
                        if isinstance(v, str):
                            name = v
                            break
                if name:
                    cameras.append(name)
    except Exception:
        pass

    # Fallback: plain-text parsing
    if not cameras:
        try:
            result = subprocess.run(
                ['system_profiler', 'SPCameraDataType'],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped and not line.startswith(' ') and stripped.endswith(':'):
                    cameras.append(stripped.rstrip(':'))
        except Exception:
            pass

    if not cameras:
        return True, 'No camera detected'

    # Pick the camera matching VIDEO_DEVICE_INDEX (default 0)
    try:
        idx = int(os.environ.get('VIDEO_DEVICE_INDEX', '0'))
    except Exception:
        idx = 0
    cam_name = cameras[idx] if 0 <= idx < len(cameras) else cameras[0]

    # Zero Trust: only trust cameras on the allowlist
    cam_lower = cam_name.lower()
    for trusted_keyword in CAMERA_ALLOWLIST:
        if trusted_keyword in cam_lower:
            return False, cam_name  # trusted native sensor

    return True, cam_name  # not on allowlist → untrusted


# ---------------------------------------------------------------------------
# Deepfake Detector (HuggingFace ViT + MPS acceleration)
# ---------------------------------------------------------------------------
MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"


class ViTWrapper(torch.nn.Module):
    """Thin wrapper so GradCAM receives a raw logits tensor, not a ModelOutput."""

    def __init__(self, vit_model: ViTForImageClassification) -> None:
        super().__init__()
        self.vit = vit_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).logits


class DeepfakeDetector:
    """Wraps a HuggingFace ViT fine-tuned for deepfake detection.

    Uses ViTForImageClassification + ViTImageProcessor from the
    'prithivMLmods/Deep-Fake-Detector-v2-Model' checkpoint.
    Weights are downloaded automatically on first run.

    Features:
    - MPS (Apple Silicon GPU) acceleration when available, CPU fallback.
    - 25-frame moving-average score buffer for temporal smoothing.
    - Trend indicator (⬆️ rising / ⬇️ falling / ➖ stable).
    """

    def __init__(self) -> None:
        # --- Device selection: prefer Apple MPS, fall back to CPU ---
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Neural Network running on: {self.device}")

        print(f"Downloading / loading deepfake model: {MODEL_ID} …")
        self.processor = ViTImageProcessor.from_pretrained(MODEL_ID)
        self.model = ViTForImageClassification.from_pretrained(MODEL_ID)
        self.model.eval()
        self.model.to(self.device)
        print("Deepfake detector ready.")

        # Determine which class index corresponds to "Fake" / "Deepfake"
        self.fake_index = 0  # fallback
        id2label = getattr(self.model.config, "id2label", {})
        for idx, label in id2label.items():
            if "fake" in str(label).lower():
                self.fake_index = int(idx)
                break

        # --- Temporal smoothing state ---
        self.score_buffer: collections.deque = collections.deque(maxlen=50)
        self.current_avg_score: float = 0.0
        self.trend: str = ""

        # --- GradCAM (XAI) setup ---
        self._cam_wrapper = ViTWrapper(self.model)
        # Target the last encoder layer's layer norm
        target_layer = self.model.vit.encoder.layer[-1].layernorm_before
        self.grad_cam = GradCAM(model=self._cam_wrapper,
                                target_layers=[target_layer])
        self.current_heatmap: np.ndarray | None = None  # set by inference thread
        self._heatmap_lock = threading.Lock()

    # ----- public helpers --------------------------------------------------
    def reset_buffer(self) -> None:
        """Clear the rolling score buffer (call when face is lost)."""
        self.score_buffer.clear()
        self.current_avg_score = 0.0
        self.trend = ""

    @torch.no_grad()
    def predict(self, face_bgr: np.ndarray) -> float:
        """Return deepfake probability (0-1) for a BGR face crop."""
        # Convert BGR numpy array → RGB PIL Image
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        probs = F.softmax(outputs.logits, dim=1)
        fake_prob = probs[0, self.fake_index].cpu().item()
        return float(fake_prob)

    def update_buffer(self, prob: float) -> None:
        """Append a new probability, recompute moving average & trend."""
        # Apply calibration offset to compensate for webcam compression artifacts
        calibrated = max((prob * 100.0) - 5.0, 0.0)
        prev_avg = self.current_avg_score
        self.score_buffer.append(calibrated)
        self.current_avg_score = sum(self.score_buffer) / len(self.score_buffer)

        delta = self.current_avg_score - prev_avg
        if delta > 2.0:
            self.trend = "⬆️"
        elif delta < -2.0:
            self.trend = "⬇️"
        else:
            self.trend = "➖"

    def compute_heatmap(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """Return a grayscale GradCAM heatmap (H×W float32 0-1) for *face_bgr*."""
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # GradCAM needs grad, temporarily enable it
            grayscale_cam = self.grad_cam(input_tensor=pixel_values,
                                          targets=None)  # uses top predicted class
            return grayscale_cam[0]  # (H, W) float32 in [0, 1]
        except Exception:
            return None


def _inference_worker(detector: DeepfakeDetector,
                      job_queue: queue.Queue,
                      result_dict: dict,
                      result_lock: threading.Lock) -> None:
    """Background thread: pull face crops from *job_queue* and run inference."""
    while True:
        item = job_queue.get()
        if item is None:  # poison pill → stop
            break
        face_crop, frame_id = item
        try:
            prob = detector.predict(face_crop)
            detector.update_buffer(prob)

            # Compute GradCAM heatmap if risk is elevated
            if detector.current_avg_score > 60.0:
                heatmap = detector.compute_heatmap(face_crop)
            else:
                heatmap = None
            with detector._heatmap_lock:
                detector.current_heatmap = heatmap
        except Exception:
            prob = 0.0
        with result_lock:
            result_dict["avg_score"] = detector.current_avg_score
            result_dict["trend"] = detector.trend
            result_dict["frame_id"] = frame_id


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_overlay(frame: np.ndarray, text: str, status_color: Tuple[int, int, int], blink_count: int) -> None:
    # semi-transparent bar at top
    overlay = frame.copy()
    h, w = frame.shape[:2]
    bar_h = 60
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    # Draw left text: title
    title = text
    (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
    cv2.putText(frame, title, ((w - tw) // 2, 40), font, font_scale, status_color, thickness, cv2.LINE_AA)

    # Blink counter at top-right
    if blink_count is None:
        counter_text = ""
    elif isinstance(blink_count, int) and blink_count >= 100:
        counter_text = f"Risk: {blink_count}%"
    else:
        counter_text = f"Blinks: {blink_count}"
    (ctw, cth), _ = cv2.getTextSize(counter_text, font, 0.7, 2)
    cv2.putText(frame, counter_text, (w - ctw - 16, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_unified_risk_panel(frame: np.ndarray, final_pct: float,
                            s_df: float, s_live: float,
                            s_chal: float, s_cam: float) -> None:
    """Draw the Unified Fraud Risk panel at the bottom of the frame."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 500, 95
    x0 = (w - panel_w) // 2
    y0 = h - panel_h - 10

    # Determine tier
    if final_pct < 35:
        bg_color = (0, 130, 0)       # green
        label = "STATUS: SAFE (Low Risk)"
    elif final_pct < 70:
        bg_color = (0, 180, 230)     # yellow/orange BGR
        label = "STATUS: SUSPICIOUS (Review Required)"
    else:
        bg_color = (0, 0, 200)       # red
        label = "STATUS: HIGH RISK (Fraud Detected)"

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title + percentage
    title = f"{label}  {final_pct:.1f}%"
    cv2.putText(frame, title, (x0 + 10, y0 + 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y = x0 + 10, y0 + 38
    bar_w, bar_h_px = panel_w - 20, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (50, 50, 50), -1)
    fill_w = int(bar_w * min(max(final_pct, 0), 100) / 100)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), (255, 255, 255), -1)

    # Signal breakdown
    breakdown = f"[DF:{s_df:.2f} | Live:{s_live:.2f} | Chal:{s_chal:.2f} | Cam:{s_cam:.2f}]"
    cv2.putText(frame, breakdown, (x0 + 10, y0 + 75), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def draw_deepfake_overlay(frame: np.ndarray, risk_score: float, trend: str = "") -> None:
    """Draw a deepfake-risk panel at the bottom-left of *frame*."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 380, 60
    x0, y0 = 10, h - panel_h - 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Colour based on threshold (applied to the averaged score)
    if risk_score > 85:
        color = (0, 0, 255)  # red (BGR)
    else:
        color = (0, 200, 0)  # green (BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"DEEPFAKE RISK: {risk_score:.1f}% {trend}"
    cv2.putText(frame, label, (x0 + 10, y0 + 25), font, 0.65, color, 2, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y = x0 + 10, y0 + 38
    bar_w, bar_h_px = panel_w - 20, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (80, 80, 80), -1)
    fill_w = int(bar_w * min(max(risk_score, 0), 100) / 100)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), color, -1)


def draw_proximity_warning(frame: np.ndarray) -> None:
    """Draw a prominent 'MOVE CLOSER' warning banner at the centre of *frame*."""
    h, w = frame.shape[:2]
    banner_w, banner_h = 420, 50
    x0 = (w - banner_w) // 2
    y0 = (h - banner_h) // 2

    # Semi-transparent yellow background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + banner_w, y0 + banner_h), (0, 200, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "WARNING: MOVE CLOSER TO CAMERA"
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    tx = x0 + (banner_w - tw) // 2
    ty = y0 + (banner_h + th) // 2
    cv2.putText(frame, text, (tx, ty), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_challenge_overlay(frame: np.ndarray, text: str, remaining: float,
                           result: str = "") -> None:
    """Draw a challenge instruction or result banner.

    *result* is one of '', 'PASSED', 'FAILED'.
    """
    h, w = frame.shape[:2]
    banner_w, banner_h = 520, 70
    x0 = (w - banner_w) // 2
    y0 = 70  # below the top status bar

    if result == "PASSED":
        bg_color = (0, 180, 0)    # green
        fg_color = (255, 255, 255)
        label = "CHALLENGE PASSED"
    elif result == "FAILED":
        bg_color = (0, 0, 220)    # red
        fg_color = (255, 255, 255)
        label = "CHALLENGE FAILED"
    else:
        bg_color = (200, 130, 0)  # dark cyan / teal-ish
        fg_color = (255, 255, 255)
        label = f"CHALLENGE: {text} ({remaining:.1f}s)"

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + banner_w, y0 + banner_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
    tx = x0 + (banner_w - tw) // 2
    ty = y0 + (banner_h + th) // 2
    cv2.putText(frame, label, (tx, ty), font, 0.8, fg_color, 2, cv2.LINE_AA)


def main():
    # Settings
    EAR_THRESHOLD = 0.21  # below this is considered a blink/closed eye

    # Print mediapipe version for debugging when imports fail
    try:
        mp_version = getattr(mp, '__version__', 'unknown')
    except Exception:
        mp_version = 'unknown'
    print(f"MediaPipe version: {mp_version}")

    # Robust import: prefer legacy `solutions` API when available (face_mesh),
    # otherwise try to use the newer Tasks API (FaceLandmarker) which is present
    # in newer MediaPipe distributions (>=0.10.x).
    use_solutions = False
    try:
        mp_face_mesh = mp.solutions.face_mesh
        use_solutions = True
    except Exception:
        # If solutions isn't available, fall back to tasks API.
        print("\nWARNING: MediaPipe 'solutions' API not available, attempting to use Tasks API (face landmarker)...")

    # If solutions API is available, use the original FaceMesh pipeline.
    if use_solutions:
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        face_landmarker = None
        tasks_mode = False
    else:
        # Try to import Tasks API modules
        try:
            from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker_mod
            from mediapipe.tasks.python.vision.core import image as mp_image_module
            from mediapipe.tasks.python.core import base_options as mp_base_options
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode
        except Exception as e:
            print("Failed to import MediaPipe Tasks API modules:", e)
            print("Available top-level attributes on the 'mediapipe' module:\n", dir(mp))
            print("Please install a MediaPipe distribution that includes either 'solutions' or 'tasks' API.")
            return

        # Determine model path: allow override via env var, otherwise use a default cache path
        import os

        model_path = os.environ.get('MP_FACE_LANDMARKER_MODEL')
        if not model_path:
            model_path = os.path.expanduser('~/.cache/mediapipe/face_landmarker.tflite')

        if not os.path.exists(model_path):
            print(f"FaceLandmarker model not found at: {model_path}")
            print("Please download a MediaPipe FaceLandmarker TFLite model and set the path via the environment variable MP_FACE_LANDMARKER_MODEL,")
            print("or place the model at '~/.cache/mediapipe/face_landmarker.tflite'.")
            print("See: https://developers.google.com/mediapipe/tasks/vision/face_landmarker for model information.")
            return

        # Prepare async callback to receive results from the live landmarker
        import threading

        latest_result = {'landmarks': None}
        result_lock = threading.Lock()

        def _result_callback(result, image, timestamp_ms):
            # result is a FaceLandmarkerResult
            with result_lock:
                latest_result['landmarks'] = result.face_landmarks

        # Create the FaceLandmarker in LIVE_STREAM mode so we can feed frames asynchronously
        base_options = mp_base_options.BaseOptions(model_asset_path=model_path)
        options = mp_face_landmarker_mod.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_running_mode.VisionTaskRunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_result_callback,
        )

        try:
            face_landmarker = mp_face_landmarker_mod.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print("Failed to create FaceLandmarker from model:", e)
            return

        tasks_mode = True

    # --- Virtual camera security check (runs once, before capture) ---
    is_untrusted_cam, cam_name = check_virtual_camera()
    print(f"Detected camera: {cam_name}")
    if is_untrusted_cam:
        print(f"WARNING: Untrusted / external camera detected -> {cam_name}")
        print("A +20% risk penalty will be applied to the deepfake score.")
    else:
        print(f"Trusted camera verified: {cam_name}")

    # Use AVFoundation backend on macOS which tends to be more stable
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    except Exception:
        cap = cv2.VideoCapture(0)

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    blink_count = 0
    blink_in_progress = False

    # --- Deepfake detector setup ---
    detector = DeepfakeDetector()
    print("Deepfake detector loaded.")

    df_job_queue: queue.Queue = queue.Queue(maxsize=1)
    df_result: dict = {"avg_score": 0.0, "trend": "", "frame_id": -1}
    df_result_lock = threading.Lock()
    df_thread = threading.Thread(
        target=_inference_worker,
        args=(detector, df_job_queue, df_result, df_result_lock),
        daemon=True,
    )
    df_thread.start()

    frame_counter = 0
    current_risk_score = 0.0  # smoothed score displayed every frame
    current_trend = ""       # trend indicator displayed every frame
    face_too_far = False     # proximity gate state (persists between checks)
    DEEPFAKE_INTERVAL = 15   # run inference every N frames
    MIN_FACE_PX = 120        # minimum face bbox size to run deepfake inference

    # --- Challenge-Response Liveness ---
    CHALLENGES = ['BLINK TWICE', 'TURN HEAD LEFT', 'TURN HEAD RIGHT', 'LOOK UP']
    current_challenge = None          # active challenge string or None
    challenge_start_time = 0.0        # time.time() when challenge was issued
    challenge_duration = 4.0          # seconds to complete the challenge
    challenge_passed = False          # True for a brief flash after passing
    challenge_failed = False          # True for a brief flash after failing
    challenge_result_time = 0.0       # when the pass/fail was recorded
    CHALLENGE_RESULT_DISPLAY = 1.5    # seconds to show PASSED / FAILED
    CHALLENGE_INTERVAL_MIN = 10.0     # min seconds between challenges
    CHALLENGE_INTERVAL_MAX = 15.0     # max seconds between challenges
    next_challenge_time = time.time() + random.uniform(CHALLENGE_INTERVAL_MIN, CHALLENGE_INTERVAL_MAX)
    blinks_in_challenge = 0           # blink counter specific to challenge
    initial_nose_pos = None           # (x, y) of nose tip at challenge start
    HEAD_DISPLACEMENT_THRESH = 0.05   # fraction of frame width/height

    # --- Unified Risk Engine state ---
    last_blink_time = time.time()     # tracks recency of last blink
    prev_blink_count = 0              # to detect new blinks
    last_challenge_result = 'PENDING' # 'PASSED', 'FAILED', or 'PENDING'

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        # Flip horizontally for a mirror-view (optional, nice for UX)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ear = 0.0
        face_present = False

        if not tasks_mode:
            # Use legacy solutions FaceMesh
            results = face_mesh.process(rgb)

            if results and getattr(results, 'multi_face_landmarks', None):
                face_present = True
                # Use first face
                face_landmarks = results.multi_face_landmarks[0]
                pts = landmarks_to_pixel_list(face_landmarks, w, h)

                left_ear = eye_aspect_ratio(pts, LEFT_EYE)
                right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # Blink state machine: detect rising edge after a low EAR
                if ear < EAR_THRESHOLD and not blink_in_progress:
                    blink_in_progress = True

                if ear >= EAR_THRESHOLD and blink_in_progress:
                    blink_count += 1
                    blink_in_progress = False
                    # Track blinks during active challenge
                    if current_challenge == 'BLINK TWICE':
                        blinks_in_challenge += 1

                # Optional: draw small circles for eye landmarks (clean, unobtrusive)
                for idx in (LEFT_EYE + RIGHT_EYE):
                    (x, y) = pts[idx]
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        else:
            # Tasks API: feed frame asynchronously and read latest callback result
            try:
                mp_img = mp_image_module.Image(mp_image_module.ImageFormat.SRGB, rgb)
                face_landmarker.detect_async(mp_img, int(time.time() * 1000))
            except Exception:
                # Non-fatal: continue, results may arrive later via callback
                pass

            with result_lock:
                lm_list = latest_result.get('landmarks')

            if lm_list and len(lm_list) > 0:
                face_present = True
                face_landmarks = lm_list[0]
                # Convert normalized landmarks to pixel coordinates
                pts = []
                for lm in face_landmarks:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    pts.append((x_px, y_px))

                # Some Tasks models contain 468 points similar to Face Mesh; compute EAR if indices exist
                try:
                    left_ear = eye_aspect_ratio(pts, LEFT_EYE)
                    right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
                    ear = (left_ear + right_ear) / 2.0
                except Exception:
                    ear = 0.0

                # Blink state machine (same as above)
                if ear < EAR_THRESHOLD and not blink_in_progress:
                    blink_in_progress = True

                if ear >= EAR_THRESHOLD and blink_in_progress:
                    blink_count += 1
                    blink_in_progress = False
                    # Track blinks during active challenge
                    if current_challenge == 'BLINK TWICE':
                        blinks_in_challenge += 1

                # Optional: draw small circles for eye landmarks (clean, unobtrusive)
                for idx in (LEFT_EYE + RIGHT_EYE):
                    if idx < len(pts):
                        (x, y) = pts[idx]
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # --- Deepfake inference (every DEEPFAKE_INTERVAL frames) ---
        frame_counter += 1
        if face_present and frame_counter % DEEPFAKE_INTERVAL == 0:
            # Compute face bounding box from landmarks
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 20
            x1 = max(min(xs) - pad, 0)
            y1 = max(min(ys) - pad, 0)
            x2 = min(max(xs) + pad, w)
            y2 = min(max(ys) + pad, h)
            face_w = x2 - x1
            face_h = y2 - y1

            # Proximity gate: skip inference if face is too small
            if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
                face_too_far = True
            else:
                face_too_far = False
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    # Non-blocking put: drop old job if queue full
                    try:
                        df_job_queue.put_nowait((face_crop.copy(), frame_counter))
                    except queue.Full:
                        pass  # previous job still running; skip this frame

        # Reset score buffer and proximity flag when no face is visible
        if not face_present:
            detector.reset_buffer()
            face_too_far = False

        # Store face bbox for heatmap overlay
        face_bbox = None
        if face_present and 'pts' in locals():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 20
            bx1 = max(min(xs) - pad, 0)
            by1 = max(min(ys) - pad, 0)
            bx2 = min(max(xs) + pad, w)
            by2 = min(max(ys) + pad, h)
            face_bbox = (bx1, by1, bx2, by2)

        # Read latest smoothed risk score + trend
        with df_result_lock:
            current_risk_score = df_result["avg_score"]   # already 0-100%
            current_trend = df_result["trend"]

        # ================================================================
        # Unified Fraud Risk Engine — aggregate all signals
        # ================================================================
        now_risk = time.time()

        # S_df: deepfake signal (0-1)
        s_df = min(current_risk_score / 100.0, 1.0)

        # S_live: liveness signal based on time since last blink
        time_since_blink = now_risk - last_blink_time
        if not face_present:
            s_live = 1.0
        elif time_since_blink < 3.0:
            s_live = 0.0
        elif time_since_blink < 8.0:
            s_live = (time_since_blink - 3.0) / 5.0  # linear 0→1
        else:
            s_live = 1.0

        # S_chal: challenge signal
        if last_challenge_result == 'PASSED':
            s_chal = 0.0
        elif last_challenge_result == 'FAILED':
            s_chal = 1.0
        else:  # PENDING
            s_chal = 0.5

        # S_cam: camera trust signal
        s_cam = 1.0 if is_untrusted_cam else 0.0

        # Weighted aggregation
        raw_risk = (s_df * W_DEEPFAKE + s_live * W_LIVENESS +
                    s_chal * W_CHALLENGE + s_cam * W_CAMERA)

        # Critical penalty override (high-confidence deepfake, virtual cam, or failed challenge)
        if s_df > 0.70 or s_cam == 1.0 or s_chal == 1.0:
            raw_risk = max(raw_risk, 0.85)

        final_risk_score = raw_risk * 100.0

        # --- Draw top status bar ---
        if final_risk_score >= 70:
            status_text = "HIGH RISK - FRAUD DETECTED"
            status_color = (0, 0, 255)
        elif final_risk_score >= 35:
            status_text = "SUSPICIOUS - REVIEW REQUIRED"
            status_color = (0, 180, 230)
        else:
            if blink_count > 0:
                status_text = "LIVENESS VERIFIED"
                status_color = (0, 180, 0)
            else:
                status_text = "SCANNING LIVENESS"
                status_color = (0, 200, 255)
        draw_overlay(frame, status_text, status_color, blink_count)

        # --- Draw unified risk panel ---
        draw_unified_risk_panel(frame, final_risk_score, s_df, s_live, s_chal, s_cam)

        # --- GradCAM XAI heatmap overlay ---
        with detector._heatmap_lock:
            heatmap = detector.current_heatmap
        if heatmap is not None and face_bbox is not None:
            bx1, by1, bx2, by2 = face_bbox
            fw, fh = bx2 - bx1, by2 - by1
            if fw > 0 and fh > 0:
                cam_resized = cv2.resize(heatmap, (fw, fh))
                cam_color = cv2.applyColorMap(
                    np.uint8(255 * cam_resized), cv2.COLORMAP_JET
                )
                face_region = frame[by1:by2, bx1:bx2]
                blended = cv2.addWeighted(face_region, 0.55, cam_color, 0.45, 0)
                frame[by1:by2, bx1:bx2] = blended
                cv2.putText(frame, "AI Artifact Detection Map", (bx1, by1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

        # Proximity warning overlay
        if face_too_far:
            draw_proximity_warning(frame)

        # ---------------------------------------------------------------
        # Challenge-Response Liveness Logic
        # ---------------------------------------------------------------
        now = time.time()

        # Track last blink time for liveness signal
        if blink_count > prev_blink_count:
            last_blink_time = now
            prev_blink_count = blink_count

        # Show pass/fail result flash for CHALLENGE_RESULT_DISPLAY seconds
        if challenge_passed and (now - challenge_result_time < CHALLENGE_RESULT_DISPLAY):
            draw_challenge_overlay(frame, "", 0, result="PASSED")
        elif challenge_failed and (now - challenge_result_time < CHALLENGE_RESULT_DISPLAY):
            draw_challenge_overlay(frame, "", 0, result="FAILED")
        else:
            # Clear result flags after display time
            challenge_passed = False
            challenge_failed = False

            # --- Trigger a new challenge ---
            if current_challenge is None and face_present and now >= next_challenge_time:
                current_challenge = random.choice(CHALLENGES)
                challenge_start_time = now
                blinks_in_challenge = 0
                if len(pts) > 1:
                    initial_nose_pos = (pts[1][0], pts[1][1])
                else:
                    initial_nose_pos = None

            # --- Active challenge: verify & draw ---
            if current_challenge is not None:
                elapsed = now - challenge_start_time
                remaining = max(challenge_duration - elapsed, 0.0)

                passed = False

                if current_challenge == 'BLINK TWICE':
                    if blinks_in_challenge >= 2:
                        passed = True

                elif current_challenge in ('TURN HEAD LEFT', 'TURN HEAD RIGHT'):
                    if initial_nose_pos and face_present and len(pts) > 1:
                        curr_nose_x = pts[1][0]
                        dx = curr_nose_x - initial_nose_pos[0]
                        threshold_px = w * HEAD_DISPLACEMENT_THRESH
                        if current_challenge == 'TURN HEAD LEFT' and dx < -threshold_px:
                            passed = True
                        elif current_challenge == 'TURN HEAD RIGHT' and dx > threshold_px:
                            passed = True

                elif current_challenge == 'LOOK UP':
                    if initial_nose_pos and face_present and len(pts) > 1:
                        curr_nose_y = pts[1][1]
                        dy = initial_nose_pos[1] - curr_nose_y
                        threshold_py = h * HEAD_DISPLACEMENT_THRESH
                        if dy > threshold_py:
                            passed = True

                if passed:
                    challenge_passed = True
                    challenge_result_time = now
                    current_challenge = None
                    last_challenge_result = 'PASSED'
                    next_challenge_time = now + random.uniform(CHALLENGE_INTERVAL_MIN, CHALLENGE_INTERVAL_MAX)
                    draw_challenge_overlay(frame, "", 0, result="PASSED")
                elif remaining <= 0:
                    challenge_failed = True
                    challenge_result_time = now
                    current_challenge = None
                    last_challenge_result = 'FAILED'
                    next_challenge_time = now + random.uniform(CHALLENGE_INTERVAL_MIN, CHALLENGE_INTERVAL_MAX)
                    draw_challenge_overlay(frame, "", 0, result="FAILED")
                else:
                    draw_challenge_overlay(frame, current_challenge, remaining)

        # Show EAR and face presence as a small helper
        helper_text = f"EAR: {ear:.3f}  Face: {'Yes' if face_present else 'No'}"
        cv2.putText(frame, helper_text, (12, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    # Stop the deepfake inference thread
    try:
        df_job_queue.put(None)  # poison pill
        df_thread.join(timeout=2)
    except Exception:
        pass

    # If Tasks API landmarker was created, close it
    try:
        if tasks_mode and face_landmarker is not None:
            face_landmarker.close()
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
