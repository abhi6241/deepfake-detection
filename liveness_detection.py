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


# Face Mesh landmark indices for eyes (MediaPipe Face Mesh)
# Left eye: [33, 160, 158, 133, 153, 144]
# Right eye: [263, 387, 385, 362, 380, 373]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


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
# Deepfake Detector (HuggingFace ViT + MPS acceleration)
# ---------------------------------------------------------------------------
MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"


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
        self.score_buffer: collections.deque = collections.deque(maxlen=25)
        self.current_avg_score: float = 0.0
        self.trend: str = ""

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
        prev_avg = self.current_avg_score
        self.score_buffer.append(prob * 100.0)  # store as percentage
        self.current_avg_score = sum(self.score_buffer) / len(self.score_buffer)

        delta = self.current_avg_score - prev_avg
        if delta > 2.0:
            self.trend = "⬆️"
        elif delta < -2.0:
            self.trend = "⬇️"
        else:
            self.trend = "➖"


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
    # Blink counter / risk display at top-right
    # If blink_count is used as a risk score (100 for critical), display as Risk: 100%
    if blink_count is None:
        counter_text = ""
    elif isinstance(blink_count, int) and blink_count >= 100:
        counter_text = f"Risk: {blink_count}%"
    else:
        counter_text = f"Blinks: {blink_count}"
    (ctw, cth), _ = cv2.getTextSize(counter_text, font, 0.7, 2)
    cv2.putText(frame, counter_text, (w - ctw - 16, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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
    if risk_score > 70:
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

    # SECURITY CHECK: enumerate available camera devices on macOS and detect virtual cameras
    def enumerate_cameras_mac() -> List[str]:
        """Return a list of camera device names as reported by system_profiler (macOS).

        Uses `system_profiler SPCameraDataType -json` when available, otherwise falls back to
        parsing the plain-text output. Returns an empty list on non-macOS or failure.
        """
        if os.name != 'posix':
            return []

        try:
            out = subprocess.check_output(['system_profiler', 'SPCameraDataType', '-json'], text=True, stderr=subprocess.DEVNULL)
            data = json.loads(out)
            cameras = []
            for entry in data.get('SPCameraDataType', []):
                name = entry.get('_name') or entry.get('camera') or None
                if not name:
                    for v in entry.values():
                        if isinstance(v, str):
                            name = v
                            break
                if name:
                    cameras.append(name)
            return cameras
        except Exception:
            try:
                out = subprocess.check_output(['system_profiler', 'SPCameraDataType'], text=True, stderr=subprocess.DEVNULL)
                cameras = []
                for line in out.splitlines():
                    if line and not line.startswith(' ') and line.strip().endswith(':'):
                        cameras.append(line.strip().rstrip(':'))
                return cameras
            except Exception:
                return []

    def is_suspicious_camera_name(name: str) -> bool:
        if not name:
            return False
        s = name.lower()
        tokens = ['virtual', 'obs', 'v-camera', 'v camera', 'manycam', 'snap camera', 'vcam', 'v-cam']
        return any(t in s for t in tokens)

    # Run camera enumeration & security check before opening the capture device
    security_alert = False
    risk_score = 0
    selected_camera_name = None
    try:
        cameras = enumerate_cameras_mac()
        try:
            selected_index = int(os.environ.get('VIDEO_DEVICE_INDEX', '0'))
        except Exception:
            selected_index = 0

        if cameras:
            if 0 <= selected_index < len(cameras):
                selected_camera_name = cameras[selected_index]
            else:
                selected_camera_name = cameras[0]

            if is_suspicious_camera_name(selected_camera_name):
                print("CRITICAL SECURITY ALERT: VIRTUAL CAMERA DETECTED ->", selected_camera_name)
                security_alert = True
                risk_score = 100
        else:
            selected_camera_name = None
    except Exception:
        security_alert = False
        risk_score = 0

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

        # Read latest smoothed risk score + trend
        with df_result_lock:
            current_risk_score = df_result["avg_score"]   # already 0-100%
            current_trend = df_result["trend"]

        # Determine UI status (security alert overrides liveness)
        if 'security_alert' in locals() and security_alert:
            status_text = "CRITICAL SECURITY ALERT: VIRTUAL CAMERA DETECTED"
            status_color = (0, 0, 255)  # red
            # Pass the risk_score into the overlay so it shows "Risk: 100%"
            draw_overlay(frame, status_text, status_color, int(risk_score))
        else:
            if blink_count > 0:
                status_text = "LIVENESS VERIFIED"
                status_color = (0, 180, 0)  # green-ish
            else:
                status_text = "SCANNING LIVENESS"
                status_color = (0, 200, 255)  # yellow-ish (BGR)

            draw_overlay(frame, status_text, status_color, blink_count)
        draw_deepfake_overlay(frame, current_risk_score, current_trend)

        # Proximity warning overlay
        if face_too_far:
            draw_proximity_warning(frame)

        # Show EAR and face presence as a small helper
        helper_text = f"EAR: {ear:.3f}  Face: {'Yes' if face_present else 'No'}"
        cv2.putText(frame, helper_text, (12, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

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
