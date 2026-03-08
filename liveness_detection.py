#!/usr/bin/env python3
"""
Real-time liveness detection using OpenCV + MediaPipe Face Mesh.

Features:
- Capture default webcam (optimized for macOS using AVFoundation backend)
- Use MediaPipe Face Mesh to obtain facial landmarks
- Compute Eye Aspect Ratio (EAR) to detect blinks
- UI overlay: "SCANNING LIVENESS" (yellow) -> "LIVENESS VERIFIED" (green) after 1 blink
- Blink counter displayed
- Clean camera release on 'q'

Usage:
    python3 liveness_detection.py

Dependencies:
    mediapipe, opencv-python, numpy

"""
import math
import time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


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
    counter_text = f"Blinks: {blink_count}"
    (ctw, cth), _ = cv2.getTextSize(counter_text, font, 0.7, 2)
    cv2.putText(frame, counter_text, (w - ctw - 16, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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

        # Determine UI status
        if blink_count > 0:
            status_text = "LIVENESS VERIFIED"
            status_color = (0, 180, 0)  # green-ish
        else:
            status_text = "SCANNING LIVENESS"
            status_color = (0, 200, 255)  # yellow-ish (BGR)

        draw_overlay(frame, status_text, status_color, blink_count)

        # Show EAR and face presence as a small helper
        helper_text = f"EAR: {ear:.3f}  Face: {'Yes' if face_present else 'No'}"
        cv2.putText(frame, helper_text, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
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
