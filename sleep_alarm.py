"""
Sleep Alarm - Eye Aspect Ratio (EAR) Drowsiness Detector
Compatible with MediaPipe 0.10.x (new Tasks API)

Requirements:
    pip install opencv-python mediapipe scipy pygame numpy requests

Usage:
    python sleep_alarm.py
    python sleep_alarm.py --alarm path/to/alarm.wav
    python sleep_alarm.py --ear 0.20 --frames 25 --cam 0
"""

import cv2
import numpy as np
import pygame
import argparse
import os
import sys
import time
import urllib.request
from scipy.spatial import distance as dist

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base

# ─────────────────────────────────────────────────────────────
# New MediaPipe FaceLandmarker eye indices (478-point model)
# Each eye uses 6 specific points for EAR calculation
# ─────────────────────────────────────────────────────────────
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 380, 373]

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def download_model():
    """Download the MediaPipe face landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Downloading face landmarker model (~30 MB, one-time)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        print(f"        Please manually download from:\n        {MODEL_URL}")
        print(f"        and place it in the same folder as sleep_alarm.py")
        sys.exit(1)


def ear(eye_pts: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 landmark points.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    ~0.25-0.35 when open, drops below threshold when closed.
    """
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)


def get_eye_pts(landmarks, indices, img_w, img_h):
    """Extract pixel (x, y) from landmark list by indices."""
    return np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in indices
    ], dtype=np.float64)


def init_alarm(path: str):
    """Init pygame mixer and load alarm sound."""
    pygame.mixer.init()
    if os.path.exists(path):
        try:
            pygame.mixer.music.load(path)
            print(f"[INFO] Alarm loaded: {path}")
            return True
        except Exception as e:
            print(f"[WARN] Could not load '{path}': {e}")
    else:
        print(f"[WARN] alarm.wav not found. Using terminal beep fallback.")
    return False


def play_alarm(alarm_loaded: bool):
    if alarm_loaded:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
    else:
        sys.stdout.write('\a')
        sys.stdout.flush()


def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


def draw_eye_contour(frame, pts, color):
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.drawContours(frame, [hull], -1, color, 1)


def draw_hud(frame, avg_ear, counter, alarm_on, threshold, max_frames):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

    ear_color = (0, 255, 100) if avg_ear >= threshold else (0, 80, 255)
    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ear_color, 2)

    bar_w = int((min(counter, max_frames) / max_frames) * 200)
    cv2.rectangle(frame, (12, 40), (212, 56), (60, 60, 60), -1)
    bar_color = (0, 200, 80) if not alarm_on else (0, 60, 220)
    cv2.rectangle(frame, (12, 40), (12 + bar_w, 56), bar_color, -1)
    cv2.putText(frame, f"Closed: {counter}/{max_frames}", (14, 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    if alarm_on:
        pulse = int(abs(np.sin(time.time() * 4)) * 60)
        cv2.rectangle(frame, (w - 160, 8), (w - 10, 50), (0, 0, 180 + pulse), -1)
        cv2.putText(frame, "!! WAKE UP !!", (w - 155, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Q = quit   R = reset   +/- = adjust threshold",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)


def main():
    parser = argparse.ArgumentParser(description="Sleep Alarm")
    parser.add_argument("--alarm",  default="alarm.wav")
    parser.add_argument("--ear",    type=float, default=0.21)
    parser.add_argument("--frames", type=int,   default=20)
    parser.add_argument("--cam",    type=int,   default=0)
    args = parser.parse_args()

    EAR_THRESHOLD = args.ear
    MAX_FRAMES    = args.frames

    # Download model if needed
    download_model()

    # Init alarm
    alarm_loaded = init_alarm(args.alarm)

    # Init MediaPipe FaceLandmarker (new Tasks API)
    base_opts = mp_base.BaseOptions(model_asset_path=MODEL_PATH)
    options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    closed_counter = 0
    alarm_on       = False
    no_face_frames = 0

    print("=" * 50)
    print("  Sleep Alarm running  |  Press Q to quit")
    print(f"  EAR threshold: {EAR_THRESHOLD}   Frames: {MAX_FRAMES}")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        avg_ear = 0.0

        # Convert to MediaPipe Image
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = face_landmarker.detect(mp_img)

        if result.face_landmarks:
            no_face_frames = 0
            lm = result.face_landmarks[0]   # first face

            l_pts = get_eye_pts(lm, LEFT_EYE,  w, h)
            r_pts = get_eye_pts(lm, RIGHT_EYE, w, h)

            l_ear   = ear(l_pts)
            r_ear   = ear(r_pts)
            avg_ear = (l_ear + r_ear) / 2.0

            eye_color = (0, 230, 100) if avg_ear >= EAR_THRESHOLD else (0, 60, 255)
            draw_eye_contour(frame, l_pts, eye_color)
            draw_eye_contour(frame, r_pts, eye_color)

            if avg_ear < EAR_THRESHOLD:
                closed_counter += 1
                if closed_counter >= MAX_FRAMES:
                    alarm_on = True
                    play_alarm(alarm_loaded)
            else:
                closed_counter = 0
                alarm_on       = False
                stop_alarm()

        else:
            no_face_frames += 1
            if no_face_frames > 60:
                alarm_on = False
                stop_alarm()
            cv2.putText(frame, "No face detected", (w // 2 - 110, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 200), 2)

        draw_hud(frame, avg_ear, closed_counter, alarm_on, EAR_THRESHOLD, MAX_FRAMES)
        cv2.imshow("Sleep Alarm", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            closed_counter = 0
            alarm_on       = False
            stop_alarm()
        elif key in (ord('+'), ord('=')):
            EAR_THRESHOLD = min(0.40, EAR_THRESHOLD + 0.01)
            print(f"  Threshold -> {EAR_THRESHOLD:.2f}")
        elif key == ord('-'):
            EAR_THRESHOLD = max(0.10, EAR_THRESHOLD - 0.01)
            print(f"  Threshold -> {EAR_THRESHOLD:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    stop_alarm()
    pygame.mixer.quit()
    print("Sleep Alarm stopped.")


if __name__ == "__main__":
    main()
