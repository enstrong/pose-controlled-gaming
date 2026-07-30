import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/gait_classifier.pkl")
MIN_VISIBILITY = 0.35
REQUIRED_LANDMARKS = [11, 12, 23, 24, 25, 26]   # shoulders, hips, knees


def load_model():
    with MODEL_PATH.open("rb") as f:
        data = pickle.load(f)
    return data["classifier"], data["window_frames"]


def get_y(name, frame_buffer):
    return np.array([row[f"{name}_y"] for row in frame_buffer], dtype=np.float32)

def get_x(name, frame_buffer):
    return np.array([row[f"{name}_x"] for row in frame_buffer], dtype=np.float32)


def extract_features_from_buffer(frame_buffer):
    def y(name): return get_y(name, frame_buffer)
    def x(name): return get_x(name, frame_buffer)

    ls_y  = y("left_shoulder");  rs_y  = y("right_shoulder")
    lh_y  = y("left_hip");       rh_y  = y("right_hip")
    lk_y  = y("left_knee");      rk_y  = y("right_knee")
    la_y  = y("left_ankle");     ra_y  = y("right_ankle")
    lf_y  = y("left_foot_index"); rf_y = y("right_foot_index")
    lh_x  = x("left_hip");       rh_x  = x("right_hip")

    shoulder_diff      = ls_y - rs_y
    shoulder_asym_mean = float(np.mean(np.abs(shoulder_diff)))
    shoulder_asym_std  = float(np.std(shoulder_diff))
    hip_center_x       = (lh_x + rh_x) / 2.0
    hip_sway           = float(np.std(hip_center_x))
    hip_diff_mean      = float(np.mean(np.abs(lh_y - rh_y)))
    lkr  = float(np.max(lk_y) - np.min(lk_y))
    rkr  = float(np.max(rk_y) - np.min(rk_y))
    krr  = abs(lkr - rkr) / (max(lkr, rkr) + 1e-6)
    kds  = float(np.std(np.abs(lk_y - rk_y)))
    kdm  = float(np.mean(np.abs(lk_y - rk_y)))
    lar  = float(np.max(la_y) - np.min(la_y))
    rar  = float(np.max(ra_y) - np.min(ra_y))
    arr  = abs(lar - rar) / (max(lar, rar) + 1e-6)
    lfr  = float(np.max(lf_y) - np.min(lf_y))
    rfr  = float(np.max(rf_y) - np.min(rf_y))
    ffrr = abs(lfr - rfr) / (max(lfr, rfr) + 1e-6)
    vm   = float(np.std((ls_y + rs_y) / 2.0))
    corr = float(np.corrcoef(lk_y, rk_y)[0, 1]) \
           if np.std(lk_y) > 1e-6 and np.std(rk_y) > 1e-6 else 0.0
    kac  = -corr

    return np.array([[
        shoulder_asym_mean, shoulder_asym_std, hip_sway, hip_diff_mean,
        lkr, rkr, krr, kds, kdm, lar, rar, arr, lfr, rfr, ffrr, vm, kac,
    ]], dtype=np.float32)


def lm_to_dict(landmarks):
    names = [
        "nose","left_eye_inner","left_eye","left_eye_outer",
        "right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_pinky","right_pinky",
        "left_index","right_index","left_thumb","right_thumb",
        "left_hip","right_hip","left_knee","right_knee",
        "left_ankle","right_ankle","left_heel","right_heel",
        "left_foot_index","right_foot_index",
    ]
    row = {}
    for name, lm in zip(names, landmarks):
        row[f"{name}_x"] = lm.x
        row[f"{name}_y"] = lm.y
        row[f"{name}_z"] = lm.z
    return row


def main():
    clf, window_frames = load_model()
    print(f"Model loaded. Window: {window_frames} frames.")

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    pose       = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam not found")
        pose.close()
        return

    time.sleep(2)
    for _ in range(10): cap.read()

    print("Gait analyzer running.")
    print("Walk toward the camera. Verdict updates every second.")
    print("Press Q to quit.\n")

    frame_buffer = deque(maxlen=window_frames)

    verdict_history = deque(maxlen=5)
    current_verdict = "Collecting data..."
    verdict_conf    = 0.0
    frames_collected = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            h, w    = frame.shape[:2]

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
                lms = results.pose_landmarks.landmark

                vis_ok = all(
                    lms[i].visibility >= MIN_VISIBILITY
                    for i in REQUIRED_LANDMARKS
                )

                if vis_ok:
                    frame_buffer.append(lm_to_dict(lms))
                    frames_collected += 1

                    if len(frame_buffer) == window_frames:
                        feats = extract_features_from_buffer(frame_buffer)
                        pred  = str(clf.predict(feats)[0])
                        conf  = 0.0
                        if hasattr(clf, "predict_proba"):
                            conf = float(np.max(clf.predict_proba(feats)))

                        verdict_history.append((pred, conf))

                        # Majority vote over recent verdicts
                        from collections import Counter
                        votes = Counter(v for v, _ in verdict_history)
                        current_verdict = votes.most_common(1)[0][0]
                        verdict_conf    = np.mean([
                            c for v, c in verdict_history
                            if v == current_verdict
                        ])
                else:
                    cv2.putText(frame, "Move back — body not fully visible",
                                (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 100, 255), 2, cv2.LINE_AA)

            if current_verdict == "NORMAL_GAIT":
                verdict_color = (50, 220, 50)
                verdict_display = "NORMAL GAIT"
            elif current_verdict == "ASYMMETRIC_GAIT":
                verdict_color = (50, 50, 255)
                verdict_display = "ASYMMETRIC GAIT"
            else:
                verdict_color  = (180, 180, 180)
                verdict_display = current_verdict

            fill_ratio = len(frame_buffer) / window_frames
            bar_w      = int(w * 0.6)
            filled_w   = int(bar_w * fill_ratio)
            cv2.rectangle(frame, (20, 20), (20 + bar_w, 44), (60, 60, 60), -1)
            cv2.rectangle(frame, (20, 20), (20 + filled_w, 44), (100, 200, 100), -1)
            cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{window_frames}",
                        (bar_w + 30, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.putText(frame, verdict_display,
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, verdict_color, 5, cv2.LINE_AA)

            if verdict_conf > 0:
                cv2.putText(frame, f"confidence: {verdict_conf:.2f}",
                            (20, 148), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, verdict_color, 3, cv2.LINE_AA)

            cv2.imshow("Gait Analysis — pose-controlled-gaming", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Closed cleanly.")


if __name__ == "__main__":
    main()