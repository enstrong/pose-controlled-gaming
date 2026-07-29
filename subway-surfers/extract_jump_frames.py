import argparse
import csv
import time
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from pathlib import Path

LIFT_THRESHOLD = 0.10
BASELINE_FRAMES = 60
OUTPUT_PATH = "data/gesture_data.csv"

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HEADER = ["label", "timestamp"]
for name in LANDMARK_NAMES:
    HEADER += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]


def foot_y(landmarks):
    lf = landmarks[31]
    rf = landmarks[32]
    return (lf.y + rf.y) / 2.0


def write_row(writer, landmarks):
    row = ["JUMP", datetime.now().isoformat()]
    for lm in landmarks:
        row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6), round(lm.visibility, 4)]
    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to jump video file")
    parser.add_argument("--threshold", type=float, default=LIFT_THRESHOLD)
    args = parser.parse_args()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} fps")
    print(f"Computing baseline from first {BASELINE_FRAMES} frames...")

    baseline_values = []
    frame_idx = 0
    while frame_idx < BASELINE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            baseline_values.append(foot_y(results.pose_landmarks.landmark))
        frame_idx += 1

    if not baseline_values:
        print("ERROR: No pose detected in baseline frames. Stand still at the start of the video.")
        return

    baseline = float(np.mean(baseline_values))
    print(f"Baseline foot y: {baseline:.4f}")
    print(f"Airborne threshold: foot_y < {baseline - args.threshold:.4f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    import os
    file_exists = os.path.exists(OUTPUT_PATH)
    csv_file = open(OUTPUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(HEADER)

    kept = 0
    skipped = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            fy = foot_y(lms)
            if fy < (baseline - args.threshold):
                write_row(writer, lms)
                kept += 1
            else:
                skipped += 1
        else:
            skipped += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{total_frames} — kept {kept}, skipped {skipped}")

    csv_file.flush()
    csv_file.close()
    cap.release()
    pose.close()

    print(f"\nDone.")
    print(f"Airborne frames kept (JUMP): {kept}")
    print(f"Non-airborne frames skipped: {skipped}")
    print(f"Appended to: {OUTPUT_PATH}")
    print(f"\nNow delete all old JUMP rows from your CSV and retrain.")


if __name__ == "__main__":
    main()