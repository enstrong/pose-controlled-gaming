import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

OUTPUT_PATH = "data/gait_data.csv"
MIN_VISIBILITY = 0.35
PREP_SECONDS = 5.0      # time to walk to the far end of the room
RECORD_SECONDS = 5.0    # time to walk toward camera

LABEL_CONFIG = {
    ord("n"): "NORMAL_GAIT",
    ord("a"): "ASYMMETRIC_GAIT",
}

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

REQUIRED_LANDMARKS = [
    11, 12,  # shoulders
    23, 24,  # hips
    25, 26,  # knees
]

HEADER = ["label", "timestamp"]
for name in LANDMARK_NAMES:
    HEADER += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]


def pose_ok(landmarks):
    return all(
        landmarks[i].visibility >= MIN_VISIBILITY
        for i in REQUIRED_LANDMARKS
    )


def write_row(writer, label, landmarks):
    row = [label, datetime.now().isoformat()]
    for lm in landmarks:
        row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6), round(lm.visibility, 4)]
    writer.writerow(row)


def draw_overlay(frame, mode_text, counts, total, skipped, prep_remaining=None):
    h, w = frame.shape[:2]

    # Big status text — readable from across the room
    color = (0, 255, 100) if "RECORDING" in mode_text else \
            (0, 255, 255) if "GET READY" in mode_text else \
            (180, 180, 180)

    cv2.putText(frame, mode_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5, cv2.LINE_AA)

    if prep_remaining is not None:
        cv2.putText(frame, f"Walk to far end: {prep_remaining:.1f}s",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 200, 255), 3, cv2.LINE_AA)

    y = 190
    for label_name, count in counts.items():
        # Green if 7+ recordings, yellow otherwise
        col = (0, 255, 0) if count >= 7 else (255, 255, 100)
        cv2.putText(frame, f"{label_name}: {count} recordings",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, col, 3, cv2.LINE_AA)
        y += 50

    cv2.putText(frame, f"Frames saved: {total}  skipped: {skipped}",
                (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "N = NORMAL    A = ASYMMETRIC    Q = quit",
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (200, 200, 200), 2, cv2.LINE_AA)


def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_PATH)
    csv_file = open(OUTPUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(HEADER)
        print(f"Created: {OUTPUT_PATH}")
    else:
        print(f"Appending to: {OUTPUT_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam not open")
        csv_file.close()
        pose.close()
        return

    time.sleep(2)
    for _ in range(10):
        cap.read()

    counts = {"NORMAL_GAIT": 0, "ASYMMETRIC_GAIT": 0}
    total_frames = 0
    skipped = 0

    state = "IDLE"
    current_label = None
    state_started_at = None

    print("\nControls:")
    print("  N = NORMAL_GAIT      A = ASYMMETRIC_GAIT      Q = quit")
    print("\nWorkflow per recording:")
    print(f"  1. Press N or A")
    print(f"  2. You have {PREP_SECONDS:.0f}s to walk to the far end of the room")
    print(f"  3. Turn and walk toward the camera — {RECORD_SECONDS:.0f}s of recording")
    print(f"  4. Repeat. Aim for 7+ recordings per class.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            now = time.monotonic()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting.")
                break

            if state == "IDLE" and key in LABEL_CONFIG:
                current_label = LABEL_CONFIG[key]
                state = "PREP"
                state_started_at = now
                print(f"\nGet ready: {current_label}")
                print(f"Walk to the far end of the room now...")

            if state == "PREP":
                elapsed = now - state_started_at
                remaining = PREP_SECONDS - elapsed
                if remaining <= 0:
                    state = "RECORD"
                    state_started_at = now
                    print(f"RECORDING {current_label} — walk toward camera!")
                mode_text = f"GET READY: {current_label[:6]}"
                prep_remaining = max(0.0, remaining)
            elif state == "RECORD":
                elapsed = now - state_started_at
                remaining = RECORD_SECONDS - elapsed

                if remaining <= 0:
                    counts[current_label] += 1
                    print(f"Done. {current_label} recording #{counts[current_label]}")
                    state = "IDLE"
                    current_label = None
                    state_started_at = None

                    mode_text = "IDLE — press N or A"
                    prep_remaining = None
                else:
                    mode_text = f"RECORDING {current_label[:6]} {remaining:.1f}s"
                    prep_remaining = None
            else:
                mode_text = "IDLE — press N or A"
                prep_remaining = None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
                lms = results.pose_landmarks.landmark

                if state == "RECORD":
                    if pose_ok(lms):
                        write_row(writer, current_label, lms)
                        total_frames += 1
                    else:
                        skipped += 1
                        cv2.putText(frame, "POSE WEAK — keep body visible",
                                    (20, frame.shape[0] - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if state == "RECORD":
                    skipped += 1
                cv2.putText(frame, "NO POSE DETECTED",
                            (20, frame.shape[0] - 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3, cv2.LINE_AA)

            draw_overlay(frame, mode_text, counts, total_frames, skipped, prep_remaining)
            cv2.imshow("Gait Data Collection — pose-controlled-gaming", frame)

    finally:
        csv_file.flush()
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    print("\nFinal counts:")
    for label, count in counts.items():
        print(f"  {label}: {count} recordings")
    print(f"Total frames saved: {total_frames}")
    print(f"Frames skipped (low visibility): {skipped}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()