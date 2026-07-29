import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp


OUTPUT_PATH = "data/gesture_data.csv"
MIN_VISIBILITY = 0.45
COUNTDOWN_SECONDS = 3

LABEL_CONFIG = {
    ord("s"): {"label": "STOP",   "seconds": 8.0},
    ord("l"): {"label": "LEFT",   "seconds": 10.0},
    ord("r"): {"label": "RIGHT",  "seconds": 10.0},
    ord("n"): {"label": "RUN",    "seconds": 10.0},
    ord("c"): {"label": "CROUCH", "seconds": 4.0},
    ord("j"): {"label": "JUMP",   "seconds": 0.7},
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

IMPORTANT_LANDMARKS = [
    11, 12,  # shoulders
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
]
# head, shoulders, knees and toes, knees and toes(elite reference)

HEADER = ["label", "timestamp"]
for name in LANDMARK_NAMES:
    HEADER += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]


def pose_is_visible(landmarks):
    return all(landmarks[index].visibility >= MIN_VISIBILITY for index in IMPORTANT_LANDMARKS)


def write_pose_row(writer, label, landmarks):
    row = [label, datetime.now().isoformat()]
    for landmark in landmarks:
        row += [
            round(landmark.x, 6),
            round(landmark.y, 6),
            round(landmark.z, 6),
            round(landmark.visibility, 4),
        ]
    writer.writerow(row)


def draw_status(frame, mode_text, label_counts, frame_count, skipped_count):
    cv2.putText(
        frame,
        mode_text,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    y_pos = 105
    for label_name, count in label_counts.items():
        color = (0, 255, 0) if count >= 180 else (255, 255, 100)
        cv2.putText(
            frame,
            f"{label_name}: {count}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        y_pos += 32

    cv2.putText(
        frame,
        f"Saved: {frame_count}  Skipped low-vis: {skipped_count}",
        (20, y_pos + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_PATH)
    csv_file = open(OUTPUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(HEADER)
        print(f"Created new dataset: {OUTPUT_PATH}")
    else:
        print(f"Appending to existing dataset: {OUTPUT_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        csv_file.close()
        pose.close()
        return

    time.sleep(2)
    for _ in range(10):
        cap.read()

    label_counts = {config["label"]: 0 for config in LABEL_CONFIG.values()}
    frame_count = 0
    skipped_count = 0
    capture = None

    print("\nCapture controls:")
    print("  S = STOP    1.5s — stand completely still, center frame")
    print("  L = LEFT    1.5s — walk left, pause, walk back, pause, repeat")
    print("  R = RIGHT   1.5s — walk right, pause, walk back, pause, repeat")
    print("  N = RUN    10.0s — run in place, center frame")
    print("  C = CROUCH  4.0s — hold a crouch")
    print("  J = JUMP    0.7s after countdown — one jump, fully in air")
    print("  Q = quit and save")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            now = time.monotonic()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting and saving.")
                break
            if capture is None and key in LABEL_CONFIG:
                config = LABEL_CONFIG[key]
                capture = {
                    "label": config["label"],
                    "record_seconds": config["seconds"],
                    "countdown_started_at": now,
                    "record_started_at": None,
                }
                print(f"Get ready for {config['label']}...")

            mode_text = "IDLE - press S/L/M/R/C/J"
            should_record = False

            if capture is not None:
                elapsed_countdown = now - capture["countdown_started_at"]
                countdown_remaining = COUNTDOWN_SECONDS - elapsed_countdown

                if countdown_remaining > 0:
                    mode_text = f"{capture['label']} in {int(countdown_remaining) + 1}"
                else:
                    if capture["record_started_at"] is None:
                        capture["record_started_at"] = now
                        print(f"Recording {capture['label']} now.")

                    elapsed_recording = now - capture["record_started_at"]
                    remaining = capture["record_seconds"] - elapsed_recording
                    if remaining > 0:
                        should_record = True
                        mode_text = f"RECORDING {capture['label']} {remaining:.1f}s"
                    else:
                        print(f"Finished {capture['label']}.")
                        capture = None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                landmarks = results.pose_landmarks.landmark
                if should_record:
                    if pose_is_visible(landmarks):
                        write_pose_row(writer, capture["label"], landmarks)
                        label_counts[capture["label"]] += 1
                        frame_count += 1
                    else:
                        skipped_count += 1
                        cv2.putText(
                            frame,
                            "LOW VISIBILITY - frame skipped",
                            (20, 430),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA,
                        )
            else:
                if should_record:
                    skipped_count += 1
                cv2.putText(
                    frame,
                    "No person detected",
                    (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            h, w = frame.shape[:2]
            cv2.line(frame, (w // 3, 0), (w // 3, h), (200, 200, 200), 1)
            cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (200, 200, 200), 1)
            draw_status(frame, mode_text, label_counts, frame_count, skipped_count)
            cv2.imshow("Data Collection - Subway Surfers CV", frame)
    finally:
        csv_file.flush()
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    print("\nFinal counts:")
    for label_name, count in label_counts.items():
        print(f"  {label_name}: {count} frames")
    print(f"Skipped low-visibility/no-pose frames: {skipped_count}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
