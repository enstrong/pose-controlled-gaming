import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Webcam setup ─────────────────────────────────────────────
# change this to 1, 2, etc. if you have multiple cameras and want to use a different one
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Check System Preferences → Privacy & Security → Camera")
    exit()

import time
time.sleep(2)

for _ in range(10):
    cap.read()

print("Pipeline running. Press Q to quit.")
print("Watching shoulder positions...")
print("-" * 50)

# ── Landmarks or points on ur body ───────────────────────────
# MediaPipe got 33 landmarks
# full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26

# ── Main loop ────────────────────────────────────────────────
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    # ── Draw landmarks on the frame ──────────────────────────
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        landmarks = results.pose_landmarks.landmark

        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        lk = landmarks[LEFT_KNEE]
        rk = landmarks[RIGHT_KNEE]
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]

        shoulder_width = abs(rs.x - ls.x)
        body_center_x = (ls.x + rs.x) / 2.0

        # frame is 1.0 wide. are you left (< 0.33), center (0.33-0.67), or right (> 0.67)?
        if body_center_x < 0.33:
            lane = "LEFT"
            lane_color = (255, 100, 100)   # blue in BGR
        elif body_center_x > 0.67:
            lane = "RIGHT"
            lane_color = (100, 100, 255)   # red in BGR
        else:
            lane = "CENTER"
            lane_color = (100, 255, 100)   # green in BGR

        # shoulder height. lower .y value means higher in the image
        shoulder_y = (ls.y + rs.y) / 2.0
        hip_y = (lh.y + rh.y) / 2.0
        torso_height = hip_y - shoulder_y 

        # knee alternation for running detection later
        knee_diff = abs(lk.y - rk.y)

        # ── Print to terminal every 15 frames (2x per second at 30fps) ──────────────────────────────────────────
        if frame_count % 15 == 0:
            print(
                f"Lane: {lane:6s} | "
                f"Body center X: {body_center_x:.3f} | "
                f"Shoulder Y: {shoulder_y:.3f} | "
                f"Torso height: {torso_height:.3f} | "
                f"Knee diff: {knee_diff:.3f}"
            )

        # ── Draw overlay text on the video frame ─────────────
        h, w = frame.shape[:2]

        # Lane indicator
        cv2.putText(
            frame, f"LANE: {lane}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            lane_color,
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame, f"Center X: {body_center_x:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, f"Shoulder Y: {shoulder_y:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, f"Knee diff: {knee_diff:.3f}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        # draw a vertical line showing where lane boundaries are
        cv2.line(frame, (w // 3, 0), (w // 3, h), (200, 200, 200), 1)
        cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (200, 200, 200), 1)

    else:
        # are you shy? hide from the camera? or just not in frame?
        cv2.putText(
            frame, "No person detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
        )

    cv2.imshow("Subway Surfers CV — Pipeline Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nQuitting.")
        break

# ── cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pose.close()
print("Pipeline closed cleanly.")