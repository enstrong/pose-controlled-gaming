import pickle
import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "gesture_classifier.pkl"

ACTION_TO_KEY = {
    "JUMP":   Key.up,
    "CROUCH": Key.down,
}

LEFT_BOUNDARY  = 0.40
RIGHT_BOUNDARY = 0.60

# cooldowns for each action type
LANE_COOLDOWN   = 0.35
ACTION_COOLDOWN = 0.50
CONF_THRESHOLD  = 0.80

MIN_VISIBILITY = 0.45

LANDMARK_INDEX = {
    "left_shoulder":    11,
    "right_shoulder":   12,
    "left_hip":         23,
    "right_hip":        24,
    "left_knee":        25,
    "right_knee":       26,
    "left_ankle":       27,
    "right_ankle":      28,
    "left_heel":        29,
    "right_heel":       30,
    "left_foot_index":  31,
    "right_foot_index": 32,
    "left_elbow":       13,
    "right_elbow":      14,
    "left_wrist":       15,
    "right_wrist":      16,
}


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run: python src/train.py")
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["classifier"] if isinstance(data, dict) else data


def visible(landmarks):
    return all(
        landmarks[i].visibility >= MIN_VISIBILITY
        for i in LANDMARK_INDEX.values()
    )


def extract_features(landmarks, kd_win):
    def xy(name):
        lm = landmarks[LANDMARK_INDEX[name]]
        return lm.x, lm.y

    ls_x, ls_y   = xy("left_shoulder")
    rs_x, rs_y   = xy("right_shoulder")
    lh_x, lh_y   = xy("left_hip")
    rh_x, rh_y   = xy("right_hip")
    lk_x, lk_y   = xy("left_knee")
    rk_x, rk_y   = xy("right_knee")
    la_x, la_y   = xy("left_ankle")
    ra_x, ra_y   = xy("right_ankle")
    lhe_x, lhe_y = xy("left_heel")
    rhe_x, rhe_y = xy("right_heel")
    lf_x, lf_y   = xy("left_foot_index")
    rf_x, rf_y   = xy("right_foot_index")
    le_x, le_y   = xy("left_elbow")
    re_x, re_y   = xy("right_elbow")
    lw_x, lw_y   = xy("left_wrist")
    rw_x, rw_y   = xy("right_wrist")

    body_center_x  = (ls_x + rs_x) / 2.0
    shoulder_y_avg = (ls_y + rs_y) / 2.0
    hip_y_avg      = (lh_y + rh_y) / 2.0
    torso_length   = hip_y_avg - shoulder_y_avg
    safe_torso     = torso_length if abs(torso_length) > 1e-6 else 1e-6
    knee_diff      = abs(lk_y - rk_y)
    knee_y_avg     = (lk_y + rk_y) / 2.0
    foot_y_avg     = (lf_y + rf_y) / 2.0
    ankle_y_avg    = (la_y + ra_y) / 2.0
    heel_y_avg     = (lhe_y + rhe_y) / 2.0
    wrist_y_avg    = (lw_y + rw_y) / 2.0
    elbow_y_avg    = (le_y + re_y) / 2.0

    kd_win.append(knee_diff)
    knee_diff_var = float(np.var(kd_win))

    return np.array([[
        body_center_x,
        ls_y - rs_y,
        abs(rs_x - ls_x),
        shoulder_y_avg / safe_torso,
        shoulder_y_avg,
        torso_length,
        torso_length / (abs(hip_y_avg - knee_y_avg) + 1e-6),
        knee_y_avg,
        knee_y_avg / safe_torso,
        knee_diff,
        knee_diff_var,
        foot_y_avg,
        foot_y_avg / safe_torso,
        ankle_y_avg,
        heel_y_avg,
        (hip_y_avg - foot_y_avg) / safe_torso,
        abs(foot_y_avg - knee_y_avg) / safe_torso,
        wrist_y_avg,
        wrist_y_avg / safe_torso,
        elbow_y_avg,
        shoulder_y_avg - wrist_y_avg,
        (foot_y_avg - shoulder_y_avg) / safe_torso,
    ]], dtype=np.float32)


def tap(keyboard, key):
    keyboard.press(key)
    keyboard.release(key)


def main():
    clf      = load_model(MODEL_PATH)
    keyboard = Controller()
    kd_win   = deque(maxlen=10)

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    pose       = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam not found")
        pose.close()
        return

    time.sleep(2)
    for _ in range(10):
        cap.read()

    print("Running. Focus Subway Surfers, then move.")
    print("Left side of frame = LEFT | Right side = RIGHT")
    print("JUMP / CROUCH from classifier (conf >= 0.80)")
    print("Press Q to quit.\n")

    last_lane_time   = 0.0
    last_action_time = 0.0
    prev_lane        = "CENTER"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            now     = time.monotonic()

            action    = "NO_POSE"
            body_x    = None
            lane      = "CENTER"
            conf      = 0.0
            conf_text = ""

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
                lms = results.pose_landmarks.landmark

                if visible(lms):
                    feats  = extract_features(lms, kd_win)
                    action = str(clf.predict(feats)[0]).upper()
                    body_x = (lms[11].x + lms[12].x) / 2.0

                    if hasattr(clf, "predict_proba"):
                        conf = float(np.max(clf.predict_proba(feats)))
                    conf_text = f"conf {conf:.2f}"

                    if body_x < LEFT_BOUNDARY:
                        lane = "LEFT"
                    elif body_x > RIGHT_BOUNDARY:
                        lane = "RIGHT"
                    else:
                        lane = "CENTER"

                    lane_cooldown_ok = (now - last_lane_time) >= LANE_COOLDOWN
                    if lane != prev_lane and lane_cooldown_ok:
                        if lane == "LEFT":
                            tap(keyboard, Key.left)
                            print(f"LANE → LEFT  (x={body_x:.2f})")
                        elif lane == "RIGHT":
                            tap(keyboard, Key.right)
                            print(f"LANE → RIGHT  (x={body_x:.2f})")
                        elif lane == "CENTER":
                            if prev_lane == "LEFT":
                                tap(keyboard, Key.right)
                                print("LANE → CENTER (was LEFT, pressing RIGHT)")
                            elif prev_lane == "RIGHT":
                                tap(keyboard, Key.left)
                                print("LANE → CENTER (was RIGHT, pressing LEFT)")
                        last_lane_time = now
                    prev_lane = lane

                    action_cooldown_ok = (now - last_action_time) >= ACTION_COOLDOWN
                    if action in ACTION_TO_KEY and conf >= CONF_THRESHOLD and action_cooldown_ok:
                        tap(keyboard, ACTION_TO_KEY[action])
                        last_action_time = now
                        print(f"{action}  conf={conf:.2f}")

                else:
                    action = "LOW_VIS"
                    kd_win.clear()

            else:
                kd_win.clear()

            # ── Overlay ─────────────────────────────────────────────
            h, w = frame.shape[:2]
            cv2.line(frame, (int(w * LEFT_BOUNDARY), 0),
                            (int(w * LEFT_BOUNDARY), h), (180, 180, 180), 1)
            cv2.line(frame, (int(w * RIGHT_BOUNDARY), 0),
                            (int(w * RIGHT_BOUNDARY), h), (180, 180, 180), 1)

            color = (80, 220, 80) if action in (*ACTION_TO_KEY, "RUN") \
                                  else (80, 80, 255)
            cv2.putText(frame, f"{action}  {conf_text}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color, 4, cv2.LINE_AA)

            lane_color = (100, 200, 255) if lane != "CENTER" else (180, 180, 180)
            cv2.putText(frame, f"lane: {lane}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, lane_color, 3, cv2.LINE_AA)

            if body_x is not None:
                cv2.putText(frame, f"x={body_x:.2f}",
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("pose-controlled-gaming", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Closed cleanly.")


if __name__ == "__main__":
    main()