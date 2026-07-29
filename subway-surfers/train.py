import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/gesture_data.csv'
MODEL_PATH = 'models/gesture_classifier.pkl'
os.makedirs('models', exist_ok=True)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

if 'label' not in df.columns:
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = [
        "label", "timestamp",
        "nose_x","nose_y","nose_z","nose_vis",
        "left_eye_inner_x","left_eye_inner_y","left_eye_inner_z","left_eye_inner_vis",
        "left_eye_x","left_eye_y","left_eye_z","left_eye_vis",
        "left_eye_outer_x","left_eye_outer_y","left_eye_outer_z","left_eye_outer_vis",
        "right_eye_inner_x","right_eye_inner_y","right_eye_inner_z","right_eye_inner_vis",
        "right_eye_x","right_eye_y","right_eye_z","right_eye_vis",
        "right_eye_outer_x","right_eye_outer_y","right_eye_outer_z","right_eye_outer_vis",
        "left_ear_x","left_ear_y","left_ear_z","left_ear_vis",
        "right_ear_x","right_ear_y","right_ear_z","right_ear_vis",
        "mouth_left_x","mouth_left_y","mouth_left_z","mouth_left_vis",
        "mouth_right_x","mouth_right_y","mouth_right_z","mouth_right_vis",
        "left_shoulder_x","left_shoulder_y","left_shoulder_z","left_shoulder_vis",
        "right_shoulder_x","right_shoulder_y","right_shoulder_z","right_shoulder_vis",
        "left_elbow_x","left_elbow_y","left_elbow_z","left_elbow_vis",
        "right_elbow_x","right_elbow_y","right_elbow_z","right_elbow_vis",
        "left_wrist_x","left_wrist_y","left_wrist_z","left_wrist_vis",
        "right_wrist_x","right_wrist_y","right_wrist_z","right_wrist_vis",
        "left_pinky_x","left_pinky_y","left_pinky_z","left_pinky_vis",
        "right_pinky_x","right_pinky_y","right_pinky_z","right_pinky_vis",
        "left_index_x","left_index_y","left_index_z","left_index_vis",
        "right_index_x","right_index_y","right_index_z","right_index_vis",
        "left_thumb_x","left_thumb_y","left_thumb_z","left_thumb_vis",
        "right_thumb_x","right_thumb_y","right_thumb_z","right_thumb_vis",
        "left_hip_x","left_hip_y","left_hip_z","left_hip_vis",
        "right_hip_x","right_hip_y","right_hip_z","right_hip_vis",
        "left_knee_x","left_knee_y","left_knee_z","left_knee_vis",
        "right_knee_x","right_knee_y","right_knee_z","right_knee_vis",
        "left_ankle_x","left_ankle_y","left_ankle_z","left_ankle_vis",
        "right_ankle_x","right_ankle_y","right_ankle_z","right_ankle_vis",
        "left_heel_x","left_heel_y","left_heel_z","left_heel_vis",
        "right_heel_x","right_heel_y","right_heel_z","right_heel_vis",
        "left_foot_index_x","left_foot_index_y","left_foot_index_z","left_foot_index_vis",
        "right_foot_index_x","right_foot_index_y","right_foot_index_z","right_foot_index_vis",
    ]

print(f"Total samples: {len(df)}")
df = df[~df["label"].isin(["LEFT", "RIGHT", "STOP"])].copy()
print(f"After removing LEFT/RIGHT: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())

def get_col(df, name):
    return pd.to_numeric(df[f"{name}_x"], errors="coerce").to_numpy(dtype=np.float32), \
           pd.to_numeric(df[f"{name}_y"], errors="coerce").to_numpy(dtype=np.float32)

# Core landmarks
ls_x, ls_y = get_col(df, "left_shoulder")
rs_x, rs_y = get_col(df, "right_shoulder")
lh_x, lh_y = get_col(df, "left_hip")
rh_x, rh_y = get_col(df, "right_hip")
lk_x, lk_y = get_col(df, "left_knee")
rk_x, rk_y = get_col(df, "right_knee")
la_x, la_y = get_col(df, "left_ankle")
ra_x, ra_y = get_col(df, "right_ankle")
lhe_x, lhe_y = get_col(df, "left_heel")
rhe_x, rhe_y = get_col(df, "right_heel")
lf_x, lf_y = get_col(df, "left_foot_index")
rf_x, rf_y = get_col(df, "right_foot_index")
le_x, le_y = get_col(df, "left_elbow")
re_x, re_y = get_col(df, "right_elbow")
lw_x, lw_y = get_col(df, "left_wrist")
rw_x, rw_y = get_col(df, "right_wrist")

print("\nEngineering features...")

body_center_x      = (ls_x + rs_x) / 2.0
shoulder_y_avg     = (ls_y + rs_y) / 2.0
hip_y_avg          = (lh_y + rh_y) / 2.0
torso_length       = hip_y_avg - shoulder_y_avg
safe_torso         = np.where(np.abs(torso_length) > 1e-6, torso_length, 1e-6)
shoulder_y_norm    = shoulder_y_avg / safe_torso
knee_diff          = np.abs(lk_y - rk_y)
knee_y_avg         = (lk_y + rk_y) / 2.0
knee_y_norm        = knee_y_avg / safe_torso
hip_shoulder_ratio = torso_length / (np.abs(hip_y_avg - knee_y_avg) + 1e-6)
lean_angle         = ls_y - rs_y
shoulder_width     = np.abs(rs_x - ls_x)

WINDOW = 10
knee_diff_var = np.zeros(len(knee_diff))
for i in range(len(knee_diff)):
    start = max(0, i - WINDOW + 1)
    knee_diff_var[i] = np.var(knee_diff[start:i+1])

foot_y_avg         = (lf_y + rf_y) / 2.0
foot_y_norm        = foot_y_avg / safe_torso 
ankle_y_avg        = (la_y + ra_y) / 2.0
heel_y_avg         = (lhe_y + rhe_y) / 2.0


foot_to_hip_ratio  = (hip_y_avg - foot_y_avg) / safe_torso
foot_to_knee_dist  = np.abs(foot_y_avg - knee_y_avg) / safe_torso

wrist_y_avg        = (lw_y + rw_y) / 2.0
wrist_y_norm       = wrist_y_avg / safe_torso
elbow_y_avg        = (le_y + re_y) / 2.0

wrist_above_shoulder = shoulder_y_avg - wrist_y_avg

body_height_norm   = (foot_y_avg - shoulder_y_avg) / safe_torso

feature_names = [
    # Position
    "body_center_x",
    "lean_angle",
    "shoulder_width",
    # Shoulder/torso
    "shoulder_y_norm",
    "shoulder_y_avg",
    "torso_length",
    "hip_shoulder_ratio",
    # Knee
    "knee_y_avg",
    "knee_y_norm",
    "knee_diff",
    "knee_diff_var",
    # Foot
    "foot_y_avg",
    "foot_y_norm",
    "ankle_y_avg",
    "heel_y_avg",
    "foot_to_hip_ratio",
    "foot_to_knee_dist",
    # Arm
    "wrist_y_avg",
    "wrist_y_norm",
    "elbow_y_avg",
    "wrist_above_shoulder",
    # Full body
    "body_height_norm",
]

X = np.column_stack([
    body_center_x,
    lean_angle,
    shoulder_width,
    shoulder_y_norm,
    shoulder_y_avg,
    torso_length,
    hip_shoulder_ratio,
    knee_y_avg,
    knee_y_norm,
    knee_diff,
    knee_diff_var,
    foot_y_avg,
    foot_y_norm,
    ankle_y_avg,
    heel_y_avg,
    foot_to_hip_ratio,
    foot_to_knee_dist,
    wrist_y_avg,
    wrist_y_norm,
    elbow_y_avg,
    wrist_above_shoulder,
    body_height_norm,
])

df = df[df["label"] != "label"].copy()
y = df["label"].to_numpy()

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_names}")

# ── train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ── train Random Forest ───────────────────────────────────────
print("\nTraining Random Forest...")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1 # cpu cores. -1 means use all cores
)

clf.fit(X_train, y_train)
print("Training complete.")

# ── Evaluate ──────────────────────────────────────────────────
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"\nTest accuracy: {accuracy:.1%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))


print("Confusion matrix:")
classes = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=classes)
print(f"{'':10s}", end='')
for c in classes:
    print(f"{c:10s}", end='')
print()
for i, true_class in enumerate(classes):
    print(f"{true_class:10s}", end='')
    for j in range(len(classes)):
        val = cm[i][j]
        marker = " ←MISS" if i != j and val > 0 else ""
        print(f"{val:<10d}", end='')
    print()

# ── feature importance ────────────────────────────────────────
print("\nFeature importances (which features the model relies on most):")
importances = clf.feature_importances_
for name, imp in sorted(zip(feature_names, importances),
                        key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 50)
    print(f"  {name:25s} {imp:.4f}  {bar}")


model_data = {
    'classifier': clf,
    'feature_names': feature_names,
    'classes': classes,
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to {MODEL_PATH}")
print("\nDone. Check the confusion matrix for which gestures get confused.")
print("If accuracy is below 90%, collect more data for the confused classes.")