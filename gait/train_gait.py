import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH  = "data/gait_data.csv"
MODEL_PATH = "models/gait_classifier.pkl"
os.makedirs("models", exist_ok=True)

WINDOW_FRAMES = 30
STEP_FRAMES   = 10


def get_y(df, name):
    return pd.to_numeric(df[f"{name}_y"], errors="coerce").to_numpy(dtype=np.float32)

def get_x(df, name):
    return pd.to_numeric(df[f"{name}_x"], errors="coerce").to_numpy(dtype=np.float32)


def extract_window_features(window_df):
    """
    Extract temporal gait features from a window of consecutive frames.
    All features measure asymmetry or regularity over time.
    """
    ls_y  = get_y(window_df, "left_shoulder")
    rs_y  = get_y(window_df, "right_shoulder")
    lh_y  = get_y(window_df, "left_hip")
    rh_y  = get_y(window_df, "right_hip")
    lk_y  = get_y(window_df, "left_knee")
    rk_y  = get_y(window_df, "right_knee")
    la_y  = get_y(window_df, "left_ankle")
    ra_y  = get_y(window_df, "right_ankle")
    lf_y  = get_y(window_df, "left_foot_index")
    rf_y  = get_y(window_df, "right_foot_index")
    lh_x  = get_x(window_df, "left_hip")
    rh_x  = get_x(window_df, "right_hip")

    # ── Shoulder asymmetry ─────────────────────────────────────
    # one shoulder consistently lower = Trendelenburg / compensation
    shoulder_diff      = ls_y - rs_y
    shoulder_asym_mean = float(np.mean(np.abs(shoulder_diff)))
    shoulder_asym_std  = float(np.std(shoulder_diff))

    # ── Hip sway ───────────────────────────────────────────────
    # lateral hip movement during walking
    hip_center_x = (lh_x + rh_x) / 2.0
    hip_sway     = float(np.std(hip_center_x))

    # Hip vertical asymmetry (Trendelenburg drop)
    hip_diff_mean = float(np.mean(np.abs(lh_y - rh_y)))

    # ── Knee symmetry ──────────────────────────────────────────
    # During normal walking both knees reach similar peak heights
    left_knee_range  = float(np.max(lk_y) - np.min(lk_y))   # how much left knee moves
    right_knee_range = float(np.max(rk_y) - np.min(rk_y))   # how much right knee moves

    # Asymmetry ratio: 0 = perfectly symmetric, high = one knee barely moves (limp)
    knee_range_ratio = abs(left_knee_range - right_knee_range) / \
                       (max(left_knee_range, right_knee_range) + 1e-6)

    # Knee height difference variability
    knee_diff_series = np.abs(lk_y - rk_y)
    knee_diff_std    = float(np.std(knee_diff_series))
    knee_diff_mean   = float(np.mean(knee_diff_series))

    # ── Ankle / foot symmetry ──────────────────────────────────
    left_ankle_range  = float(np.max(la_y) - np.min(la_y))
    right_ankle_range = float(np.max(ra_y) - np.min(ra_y))
    ankle_range_ratio = abs(left_ankle_range - right_ankle_range) / \
                        (max(left_ankle_range, right_ankle_range) + 1e-6)

    left_foot_range   = float(np.max(lf_y) - np.min(lf_y))
    right_foot_range  = float(np.max(rf_y) - np.min(rf_y))
    foot_range_ratio  = abs(left_foot_range - right_foot_range) / \
                        (max(left_foot_range, right_foot_range) + 1e-6)

    body_center_y     = (ls_y + rs_y) / 2.0
    vertical_motion   = float(np.std(body_center_y))

    # ── Cross-correlation: do left and right knees alternate? ──
    if np.std(lk_y) > 1e-6 and np.std(rk_y) > 1e-6:
        corr = float(np.corrcoef(lk_y, rk_y)[0, 1])
    else:
        corr = 0.0
    knee_anticorr = -corr

    return [
        shoulder_asym_mean,
        shoulder_asym_std,
        hip_sway,
        hip_diff_mean,
        left_knee_range,
        right_knee_range,
        knee_range_ratio,
        knee_diff_std,
        knee_diff_mean,
        left_ankle_range,
        right_ankle_range,
        ankle_range_ratio,
        left_foot_range,
        right_foot_range,
        foot_range_ratio,
        vertical_motion,
        knee_anticorr,
    ]


FEATURE_NAMES = [
    "shoulder_asym_mean",
    "shoulder_asym_std",
    "hip_sway",
    "hip_diff_mean",
    "left_knee_range",
    "right_knee_range",
    "knee_range_ratio",
    "knee_diff_std",
    "knee_diff_mean",
    "left_ankle_range",
    "right_ankle_range",
    "ankle_range_ratio",
    "left_foot_range",
    "right_foot_range",
    "foot_range_ratio",
    "vertical_motion",
    "knee_anticorr",
]


def main():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total frames: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    X, y = [], []
    labels = df["label"].values

    for start in range(0, len(df) - WINDOW_FRAMES + 1, STEP_FRAMES):
        window = df.iloc[start : start + WINDOW_FRAMES]

        window_labels = labels[start : start + WINDOW_FRAMES]
        if len(set(window_labels)) != 1:
            continue

        features = extract_window_features(window)
        if any(np.isnan(f) for f in features):
            continue

        X.append(features)
        y.append(window_labels[0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"Windowed samples: {len(X)}")
    print(f"Window size: {WINDOW_FRAMES} frames, step: {STEP_FRAMES} frames")
    from collections import Counter
    print(f"Sample distribution: {Counter(y)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nTest accuracy: {(y_pred == y_test).mean():.1%}")
    print(classification_report(y_test, y_pred))

    classes = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Confusion matrix:")
    print(f"{'':20s}", end="")
    for c in classes: print(f"{c:22s}", end="")
    print()
    for i, tc in enumerate(classes):
        print(f"{tc:20s}", end="")
        for val in cm[i]: print(f"{val:<22d}", end="")
        print()

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 60)
        print(f"  {name:28s} {imp:.4f}  {bar}")

    model_data = {
        "classifier": clf,
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "window_frames": WINDOW_FRAMES,
        "step_frames": STEP_FRAMES,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()