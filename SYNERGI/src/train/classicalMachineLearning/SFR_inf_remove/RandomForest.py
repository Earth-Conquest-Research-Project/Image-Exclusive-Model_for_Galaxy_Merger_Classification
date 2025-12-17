# ============================================================
# RandomForest (Phase=3-Class)
# Server Output Format Fully Matched Version
# ============================================================

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# Settings
# ============================================================
SEED = 42
np.random.seed(SEED)

MODEL_NAME = "RandomForest"

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "Illustris")
TRAIN_CSV  = os.path.join(DATA_DIR, "Illustris_preprocess_SFR_no.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "classicalMachineLearning","SFR_inf_remove", MODEL_NAME)
EVAL_DIR  = os.path.join(PROJECT_ROOT, "evaluation", "classicalMachineLearning", "SFR_inf_remove1", MODEL_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ============================================================
# Class / Phase Mapping
# ============================================================
CLASS_NAMES = ["Non", "Pre", "Post"]
CLASS_ID_ORDER = [0, 1, 2]

PHASE_MAP = {
    "-1": 1,  -1: 1,       # Pre
     "0": 0,   0: 0,       # Non
     "1": 2,   1: 2,       # Post
}

def remap_phase(series):
    s = series.astype(str).str.strip().map(PHASE_MAP)
    if s.isna().any():
        bad = series[s.isna()].unique().tolist()
        raise ValueError(f"Unexpected Phase values: {bad}")
    return s.astype(int).to_numpy()

# ============================================================
# Load CSV
# ============================================================
print(f"[INFO] Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, engine="python")
df.columns = [c.strip().strip("\ufeff") for c in df.columns]

if "Phase" not in df.columns:
    raise ValueError("Phase column not found!")

y_all = remap_phase(df["Phase"])

exclude = {"Phase","SubHaloID", "Snapshot", "ID",
           "__index_level_0__", "Unnamed: 0", "unnamed: 0"}

feat_cols = [c for c in df.columns if c not in exclude]

df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

nan_rows = df[feat_cols].isna().any(axis=1)
if nan_rows.any():
    print(f"[WARN] Removing NaN rows: {nan_rows.sum()}")
    df = df.loc[~nan_rows].reset_index(drop=True)
    y_all = y_all[~nan_rows.values]

X_all = df[feat_cols].to_numpy(dtype=np.float32)
print(f"[INFO] Samples={X_all.shape[0]}, Features={len(feat_cols)}")

# ============================================================
# 80/20 Split (For depth/trees tuning evaluation)
# ============================================================
X_tr80, X_te20, y_tr80, y_te20 = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED
)

print(f"[INFO] Train80={X_tr80.shape}, Test20={X_te20.shape}")

# ============================================================
# Hyperparameter Search (5-Fold CV on 80% set)
# ============================================================
kfold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

max_depth_candidates = [None, 10, 14, 18, 22, 26, 30]
start_trees = 50
step_trees = 50
max_rounds = 20
patience = 5

hist_tr_f1, hist_va_f1 = [], []
cv_records = []
best_va_overall = -np.inf
best_cfg = None

print("\n[INFO] Hyperparameter Search Start...")

epoch = 0
for depth in max_depth_candidates:
    n_estimators = start_trees
    no_improve = 0
    best_va_depth = -np.inf

    while True:
        epoch += 1
        tr_scores, va_scores = [], []

        t0 = time.perf_counter()
        for tr_idx, va_idx in kfold_cv.split(X_tr80, y_tr80):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=SEED,
                n_jobs=-1
            )
            clf.fit(X_tr80[tr_idx], y_tr80[tr_idx])

            tr_pred = clf.predict(X_tr80[tr_idx])
            va_pred = clf.predict(X_tr80[va_idx])

            tr_scores.append(f1_score(y_tr80[tr_idx], tr_pred, average="macro"))
            va_scores.append(f1_score(y_tr80[va_idx], va_pred, average="macro"))

        mtr = float(np.mean(tr_scores))
        mva = float(np.mean(va_scores))
        elapsed = time.perf_counter() - t0

        hist_tr_f1.append(mtr)
        hist_va_f1.append(mva)

        cv_records.append({
            "epoch": epoch,
            "depth": depth,
            "n_estimators": n_estimators,
            "train_f1": mtr,
            "valid_f1": mva,
            "time_sec": elapsed
        })

        print(f"[CV {epoch:03d}] depth={depth}, trees={n_estimators} | TrainF1={mtr:.4f} | ValidF1={mva:.4f}")

        if mva > best_va_depth + 1e-6:
            best_va_depth = mva
            no_improve = 0
        else:
            no_improve += 1

        if mva > best_va_overall + 1e-6:
            best_va_overall = mva
            best_cfg = {"max_depth": depth, "n_estimators": n_estimators}

        if no_improve >= patience:
            break
        if n_estimators >= start_trees + step_trees * (max_rounds - 1):
            break

        n_estimators += step_trees

pd.DataFrame(cv_records).to_csv(os.path.join(EVAL_DIR, f"{MODEL_NAME}_cv5_results.csv"), index=False)
print(f"\n[CV Completed] Best Config = {best_cfg} | Best ValidF1={best_va_overall:.4f}")

# ============================================================
# Loss Curve (Hyperparameter Search)
# ============================================================
LOSS_PNG = os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve.png")

plt.figure(figsize=(7,4))
plt.plot([1-f for f in hist_tr_f1], label="Train Loss")
plt.plot([1-f for f in hist_va_f1], label="Valid Loss")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PNG)
plt.close()

# ============================================================
# 5-Fold Full Evaluation (Using best_cfg)
# ============================================================
print("\n[INFO] 5-FOLD Final Evaluation...")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_acc, fold_f1, fold_cm_list = [], [], []
best_fold = None
best_fold_f1 = -np.inf

for fold_id, (train_idx, test_idx) in enumerate(kfold.split(X_all, y_all), start=1):
    print(f"\n========== Fold {fold_id} ==========")

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    clf = RandomForestClassifier(
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        random_state=SEED,
        n_jobs=-1
    )

    # Train log
    TRAIN_LOG_FOLD = os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{fold_id}.txt")
    with open(TRAIN_LOG_FOLD, "w") as f:
        f.write(f"[Fold {fold_id}] Training RandomForest (best_cfg={best_cfg})\n")

    t_train0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t_train0

    with open(TRAIN_LOG_FOLD, "a") as f:
        f.write(f"train_time={t_train:.4f}s\n")

    # Save fold model
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{MODEL_NAME}_fold{fold_id}.pkl"))

    # TEST
    t_test0 = time.perf_counter()
    pred = clf.predict(X_te)
    t_test = time.perf_counter() - t_test0

    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    fold_acc.append(acc)
    fold_f1.append(f1)

    # Test Report
    REPORT_FOLD = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_fold{fold_id}.txt")
    report = classification_report(
        y_te, pred,
        labels=CLASS_ID_ORDER,
        target_names=CLASS_NAMES,
        digits=4
    )

    with open(REPORT_FOLD, "w") as f:
        f.write(f"Test time (s): {t_test:.4f}\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro-F1     : {f1:.4f}\n\n")
        f.write("[Test] Classification Report\n")
        f.write(report)

    # Confusion Matrix (Fold)
    cm = confusion_matrix(y_te, pred, labels=CLASS_ID_ORDER)
    fold_cm_list.append(cm)

    CM_FOLD = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_fold{fold_id}.png")
    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(CM_FOLD)
    plt.close()

    # Loss Curve (Fold) — RandomForest doesn't have epochs, so use 1-F1
    LOSS_FOLD = os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{fold_id}.png")
    plt.figure(figsize=(4,4))
    plt.plot([1-f1], marker="o", label="loss(1-F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_FOLD)
    plt.close()

    if f1 > best_fold_f1:
        best_fold_f1 = f1
        best_fold = fold_id

# ============================================================
# Average Confusion Matrix
# ============================================================
cm_total = np.sum(fold_cm_list, axis=0)
cm_avg = cm_total / cm_total.sum(axis=1, keepdims=True)

CM_AVG = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_avg.png")
fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_avg, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(CM_AVG)
plt.close()

# ============================================================
# Average Test Report
# ============================================================
ACC_AVG = float(np.mean(fold_acc))
F1_AVG = float(np.mean(fold_f1))

REPORT_AVG = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_avg.txt")
with open(REPORT_AVG, "w") as f:
    f.write(f"Accuracy_avg : {ACC_AVG:.4f}\n")
    f.write(f"MacroF1_avg  : {F1_AVG:.4f}\n")
    f.write(f"best_fold    : {best_fold}\n")

# ============================================================
# (NEW) Single-model confusion matrix (80/20 test)
# ============================================================
print("\n[INFO] Single-model (80/20) Confusion Matrix...")

clf_full = RandomForestClassifier(
    n_estimators=best_cfg["n_estimators"],
    max_depth=best_cfg["max_depth"],
    random_state=SEED,
    n_jobs=-1
)
clf_full.fit(X_tr80, y_tr80)

FULL_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best_fullTrain.pkl")
joblib.dump(clf_full, FULL_MODEL_PATH)

pred_full = clf_full.predict(X_te20)
cm_full = confusion_matrix(y_te20, pred_full, labels=CLASS_ID_ORDER)

CM_SINGLE = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix.png")
fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(CM_SINGLE)
plt.close()

# ============================================================
# Meta JSON
# ============================================================
meta = {
    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),

    "best_cfg": best_cfg,
    "best_fold": best_fold,
    "avg_accuracy": ACC_AVG,
    "avg_macro_f1": F1_AVG,

    "seed": SEED,
    "class_names": CLASS_NAMES,
    "num_features": len(feat_cols),
    "feature_cols": feat_cols,

    "paths": {
        "cv_results": os.path.join(EVAL_DIR, f"{MODEL_NAME}_cv5_results.csv"),
        "loss_curve_overall": LOSS_PNG,
        "confusion_matrix_avg": CM_AVG,
        "confusion_matrix_single": CM_SINGLE,
        "full_model": FULL_MODEL_PATH,
        "test_report_avg": REPORT_AVG
    }
}

META_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_meta.json")
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("\n========== ALL DONE ==========")
print(f"Best Config : {best_cfg}")
print(f"Avg Macro-F1: {F1_AVG:.4f}")