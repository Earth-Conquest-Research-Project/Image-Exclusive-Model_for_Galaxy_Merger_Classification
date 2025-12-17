# ============================================================
# DecisionTree - Depth Tuning + Full 5-Fold Evaluation
# 서버 출력 규격 완전 준수 버전 (옵션 A)
# ============================================================

import os, time, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# Settings
# ============================================================
SEED = 42
np.random.seed(SEED)

MODEL_NAME = "DecisionTree"

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "Illustris")
TRAIN_CSV  = os.path.join(DATA_DIR, "Illustris_preprocess_SFR_no.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "classicalMachineLearning","SFR_inf_remove", MODEL_NAME)
EVAL_DIR  = os.path.join(PROJECT_ROOT, "evaluation", "classicalMachineLearning", "SFR_inf_remove", MODEL_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ============================================================
# Phase Mapping
# ============================================================
PHASE_MAP = {
    "-1": 1,  -1: 1,  # pre
     "0": 0,   0: 0,  # non
     "1": 2,   1: 2   # post
}

CLASS_ID_ORDER = [0, 1, 2]
CLASS_NAMES    = ["Non", "Pre", "Post"]

def remap_phase(series):
    s = series.astype(str).str.strip().map(PHASE_MAP)
    if s.isna().any():
        raise ValueError(f"Unexpected Phase values: {series[s.isna()].unique()}")
    return s.astype(int).to_numpy()

# ============================================================
# Load CSV
# ============================================================
print(f"[INFO] Loading CSV: {TRAIN_CSV}")
df = pd.read_csv(TRAIN_CSV)
df.columns = [c.strip().strip("\ufeff") for c in df.columns]

if "Phase" not in df.columns:
    raise ValueError("Phase column missing!")

y_all = remap_phase(df["Phase"])

exclude = {"Phase", "phase_5", "ID", "SubHaloID", "Snapshot"}
feat_cols = [c for c in df.columns if c not in exclude]

# Convert feature columns to numeric
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

nan_rows = df[feat_cols].isna().any(axis=1)
if nan_rows.any():
    print(f"[WARN] Removing {nan_rows.sum()} rows with NaN.")
    df = df.loc[~nan_rows].reset_index(drop=True)
    y_all = y_all[~nan_rows.values]

X_all = df[feat_cols].to_numpy(dtype=np.float32)

print(f"[INFO] samples={len(X_all)}, features={len(feat_cols)}")


# ============================================================
# Split 80/20 (train for depth tuning)
# ============================================================
X_tr80, X_te20, y_tr80, y_te20 = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)
print(f"[INFO] train80={X_tr80.shape}, test20={X_te20.shape}")

# ============================================================
# Depth Candidates
# ============================================================
depth_candidates = list(range(2, 31)) + [None]   # None == unlimited depth
print(f"[INFO] depth candidates: {depth_candidates}")

# ============================================================
# 5-Fold CV (on 80% train set) to find best_depth
# ============================================================
kfold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_logs = []
best_depth = None
best_cv_f1 = -np.inf
no_improve = 0
patience = 5

train_losses, valid_losses = [], []

print("\n[INFO] Starting Depth Tuning...")

for epoch, depth in enumerate(depth_candidates, start=1):
    t_start = time.perf_counter()
    tr_f1_list, va_f1_list = [], []

    for tr_idx, va_idx in kfold_cv.split(X_tr80, y_tr80):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        clf.fit(X_tr80[tr_idx], y_tr80[tr_idx])

        pred_tr = clf.predict(X_tr80[tr_idx])
        pred_va = clf.predict(X_tr80[va_idx])

        tr_f1_list.append(f1_score(y_tr80[tr_idx], pred_tr, average="macro"))
        va_f1_list.append(f1_score(y_tr80[va_idx], pred_va, average="macro"))

    mtr = np.mean(tr_f1_list)
    mva = np.mean(va_f1_list)
    train_losses.append(1 - mtr)
    valid_losses.append(1 - mva)

    epoch_time = time.perf_counter() - t_start

    log_line = f"[DepthTune Epoch {epoch:02d}] depth={depth} | trainF1={mtr:.4f} | validF1={mva:.4f}"
    print(log_line)

    cv_logs.append({
        "epoch": epoch,
        "depth": depth,
        "train_macroF1": mtr,
        "valid_macroF1": mva,
        "time_sec": epoch_time
    })

    # Best depth selection
    if mva > best_cv_f1 + 1e-6:
        best_cv_f1 = mva
        best_depth = depth
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print("[INFO] Early stopping (depth tuning).")
        break

print(f"\n[INFO] Best Depth Selected = {best_depth}, validF1={best_cv_f1:.4f}")

# Save Depth Tuning Curve
DEPTH_LOSS_PNG = os.path.join(EVAL_DIR, f"{MODEL_NAME}_depth_tuning_loss.png")
plt.figure(figsize=(7,5))
plt.plot(train_losses, label="train_loss(1-F1)")
plt.plot(valid_losses, label="valid_loss(1-F1)")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(DEPTH_LOSS_PNG)
plt.close()

# Save CV results
pd.DataFrame(cv_logs).to_csv(os.path.join(EVAL_DIR, f"{MODEL_NAME}_depth_tuning_results.csv"), index=False)


# ============================================================
# 5-Fold Evaluation using best_depth
# ============================================================
print("\n[INFO] Starting Final 5-Fold Train/Test...")

kfold_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_acc_list = []
fold_f1_list = []
fold_conf_list = []

best_fold = None
best_fold_f1 = -np.inf

start_time_meta = time.strftime("%Y-%m-%dT%H:%M:%S")

for fold_id, (train_idx, test_idx) in enumerate(kfold_eval.split(X_all, y_all), start=1):
    print(f"\n========== Fold {fold_id} ==========")

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    # Train
    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)

    TRAIN_LOG_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{fold_id}.txt")
    with open(TRAIN_LOG_PATH, "w") as f:
        f.write(f"[Fold {fold_id}] Training DecisionTree (best_depth={best_depth})\n")

    t_train0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t_train0

    with open(TRAIN_LOG_PATH, "a") as f:
        f.write(f"Training time: {t_train:.4f}s\n")

    # Save fold model
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{MODEL_NAME}_fold{fold_id}.pkl"))

    # Test
    t_test0 = time.perf_counter()
    pred = clf.predict(X_te)
    t_test = time.perf_counter() - t_test0

    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")

    fold_acc_list.append(acc)
    fold_f1_list.append(f1)

    # Test Report
    REPORT_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_fold{fold_id}.txt")

    report = classification_report(
        y_te, pred,
        labels=CLASS_ID_ORDER,
        target_names=CLASS_NAMES,
        digits=4
    )

    with open(REPORT_PATH, "w") as f:
        f.write(f"Test time (s): {t_test:.4f}\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro-F1     : {f1:.4f}\n\n")
        f.write("[Test] Classification Report\n")
        f.write(report)

    # Confusion Matrix (Fold)
    cm = confusion_matrix(y_te, pred, labels=CLASS_ID_ORDER)
    fold_conf_list.append(cm)

    CM_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_fold{fold_id}.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=160)
    plt.close()

    # Loss Curve (Fold)
    LOSS_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{fold_id}.png")
    plt.figure(figsize=(6,4))
    plt.plot([1 - f1], marker="o", label="loss(1-F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PATH)
    plt.close()

    # Best Fold Tracking
    if f1 > best_fold_f1:
        best_fold_f1 = f1
        best_fold = fold_id


# ============================================================
# Confusion Matrix (Average)
# ============================================================
cm_total = np.sum(fold_conf_list, axis=0)
cm_avg = cm_total / cm_total.sum(axis=1, keepdims=True)

CM_AVG_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_avg.png")

fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_avg, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(CM_AVG_PATH, dpi=160)
plt.close()

# ============================================================
# Test Report (Average)
# ============================================================
ACC_AVG = float(np.mean(fold_acc_list))
F1_AVG  = float(np.mean(fold_f1_list))

REPORT_AVG_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_avg.txt")

with open(REPORT_AVG_PATH, "w") as f:
    f.write(f"Accuracy_avg : {ACC_AVG:.4f}\n")
    f.write(f"MacroF1_avg  : {F1_AVG:.4f}\n")
    f.write(f"best_fold    : {best_fold}\n")


# ============================================================
# Meta JSON
# ============================================================
meta = {
    "start_time": start_time_meta,
    "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "best_depth": best_depth,
    "best_fold": best_fold,
    "avg_accuracy": ACC_AVG,
    "avg_macro_f1": F1_AVG,
    "seed": SEED,
    "class_names": CLASS_NAMES,
    "num_features": len(feat_cols),
    "feature_cols": feat_cols,
    "paths": {
        "depth_tuning_loss_curve": DEPTH_LOSS_PNG,
        "confusion_matrix_avg": CM_AVG_PATH,
        "test_report_avg": REPORT_AVG_PATH
    }
}

meta_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("\n========== DONE ==========")
print(f"Best depth = {best_depth}, Best fold = {best_fold}")
print(f"Macro-F1 avg = {F1_AVG:.4f}, Accuracy avg = {ACC_AVG:.4f}")

# ============================================================
# (NEW) 전체 test 기반 confusion matrix 생성
# ============================================================

print("\n[INFO] Creating single-model test confusion matrix (80/20 split)...")

# best_depth 기반 모델 다시 학습 (전체 train80)
clf_full = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)

t_full_train0 = time.perf_counter()
clf_full.fit(X_tr80, y_tr80)
t_full_train = time.perf_counter() - t_full_train0

# Save the full model
FULL_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_bestDepth_fullTrain.pkl")
joblib.dump(clf_full, FULL_MODEL_PATH)

# Predict on 20% test set
t_full_test0 = time.perf_counter()
pred_full = clf_full.predict(X_te20)
t_full_test = time.perf_counter() - t_full_test0

# Confusion matrix
cm_full = confusion_matrix(y_te20, pred_full, labels=CLASS_ID_ORDER)

# Save CM
CM_FULL_PATH = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(CM_FULL_PATH, dpi=160)
plt.close()

print(f"[INFO] Saved single-model test confusion matrix → {CM_FULL_PATH}")