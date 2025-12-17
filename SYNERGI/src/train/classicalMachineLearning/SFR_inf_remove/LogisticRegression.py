# -*- coding: utf-8 -*-
"""
Stable Logistic Regression (SGDClassifier/partial_fit)
Dataset: final_12_datasetPhase_complete.csv
5-Fold CV on 80% split
CPU-only

Outputs per fold and average:
- LogisticRegression_confusion_matrix_foldX.png / _avg.png
- LogisticRegression_loss_curve_foldX.png
- LogisticRegression_test_report_foldX.txt / _avg.txt
- LogisticRegression_train_log_foldX.txt
- LogisticRegression_meta.json
"""

import os, time, json, random, csv
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, log_loss
)
from sklearn.linear_model import SGDClassifier
import joblib


# ============================================================
# Config
# ============================================================
CSV_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/Illustris/Illustris_preprocess_SFR_no.csv" 

MODEL_NAME = "LogisticRegression"
MODEL_DIR = f"/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/model/classicalMachineLearning/SFR_inf_remove/{MODEL_NAME}"
EVAL_DIR  = f"/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/evaluation/classicalMachineLearning/SFR_inf_remove/{MODEL_NAME}"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

CLASS_NAMES = ["non", "pre", "post"]
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Utility
# ============================================================
def make_minibatches(X, y, batch_size, seed):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sl = idx[start:start + batch_size]
        yield X[sl], y[sl]

def average_reports(reports):
    if not reports: return {}
    avg_report = {}
    keys = list(reports[0].keys()) # classes + accuracy, macro avg, weighted avg
    
    # Initialize
    for k in keys:
        if k == 'accuracy':
            avg_report[k] = 0.0
        else:
            avg_report[k] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0}
            
    # Sum
    n = len(reports)
    for r in reports:
        for k in keys:
            if k == 'accuracy':
                avg_report[k] += r[k]
            else:
                for m in ['precision', 'recall', 'f1-score', 'support']:
                    avg_report[k][m] += r[k][m]
    
    # Average
    for k in keys:
        if k == 'accuracy':
            avg_report[k] /= n
        else:
            for m in ['precision', 'recall', 'f1-score', 'support']:
                avg_report[k][m] /= n
    return avg_report

def report_dict_to_text(d, classes):
    # Formats dictionary back to text similar to sklearn classification_report
    out = f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    for c in classes:
        r = d[c]
        out += f"{c:>15} {r['precision']:10.4f} {r['recall']:10.4f} {r['f1-score']:10.4f} {int(r['support']):10}\n"
    out += "\n"
    out += f"{'accuracy':>15} {'':>10} {'':>10} {d['accuracy']:10.4f} {int(d['macro avg']['support']):10}\n"
    
    m = d['macro avg']
    out += f"{'macro avg':>15} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1-score']:10.4f} {int(m['support']):10}\n"
    w = d['weighted avg']
    out += f"{'weighted avg':>15} {w['precision']:10.4f} {w['recall']:10.4f} {w['f1-score']:10.4f} {int(w['support']):10}\n"
    return out


# ============================================================
# Load Dataset
# ============================================================
if not CSV_PATH:
    print("[Warning] CSV_PATH is empty. Please set the path.")
    # For robust execution, ensure CSV_PATH is set before running.

df = pd.read_csv(CSV_PATH) if CSV_PATH else pd.DataFrame() # Avoid crash if empty

feature_cols = [
    "StellarMass", "AbsMag_g", "AbsMag_r", "AbsMag_i", "AbsMag_z",
    "color_gr", "color_gi", "SFR", "BulgeMass",
    "EffectiveRadius", "VelocityDispersion", "Metallicity"
]

label_col = "Phase"

PHASE_MAP = {-1: 1, 0: 0, 1: 2}   # pre=1, non=0, post=2

if not df.empty:
    df = df.dropna(subset=feature_cols + [label_col]).copy()
    df[label_col] = df[label_col].map(PHASE_MAP)
    
    X_all = df[feature_cols].astype(np.float32).to_numpy()
    y_all = df[label_col].astype(int).to_numpy()
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"[info] n_samples={X_all.shape[0]}, n_features={X_all.shape[1]}")

    # ============================================================
    # 80/20 Split
    # ============================================================
    X_tr80, X_test, y_tr80, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
    )
    print(f"[split] train80={X_tr80.shape}, test20={X_test.shape}")
else:
    print("[Error] DataFrame is empty. Check CSV path.")
    exit()


# ============================================================
# 5-Fold CV
# ============================================================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

EPOCHS = 100
BATCH_SIZE = 256
ES_PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-3

fold_results = []
fold_reports = []
total_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=float)
meta_val_losses = []
meta_best_epochs = []

start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_tr80, y_tr80), 1):
    print(f"\n===== Fold {fold}/5 =====")
    
    # Paths per fold
    ckpt_path = os.path.join(MODEL_DIR, f"fold{fold}_best.joblib")
    fold_log_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{fold}.txt")
    log_file = open(fold_log_path, "w", encoding="utf-8")
    
    X_train, X_val = X_tr80[tr_idx], X_tr80[val_idx]
    y_train, y_val = y_tr80[tr_idx], y_tr80[val_idx]

    # SGD Logistic Regression
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=WEIGHT_DECAY,
        learning_rate="adaptive",
        eta0=LR,
        max_iter=1,
        warm_start=True,
        tol=None,
        random_state=SEED
    )

    classes = np.array([0, 1, 2])
    best_val_loss_fold = None
    best_epoch_fold = 0
    wait = 0
    train_losses, val_losses = [], []
    fold_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        # ---- Train with partial_fit
        first_batch = True
        for Xb, yb in make_minibatches(X_train, y_train, BATCH_SIZE, seed=SEED + epoch):
            if first_batch:
                clf.partial_fit(Xb, yb, classes=classes)
                first_batch = False
            else:
                clf.partial_fit(Xb, yb)

        # ---- Evaluate
        def safe_predict_proba(X):
            try:
                p = clf.predict_proba(X)
                return np.nan_to_num(p, nan=1e-9)
            except:
                return np.full((len(X), 3), 1/3)

        tr_proba = safe_predict_proba(X_train)
        val_proba = safe_predict_proba(X_val)

        tr_loss = log_loss(y_train, tr_proba, labels=classes)
        val_loss = log_loss(y_val, val_proba, labels=classes)

        tr_pred = tr_proba.argmax(1)
        val_pred = val_proba.argmax(1)

        tr_acc = accuracy_score(y_train, tr_pred)
        val_acc = accuracy_score(y_val, val_pred)
        tr_f1 = f1_score(y_train, tr_pred, average="macro")
        val_f1 = f1_score(y_val, val_pred, average="macro")

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        line = (f"[Epoch {epoch:03d}] "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                f"epoch_time={epoch_time:.2f}s")
        print(f"[Fold{fold}]{line}")
        log_file.write(line + "\n")

        # ---- Early Stopping
        if best_val_loss_fold is None or val_loss < best_val_loss_fold:
            best_val_loss_fold = val_loss
            best_epoch_fold = epoch
            wait = 0
            joblib.dump(clf, ckpt_path)
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                stop_msg = f"Early stopping at epoch {epoch} (best epoch={best_epoch_fold})"
                print(f"→ {stop_msg}")
                log_file.write(stop_msg + "\n")
                break
                
    total_train_time = time.time() - fold_start_time
    log_file.write(f"Total training time: {total_train_time:.2f}s\n")
    log_file.close()
    
    meta_val_losses.append(best_val_loss_fold)
    meta_best_epochs.append(best_epoch_fold)

    # ========================================================
    # Load best model and evaluate on test 20%
    # ========================================================
    clf = joblib.load(ckpt_path)

    start_t = time.time()
    proba = clf.predict_proba(X_test)
    proba = np.nan_to_num(proba, nan=1e-9)
    preds = proba.argmax(1)
    test_time = time.time() - start_t

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    
    fold_results.append({"fold": fold, "acc": acc, "f1": macro_f1})

    # 1. Confusion Matrix (Fold)
    cm = confusion_matrix(y_test, preds)
    total_cm += cm
    
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_fold{fold}.png"), dpi=180)
    plt.close()

    # 2. Loss Curve (Fold)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve Fold {fold}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{fold}.png"), dpi=180)
    plt.close()

    # 3. Test Report (Fold)
    report_dict = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
    fold_reports.append(report_dict)
    
    report_text = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_fold{fold}.txt"), "w") as f:
        f.write(f"Test time (s): {test_time:.4f}\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro-F1     : {macro_f1:.4f}\n\n")
        f.write("[Test] Classification Report\n")
        f.write(report_text)


# ============================================================
# Average Outputs
# ============================================================

# 1. Average Confusion Matrix (Normalized)
# plot() 함수 대신, 데이터 자체를 먼저 정규화합니다.
row_sums = total_cm.sum(axis=1)[:, np.newaxis]
# 0으로 나누는 것을 방지하며 나눗셈 수행
normalized_cm = np.divide(total_cm, row_sums, out=np.zeros_like(total_cm), where=row_sums != 0)

fig, ax = plt.subplots(figsize=(5, 4))

# 정규화된 행렬(normalized_cm)을 넣어줍니다.
disp_avg = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=CLASS_NAMES)

# 여기서 normalize='true' 옵션을 제거합니다.
disp_avg.plot(ax=ax, cmap="Blues", colorbar=False, values_format='.2f')

plt.title(f"Confusion Matrix Average")
plt.tight_layout()
cm_avg_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_avg.png")
plt.savefig(cm_avg_path, dpi=180)
plt.close()

# 2. Average Test Report
avg_report_dict = average_reports(fold_reports)

avg_report_text = report_dict_to_text(avg_report_dict, CLASS_NAMES)
avg_acc = avg_report_dict['accuracy']
avg_f1 = avg_report_dict['macro avg']['f1-score']

avg_report_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_avg.txt")
with open(avg_report_path, "w") as f:
    f.write(f"Accuracy     : {avg_acc:.4f}\n")
    f.write(f"Macro-F1     : {avg_f1:.4f}\n\n")
    f.write("[Average Test] Classification Report\n")
    f.write(avg_report_text)


# ============================================================
# META JSON
# ============================================================
best_fold_idx = np.argmax([r['f1'] for r in fold_results])
best_fold_num = fold_results[best_fold_idx]['fold']
best_fold_epoch = meta_best_epochs[best_fold_idx]

meta = {
    "start_time": start_time_str,
    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    "best_fold": int(best_fold_num),
    "best_epoch": int(best_fold_epoch),
    "avg_macro_f1": float(avg_f1),
    "avg_accuracy": float(avg_acc),
    "avg_val_loss": float(np.mean(meta_val_losses)),

    "epochs_run": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "device": "cpu",

    "class_names": CLASS_NAMES,

    "num_features": len(feature_cols),
    "feature_cols": feature_cols,
    "dataset": CSV_PATH,

    "preprocessing": {
        "standard_scaler": True,
        "knn": True,
        "note": "already applied in dataset"
    },

    "optimizer": "SGDClassifier",
    "lr_scheduler": "adaptive lr",

    "early_stopping": {
        "patience": ES_PATIENCE,
        "min_delta": 0.0
    },

    "paths": {
        "best_ckpt": os.path.abspath(os.path.join(MODEL_DIR, f"fold{best_fold_num}_best.joblib")),
        "train_log": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{best_fold_num}.txt")),
        "test_report": os.path.abspath(avg_report_path),
        "confusion_matrix_png": os.path.abspath(cm_avg_path),
        "loss_curve_png": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{best_fold_num}.png"))
    },

    "fold_results": fold_results
}

meta_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)


print("\n===== TRAINING COMPLETE =====")
print(f"Mean Accuracy : {avg_acc:.4f}")
print(f"Mean Macro-F1 : {avg_f1:.4f}")
print(f"Saved to: {EVAL_DIR}")