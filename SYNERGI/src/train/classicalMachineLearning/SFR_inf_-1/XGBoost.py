#!/usr/bin/env python3
"""
XGBoost Optuna + 5-fold CV training pipeline (full output version)
- Final model = best fold across best trial
- Full per-fold saving (CM/Loss/Report)
- CV averaged CM & Report
"""

import os
import time
import json
import optuna
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import seaborn as sns
import joblib
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# SETTINGS
# =====================================================================
MODEL_NAME = "xgBoost"
SEED = 42
N_TRIALS = 40
N_SPLITS = 5
EARLY_STOPPING = 20

CSV_PATH  = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/Illustris/Illustris_preprocess_SFR_-1.csv"
BASE_DIR = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI"
SAVE_DIR_MODEL = os.path.join(BASE_DIR, "model/classicalMachineLearning/SFR_inf_-1", MODEL_NAME)
SAVE_DIR_EVAL  = os.path.join(BASE_DIR, "evaluation/classicalMachineLearning/SFR_inf_-1", MODEL_NAME)
os.makedirs(SAVE_DIR_MODEL, exist_ok=True)
os.makedirs(SAVE_DIR_EVAL, exist_ok=True)

train_log_path      = os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_train_log.txt")
test_report_path    = os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_test_report.txt")
cv_avg_report_path  = os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_test_report_avg.txt")
meta_json_path      = os.path.join(SAVE_DIR_MODEL, f"{MODEL_NAME}_meta.json")
best_params_path    = os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_best_params.json")

log_file = open(train_log_path, "w", buffering=1)
class_names = ["non", "pre", "post"]
labels_int_order = [0, 1, 2]

# =====================================================================
# LOAD DATA
# =====================================================================
print(f"Loading CSV: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    engine="python",
    sep=None,
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)
df.columns = [c.strip().strip("\ufeff") for c in df.columns]

phase_candidates = [c for c in df.columns if c.strip().lower() == "phase"]
if not phase_candidates:
    raise ValueError("'Phase' column not found")
label_col = phase_candidates[0]

PHASE_MAP_ANY = {
    "-1": 1, -1: 1, "pre": 1, "PRE": 1, "Pre": 1,
     "0": 0,  0: 0, "non": 0, "NON": 0, "Non": 0,
     "1": 2,  1: 2, "post": 2, "POST": 2, "Post": 2
}
LABEL_MAP = {0: "non", 1: "pre", 2: "post"}

lab = df[label_col].astype(str).str.strip()
y_all = lab.map(PHASE_MAP_ANY)
if y_all.isna().any():
    print("Unexpected phase values:", sorted(df.loc[y_all.isna(), label_col].unique()))
    raise ValueError("Invalid phase category detected")
y_all = y_all.astype(int).to_numpy()

noise_cols = {"SubHaloID", "Snapshot", "phase_5"}
exclude = {label_col, "ID"} | {c for c in noise_cols if c in df.columns}
feat_cols = [c for c in df.columns if c not in exclude]

df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
X_all = df[feat_cols].values.astype(np.float32)
nan_mask = ~np.isnan(X_all).any(axis=1)
if nan_mask.sum() != X_all.shape[0]:
    print(f"[warn] removing {(~nan_mask).sum()} rows with NaN")
X_all = X_all[nan_mask]
y_all = y_all[nan_mask]

X_tr80, X_te20, y_tr80, y_te20 = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# =====================================================================
# FOLD HISTORY (trial → folds)
# =====================================================================
FOLD_HISTORY = {}  # {trial_no: {"folds": [...], "avg_val_f1": float}}
GLOBAL_START = datetime.now().isoformat()

# =====================================================================
# OPTUNA OBJECTIVE
# =====================================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": SEED,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "use_label_encoder": False
    }
    trial_no = trial.number + 1
    fold_records = []
    fold_f1_list = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr80, y_tr80)):
        fold_no = fold + 1
        X_tr, X_val = X_tr80[tr_idx], X_tr80[val_idx]
        y_tr, y_val = y_tr80[tr_idx], y_tr80[val_idx]

        clf = xgb.XGBClassifier(**params)
        start = time.time()
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING,
            verbose=False
        )
        fold_time = time.time() - start
        best_iter = clf.best_iteration

        # prediction
        y_tr_pred = clf.predict(X_tr)
        y_val_pred = clf.predict(X_val)

        tr_acc = accuracy_score(y_tr, y_tr_pred)
        tr_f1  = f1_score(y_tr, y_tr_pred, average="macro")
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1  = f1_score(y_val, y_val_pred, average="macro")
        fold_f1_list.append(val_f1)

        log_file.write(
            f"[Trial {trial_no:02d}] Fold {fold_no:02d} "
            f"| best_iter={best_iter} tr_f1={tr_f1:.4f} val_f1={val_f1:.4f} "
            f"val_acc={val_acc:.4f} time={fold_time:.2f}s\n"
        )

        evals_res = clf.evals_result()
        train_loss = evals_res["validation_0"]["mlogloss"]
        val_loss   = evals_res["validation_1"]["mlogloss"]

        fold_records.append({
            "fold": fold_no,
            "train_acc": float(tr_acc),
            "train_f1": float(tr_f1),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "train_loss": list(map(float, train_loss)),
            "val_loss": list(map(float, val_loss)),
            "y_true": y_val.copy(),
            "y_pred": y_val_pred.copy(),
            "best_iteration": int(best_iter),
            "elapsed": float(fold_time)
        })

    avg_val_f1 = float(np.mean(fold_f1_list))
    FOLD_HISTORY[trial_no] = {"folds": fold_records, "avg_val_f1": avg_val_f1}
    return avg_val_f1

# =====================================================================
# RUN OPTUNA
# =====================================================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

# =====================================================================
# BEST TRIAL / BEST FOLD
# =====================================================================
best_trial_no = study.best_trial.number + 1
best_trial_info = FOLD_HISTORY[best_trial_no]
fold_records = best_trial_info["folds"]
best_fold_record = max(fold_records, key=lambda fr: fr["val_f1"])
best_fold_idx = best_fold_record["fold"]
best_iter = best_fold_record["best_iteration"]

avg_val_f1 = float(np.mean([fr["val_f1"] for fr in fold_records]))
avg_val_acc = float(np.mean([fr["val_acc"] for fr in fold_records]))
avg_val_loss = float(np.mean([fr["val_loss"][-1] for fr in fold_records if fr["val_loss"]]))

with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)

# ============================================================
# SAVE BEST PARAMETER (Optuna) TO MODEL DIRECTORY
# ============================================================
best_param_save_path = os.path.join(SAVE_DIR_MODEL, f"{MODEL_NAME}_best_param.json")
with open(best_param_save_path, "w") as fw:
    json.dump(study.best_params, fw, indent=2)

print(f"[Saved] best params → {best_param_save_path}")


# =====================================================================
# SAVE PER-FOLD OUTPUT
# =====================================================================
cm_sum = np.zeros((len(labels_int_order), len(labels_int_order)), dtype=np.float32)

for fr in fold_records:
    fold_no = fr["fold"]
    y_true = fr["y_true"]
    y_pred = fr["y_pred"]

    cm_fold = confusion_matrix(y_true, y_pred, labels=labels_int_order)
    cm_sum += cm_fold

    # fold confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_fold, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{MODEL_NAME} CM (T{best_trial_no} F{fold_no})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_confusion_matrix_fold{fold_no}.png"))
    plt.close()

    # fold loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(fr["train_loss"], label="Train Loss")
    plt.plot(fr["val_loss"], label="Validation Loss")
    plt.legend(); plt.grid(True)
    plt.title(f"{MODEL_NAME} Loss (T{best_trial_no} F{fold_no})")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_loss_curve_fold{fold_no}.png"))
    plt.close()

    # fold validation report
    report_fold = classification_report(
        y_true, y_pred, labels=labels_int_order, target_names=class_names, digits=4
    )
    with open(os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_test_report_fold{fold_no}.txt"), "w") as f:
        f.write(f"[Trial {best_trial_no} Fold {fold_no}] Validation Report\n")
        f.write(f"Val_Acc : {fr['val_acc']:.4f}\n")
        f.write(f"Val_F1  : {fr['val_f1']:.4f}\n\n")
        f.write(report_fold)

# average confusion matrix (normalized)
row_sums = cm_sum.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
cm_avg = cm_sum / row_sums

plt.figure(figsize=(6,5))
sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"{MODEL_NAME} CV Confusion Avg (T{best_trial_no})")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_confusion_matrix_avg.png"))
plt.close()

# average validation report
with open(cv_avg_report_path, "w") as f:
    f.write(f"[Best Trial {best_trial_no}] {N_SPLITS}-Fold CV Average\n")
    f.write(f"Avg Val Accuracy : {avg_val_acc:.4f}\n")
    f.write(f"Avg Val Macro-F1 : {avg_val_f1:.4f}\n")
    f.write(f"Avg Val Loss     : {avg_val_loss:.4f}\n")

# best-fold representative loss curve
plt.figure(figsize=(7,5))
plt.plot(best_fold_record["train_loss"], label="Train Loss")
plt.plot(best_fold_record["val_loss"], label="Validation Loss")
plt.legend(); plt.grid(True)
plt.title(f"{MODEL_NAME} Best Fold Loss (T{best_trial_no} F{best_fold_idx})")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_loss_curve.png"))
plt.close()

# =====================================================================
# FINAL MODEL = best trial / best fold (훈련 재진행 O)
# =====================================================================
final_ckpt_path = os.path.join(
    SAVE_DIR_MODEL,
    f"{MODEL_NAME}_trial{best_trial_no}_fold{best_fold_idx}_best.pkl"
)

# FINAL MODEL TRAINING (best trial / best fold / best iteration)
best_model = xgb.XGBClassifier(
    **study.best_params,
    objective="multi:softprob",
    random_state=SEED,
    n_jobs=-1,
    eval_metric="mlogloss",
    use_label_encoder=False
)

best_tr_idx, best_val_idx = list(skf.split(X_tr80, y_tr80))[best_fold_idx - 1]
X_train_best, y_train_best = X_tr80[best_tr_idx], y_tr80[best_tr_idx]

# best_iter 만큼 트리 개수 설정
best_model.set_params(n_estimators=best_iter)

best_model.fit(
    X_train_best,
    y_train_best,
    verbose=False
)

joblib.dump(best_model, final_ckpt_path)

# =====================================================================
# EVALUATE ON TEST SET
# =====================================================================
y_pred_test = best_model.predict(X_te20)
acc = accuracy_score(y_te20, y_pred_test)
macro_f1 = f1_score(y_te20, y_pred_test, average="macro")
report = classification_report(
    y_te20, y_pred_test, labels=labels_int_order, target_names=class_names, digits=4
)

cm_test = confusion_matrix(
    y_te20, y_pred_test, labels=labels_int_order
)
plt.figure(figsize=(6,5))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"{MODEL_NAME} Test Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_EVAL, f"{MODEL_NAME}_confusion_matrix.png"))
plt.close()

with open(test_report_path, "w") as f:
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Macro-F1 : {macro_f1:.4f}\n\n")
    f.write(report)

# =====================================================================
# META.JSON SAVE
# =====================================================================
meta = {
    "best_trial": best_trial_no,
    "best_fold": best_fold_idx,
    "best_iteration": int(best_iter),
    "avg_val_accuracy": round(avg_val_acc,4),
    "avg_val_macro_f1": round(avg_val_f1,4),
    "avg_val_loss": round(avg_val_loss,4),
    "dataset": os.path.basename(CSV_PATH),
    "class_names": class_names,
    "feature_cols": feat_cols,
    "trial_folds": [
        {
            "fold": fr["fold"],
            "val_acc": round(fr["val_acc"],4),
            "val_macro_f1": round(fr["val_f1"],4),
            "best_iteration": fr["best_iteration"]
        }
        for fr in fold_records
    ],
    "test": {
        "accuracy": round(acc,4),
        "macro_f1": round(macro_f1,4)
    },
    "paths": {
        "final_model": final_ckpt_path,
        "train_log": train_log_path,
        "test_report": test_report_path,
        "cv_report_avg": cv_avg_report_path,
        "eval_dir": SAVE_DIR_EVAL,
        "model_dir": SAVE_DIR_MODEL
    },
    "note": "final model = best fold of best trial (no full-train)"
}
with open(meta_json_path, "w") as f:
    json.dump(meta, f, indent=2)

log_file.close()
print("\n\n================ DONE ================")
print(f"Final model: {final_ckpt_path}")
print(f"Meta saved:  {meta_json_path}")
print(f"Eval dir:    {SAVE_DIR_EVAL}")