#!/usr/bin/env python3
"""
CatBoost Optuna + 5-fold CV training pipeline (enhanced logging & outputs)

- Fold summary logging only (no epoch logs)
- Test report: includes test_time (hold-out test set)
- CV reports:
    - Confusion matrix per fold (Best Trial only)
    - Average confusion matrix over folds (row-normalized)
    - Loss curve per fold (Best Trial only)
    - Fold-wise validation reports (test_report_foldX.txt) + average (test_report_avg.txt)
- Train log: all Trials/Folds in one txt
- Final model:
    ✅ Best Trial의 Best Fold 모델을 재학습하여 .pkl로 저장
    ✅ 파일명 형식: {MODEL_NAME}_trial{t}_fold{n}_best.pkl
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
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

import csv
import pandas as pd

# -------------------------------
# SETTINGS
# -------------------------------
MODEL_NAME = "catBoost"
SEED = 42
N_TRIALS = 40
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
OPTUNA_DIRECTION = "maximize"

CSV_PATH = os.path.join(
    "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/Illustris/Illustris_preprocess_SFR_no.csv"
)

OUT_BASE_MODEL = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/model/classicalMachineLearning/SFR_inf_remove"
OUT_BASE_EVAL = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/evaluation/classicalMachineLearning/SFR_inf_remove"

model_dir = os.path.join(OUT_BASE_MODEL, MODEL_NAME)
eval_dir = os.path.join(OUT_BASE_EVAL, MODEL_NAME)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

train_log_path = os.path.join(eval_dir, f"{MODEL_NAME}_train_log.txt")

# (Hold-out test set report & images)
test_report_path = os.path.join(eval_dir, f"{MODEL_NAME}_test_report.txt")
confusion_png_path = os.path.join(eval_dir, f"{MODEL_NAME}_confusion_matrix.png")   # test set
loss_curve_png_path = os.path.join(eval_dir, f"{MODEL_NAME}_loss_curve.png")        # best-fold curve (best trial)

# CV outputs (Best Trial only)
cv_avg_report_path = os.path.join(eval_dir, f"{MODEL_NAME}_test_report_avg.txt")    # average over folds
best_params_path = os.path.join(model_dir, f"{MODEL_NAME}_best_params.json")
meta_json_path = os.path.join(model_dir, f"{MODEL_NAME}_meta.json")

# 최종 모델 경로는 Best Trial/Best Fold를 알고 난 뒤 결정
final_ckpt_path = None  # placeholder

# label names
LABEL_MAP = {0: "non", 1: "pre", 2: "post"}
labels_str_order = ["non", "pre", "post"]
labels_int_order = [0, 1, 2]

class_names = ["non", "pre", "post"]

# -------------------------------
# LOAD DATA
# -------------------------------
print(f"Loading CSV: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    engine="python",
    sep=None,
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)

# 컬럼 이름 정리
df.columns = [c.strip().strip("\ufeff") for c in df.columns]

# Phase 컬럼 찾기
phase_candidates = [c for c in df.columns if c.strip().lower() == "phase"]
if not phase_candidates:
    raise ValueError(f"'Phase' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)[:20]}")
label_col = phase_candidates[0]

# 다양한 표현을 0/1/2로 매핑
PHASE_MAP_ANY = {
    "-1": 1,  -1: 1, "pre": 1,  "PRE": 1,  "Pre": 1,
     "0": 0,   0: 0, "non": 0,  "NON": 0,  "Non": 0,
     "1": 2,   1: 2, "post": 2, "POST": 2, "Post": 2
}

lab = df[label_col].astype(str).str.strip()
y_all = lab.map(PHASE_MAP_ANY)
if y_all.isna().any():
    bad_vals = sorted(df.loc[y_all.isna(), label_col].unique().tolist())
    raise ValueError(f"Phase에 예상치 못한 값 존재: {bad_vals}")
y_all = y_all.astype(int).to_numpy()

# feature 컬럼 선택
noise_cols = {"SubHaloID", "Snapshot"}
present_noise = [c for c in noise_cols if c in df.columns]
exclude = {label_col, "ID"} | set(present_noise)
feat_cols = [c for c in df.columns if c not in exclude]

# 수치 변환
for c in feat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

X_all = df[feat_cols]
nan_rows = X_all.isna().any(axis=1)
if nan_rows.any():
    print(f"[warn] removing {nan_rows.sum()} rows with NaN")
    X_all = X_all.loc[~nan_rows].copy()
    y_all = y_all[~nan_rows.values]

X_all = X_all.values.astype(np.float32)

print(f"[info] n_samples={X_all.shape[0]}, n_features={X_all.shape[1]}")
print(f"[info] features: {feat_cols[:8]}{'...' if len(feat_cols) > 8 else ''}")

# 80/20 Hold-out split
X_tr80, X_te20, y_tr80, y_te20 = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=SEED,
    stratify=y_all
)

# Stratified K-fold 설정
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Train log 파일
log_file = open(train_log_path, "w", buffering=1)

# 각 Trial/Fold 결과 저장용 (Best Trial 출력용)
# fold_history[trial_no] = {
#   "folds": [
#      {
#        "fold": int,
#        "best_iter": int,
#        "train_acc": float,
#        "train_f1": float,
#        "val_acc": float,
#        "val_f1": float,
#        "train_loss": [...],
#        "val_loss": [...],
#        "y_true": np.ndarray,
#        "y_pred": np.ndarray
#      }, ...
#   ],
#   "avg_val_f1": float
# }
fold_history = {}

study_start_time = time.time()
study_start_iso = datetime.now().isoformat()

# -------------------------------
# Optuna objective
# -------------------------------
def objective(trial):
    # 하이퍼파라미터 탐색 범위
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    }

    trial_idx = trial.number
    trial_no = trial_idx + 1

    fold_best_f1 = []
    fold_records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr80, y_tr80)):
        fold_no = fold + 1

        X_tr, X_val = X_tr80[train_idx], X_tr80[val_idx]
        y_tr, y_val = y_tr80[train_idx], y_tr80[val_idx]

        clf = CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=SEED,
            verbose=False
        )

        fold_start = time.time()
        clf.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )
        fold_elapsed = time.time() - fold_start

        best_iter = clf.get_best_iteration()

        y_tr_pred = clf.predict(X_tr)
        y_val_pred = clf.predict(X_val)

        tr_acc = accuracy_score(y_tr, y_tr_pred)
        tr_f1 = f1_score(y_tr, y_tr_pred, average="macro")
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="macro")

        fold_best_f1.append(val_f1)

        log_file.write(
            f"[Trial {trial_no:02d}] Fold {fold_no:02d} summary: "
            f"best_iter={best_iter} train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} fold_time={fold_elapsed:.2f}s\n"
        )

        # Loss history
        evals_result = clf.get_evals_result()
        train_loss_list = evals_result["learn"]["MultiClass"]
        val_loss_list = evals_result["validation"]["MultiClass"]

        fold_records.append(
            {
                "fold": fold_no,
                "best_iter": int(best_iter),
                "train_acc": float(tr_acc),
                "train_f1": float(tr_f1),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "train_loss": train_loss_list.copy(),
                "val_loss": val_loss_list.copy(),
                "y_true": y_val.copy(),
                "y_pred": y_val_pred.copy(),
            }
        )

    avg_val_f1 = float(np.mean(fold_best_f1))

    # Trial별 결과 저장
    fold_history[trial_no] = {
        "folds": fold_records,
        "avg_val_f1": avg_val_f1,
    }

    return avg_val_f1


# -------------------------------
# Run Optuna
# -------------------------------
study = optuna.create_study(direction=OPTUNA_DIRECTION)
study.optimize(objective, n_trials=N_TRIALS)

# Best params 저장
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)

# -------------------------------
# Best Trial & Best Fold 정보 정리
# -------------------------------
best_trial_no = study.best_trial.number + 1
best_trial_info = fold_history[best_trial_no]
fold_records = best_trial_info["folds"]

# Best Trial 내에서 val_f1이 가장 좋은 Fold 선택
best_fold_record = max(fold_records, key=lambda fr: fr["val_f1"])
best_fold_idx = best_fold_record["fold"]
best_iteration = best_fold_record["best_iter"]

# CV 평균 지표 (Best Trial 기준)
avg_val_acc = float(np.mean([fr["val_acc"] for fr in fold_records]))
avg_val_macro_f1 = float(np.mean([fr["val_f1"] for fr in fold_records]))
avg_val_loss = float(
    np.mean([float(fr["val_loss"][-1]) for fr in fold_records])
)

# -------------------------------
# CV Outputs (Best Trial 기준)
#   - Fold별 Confusion Matrix
#   - Fold별 Loss Curve
#   - Fold별 Validation Report
#   - 평균 Confusion Matrix
#   - 평균 Report
# -------------------------------
cm_sum = np.zeros((len(labels_int_order), len(labels_int_order)), dtype=np.float32)

for fr in fold_records:
    fold_no = fr["fold"]
    y_true_fold = fr["y_true"]
    y_pred_fold = fr["y_pred"]

    # Confusion matrix (정수 라벨 기준)
    cm_fold = confusion_matrix(
        y_true_fold,
        y_pred_fold,
        labels=labels_int_order
    )
    cm_sum += cm_fold.astype(np.float32)

    # Fold별 Confusion Matrix 이미지 저장
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_fold,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_str_order,
        yticklabels=labels_str_order,
    )
    plt.title(f"{MODEL_NAME} Confusion Matrix (Trial {best_trial_no}, Fold {fold_no})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fold_cm_path = os.path.join(
        eval_dir,
        f"{MODEL_NAME}_confusion_matrix_fold{fold_no}.png"
    )
    plt.tight_layout()
    plt.savefig(fold_cm_path)
    plt.close()

    # Fold별 Loss Curve (Train/Val)
    train_loss = fr["train_loss"]
    val_loss = fr["val_loss"]

    plt.figure(figsize=(7, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Logloss")
    plt.title(
        f"{MODEL_NAME} Loss Curve (Trial {best_trial_no}, Fold {fold_no})"
    )
    plt.legend()
    plt.grid(True)
    fold_loss_path = os.path.join(
        eval_dir,
        f"{MODEL_NAME}_loss_curve_fold{fold_no}.png"
    )
    plt.tight_layout()
    plt.savefig(fold_loss_path)
    plt.close()

    # Fold별 Validation Report 저장
    y_true_str = np.vectorize(LABEL_MAP.get)(y_true_fold)
    y_pred_str = np.vectorize(LABEL_MAP.get)(y_pred_fold)
    acc_fold = fr["val_acc"]
    f1_fold = fr["val_f1"]
    report_fold = classification_report(
        y_true_str,
        y_pred_str,
        labels=labels_str_order,
        digits=4
    )

    fold_report_path = os.path.join(
        eval_dir,
        f"{MODEL_NAME}_test_report_fold{fold_no}.txt"
    )
    with open(fold_report_path, "w") as f:
        f.write(
            f"[Fold {fold_no}] Validation report (Best trial {best_trial_no})\n"
        )
        f.write(f"Val_Accuracy : {acc_fold:.4f}\n")
        f.write(f"Val_Macro-F1 : {f1_fold:.4f}\n\n")
        f.write(report_fold)

# 평균 Confusion Matrix (row-normalized)
row_sums = cm_sum.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0  # 0 division 방지
cm_avg_norm = cm_sum / row_sums

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_avg_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels_str_order,
    yticklabels=labels_str_order,
)
plt.title(
    f"{MODEL_NAME} Confusion Matrix (Best Trial {best_trial_no} Avg over {N_SPLITS} folds)"
)
plt.xlabel("Predicted")
plt.ylabel("True")
cm_avg_path = os.path.join(
    eval_dir,
    f"{MODEL_NAME}_confusion_matrix_avg.png"
)
plt.tight_layout()
plt.savefig(cm_avg_path)
plt.close()

# 평균 Validation Report 저장
with open(cv_avg_report_path, "w") as f:
    f.write(
        f"[Best trial {best_trial_no}] Cross-validation average over {N_SPLITS} folds\n"
    )
    f.write(f"Avg Val_Accuracy : {avg_val_acc:.4f}\n")
    f.write(f"Avg Val_Macro-F1 : {avg_val_macro_f1:.4f}\n")
    f.write(f"Avg Val_Loss     : {avg_val_loss:.4f}\n")

# Best Fold 기준 Loss Curve (하나짜리 대표본: MODEL_NAME_loss_curve.png)
best_train_loss = best_fold_record["train_loss"]
best_val_loss = best_fold_record["val_loss"]

plt.figure(figsize=(7, 5))
plt.plot(best_train_loss, label="Train Loss")
plt.plot(best_val_loss, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Logloss")
plt.title(
    f"{MODEL_NAME} Best Fold Loss Curve "
    f"(Best trial {best_trial_no}, Fold {best_fold_idx})"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_curve_png_path)
plt.close()

# -------------------------------
# Final Model = Best Trial의 Best Fold 모델 재학습
# -------------------------------
# Best Fold에 해당하는 train/val index를 다시 구함 (동일 skf 설정 사용)
best_train_idx = None
best_val_idx = None
for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr80, y_tr80)):
    if (fold + 1) == best_fold_idx:
        best_train_idx = train_idx
        best_val_idx = val_idx
        break

if best_train_idx is None or best_val_idx is None:
    raise RuntimeError("Best fold index를 기반으로 train/val 인덱스를 찾지 못했습니다.")

X_tr_final = X_tr80[best_train_idx]
y_tr_final = y_tr80[best_train_idx]
X_val_final = X_tr80[best_val_idx]
y_val_final = y_tr80[best_val_idx]

# Best params로 모델 재생성
best_model = CatBoostClassifier(
    **study.best_params,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=SEED,
    verbose=False
)

best_model.fit(
    X_tr_final,
    y_tr_final,
    eval_set=(X_val_final, y_val_final),
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose=False
)

# 최종 모델 저장 (C 옵션: trial + fold 모두 포함)
final_ckpt_path = os.path.join(
    model_dir,
    f"{MODEL_NAME}_trial{best_trial_no}_fold{best_fold_idx}_best.pkl"
)
joblib.dump(best_model, final_ckpt_path)

# -------------------------------
# Test Evaluation (Hold-out 20%)
# -------------------------------
start_test = time.time()
y_pred_enc = best_model.predict(X_te20)
test_time = time.time() - start_test

y_test_mapped = np.vectorize(LABEL_MAP.get)(y_te20)
y_pred_mapped = np.vectorize(LABEL_MAP.get)(y_pred_enc)

cm_test = confusion_matrix(
    y_test_mapped,
    y_pred_mapped,
    labels=labels_str_order
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_str_order,
    yticklabels=labels_str_order,
)
plt.title(f"{MODEL_NAME} Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(confusion_png_path)
plt.close()

acc = accuracy_score(y_test_mapped, y_pred_mapped)
macro_f1 = f1_score(y_test_mapped, y_pred_mapped, average="macro")
report = classification_report(
    y_test_mapped,
    y_pred_mapped,
    labels=labels_str_order,
    digits=4
)

with open(test_report_path, "w") as f:
    f.write(f"Test time (s): {test_time:.4f}\n")
    f.write(f"Accuracy     : {acc:.4f}\n")
    f.write(f"Macro-F1     : {macro_f1:.4f}\n\n")
    f.write("[Test] Classification Report\n")
    f.write(report)

# -------------------------------
# Summary & meta.json
# -------------------------------
log_file.write("\n=== SUMMARY ===\n")
log_file.write(f"Optuna study start: {study_start_iso}\n")
log_file.write(f"Optuna study end  : {datetime.now().isoformat()}\n")
log_file.write(
    f"Best trial (avg val_macro_f1): {best_trial_no} | value={study.best_value:.4f}\n"
)
log_file.write(
    f"Best fold in best trial: {best_fold_idx} | "
    f"val_macro_f1={best_fold_record['val_f1']:.4f}\n"
)
log_file.write(f"Best iteration (fold): {best_iteration}\n")
log_file.write(
    f"CV avg val_acc: {avg_val_acc:.4f}, "
    f"avg val_macro_f1: {avg_val_macro_f1:.4f}, "
    f"avg val_loss: {avg_val_loss:.4f}\n"
)
log_file.write(f"Used model ckpt: {final_ckpt_path}\n")
log_file.close()

# meta.json 구성
meta = {
    "start_time": study_start_iso,
    "end_time": datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "seed": SEED,
    "n_trials": N_TRIALS,
    "n_splits": N_SPLITS,
    "best_trial": int(best_trial_no),
    "best_fold": int(best_fold_idx),
    "best_iteration": int(best_iteration),
    "avg_val_macro_f1": round(avg_val_macro_f1, 4),
    "avg_val_accuracy": round(avg_val_acc, 4),
    "avg_val_loss": round(avg_val_loss, 4),
    "class_names": labels_str_order,
    "num_features": len(feat_cols),
    "feature_cols": feat_cols,
    "dataset": os.path.basename(CSV_PATH),
    "early_stopping": {
        "patience": EARLY_STOPPING_ROUNDS,
        "note": "CatBoost uses early_stopping_rounds on eval_set",
    },
    "best_params": study.best_params,
    "cv_folds": [
        {
            "fold": int(fr["fold"]),
            "val_accuracy": round(float(fr["val_acc"]), 4),
            "val_macro_f1": round(float(fr["val_f1"]), 4),
            "val_loss_last": round(float(fr["val_loss"][-1]), 4),
            "best_iteration": int(fr["best_iter"]),
        }
        for fr in fold_records
    ],
    "paths": {
        "final_ckpt": final_ckpt_path,
        "train_log": train_log_path,
        "test_report": test_report_path,          # hold-out test set
        "cv_report_avg": cv_avg_report_path,      # CV 평균 리포트
        "confusion_matrix_png_test": confusion_png_path,
        "confusion_matrix_png_cv_avg": cm_avg_path,
        "loss_curve_png_best_fold": loss_curve_png_path,
        "eval_dir": eval_dir,
        "model_dir": model_dir,
    },
    "test_accuracy": round(float(acc), 4),
    "test_macro_f1": round(float(macro_f1), 4),
    "test_time": round(float(test_time), 4),
    "note": "Final model = best fold of best trial (trained on 80% train split, evaluated on 20% hold-out test)",
}

with open(meta_json_path, "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Done.")
print(f"Best trial: {best_trial_no}, best fold: {best_fold_idx}, best_iter: {best_iteration}")
print(f"Final model (best trial/best fold): {final_ckpt_path}")
print(f"Test report: {test_report_path}")
print(f"Train log: {train_log_path}")
print(f"CV avg report: {cv_avg_report_path}")
print(f"Meta: {meta_json_path}")