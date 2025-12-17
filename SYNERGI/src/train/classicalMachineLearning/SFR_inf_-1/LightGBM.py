import os
import time
import json
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
import seaborn as sns
import joblib
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 경로 및 설정
# ============================================================
SEED = 42
N_SPLITS = 5
N_TRIALS = 40

model_name = "lightGBM"
CSV_PATH  = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/Illustris/Illustris_preprocess_SFR_-1.csv"
base_dir = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI"
save_dir_model = os.path.join(base_dir, "model/classicalMachineLearning/SFR_inf_-1", model_name)
save_dir_eval  = os.path.join(base_dir, "evaluation/classicalMachineLearning/SFR_inf_-1", model_name)
os.makedirs(save_dir_model, exist_ok=True)
os.makedirs(save_dir_eval, exist_ok=True)

train_log_path      = os.path.join(save_dir_eval,  f"{model_name}_train_log.txt")
test_report_path    = os.path.join(save_dir_eval,  f"{model_name}_test_report.txt")          # hold-out test
confusion_png_path  = os.path.join(save_dir_eval,  f"{model_name}_confusion_matrix.png")     # test set CM
loss_curve_png_path = os.path.join(save_dir_eval,  f"{model_name}_loss_curve.png")           # best fold 대표 curve
meta_json_path      = os.path.join(save_dir_model, f"{model_name}_meta.json")
best_params_path    = os.path.join(save_dir_eval,  f"{model_name}_best_params.json")
cv_avg_report_path  = os.path.join(save_dir_eval,  f"{model_name}_test_report_avg.txt")      # CV 평균 report

final_ckpt_path = None  # 나중에 채움

log_file = open(train_log_path, "w", buffering=1)
start_time_total = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

class_names = ["0-Non", "1-Pre", "2-Post"]
labels_int_order = [0, 1, 2]

# ============================================================
# 데이터 로드
# ============================================================
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
    raise ValueError(f"'Phase' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)[:20]}")
label_col = phase_candidates[0]

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

noise_cols = {"SubHaloID","Snapshot","phase_5"}
present_noise = [c for c in noise_cols if c in df.columns]
exclude = {label_col, "ID"} | set(present_noise)
feat_cols = [c for c in df.columns if c not in exclude]

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
print(f"[info] features: {feat_cols[:8]}{'...' if len(feat_cols)>8 else ''}")

X_tr80, X_te20, y_tr80, y_te20 = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ============================================================
# FOLD_HISTORY: Trial/Fold별 결과 저장
# ============================================================
# FOLD_HISTORY[trial_no] = {
#   "folds": [
#       {
#           "fold": int,
#           "train_acc": float,
#           "train_f1": float,
#           "val_acc": float,
#           "val_f1": float,
#           "train_loss": [...],
#           "val_loss": [...],
#           "y_true": np.ndarray,
#           "y_pred": np.ndarray,
#           "best_iteration": int
#       }, ...
#   ],
#   "avg_val_f1": float
# }
FOLD_HISTORY = {}
study_start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

# ============================================================
# Optuna Objective
# ============================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 31, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "objective": "multiclass",
        "random_state": SEED,
        "n_jobs": -1
    }

    fold_scores = []
    fold_records = []
    trial_no = trial.number + 1

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr80, y_tr80)):
        fold_no = fold + 1

        X_tr, X_val = X_tr80[train_idx], X_tr80[val_idx]
        y_tr, y_val = y_tr80[train_idx], y_tr80[val_idx]

        clf = LGBMClassifier(**params)

        start_fold = time.time()
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=10), log_evaluation(period=0)]
        )
        elapsed = time.time() - start_fold

        # 예측 & metric
        if getattr(clf, "best_iteration_", None) is not None:
            y_tr_pred  = clf.predict(X_tr, num_iteration=clf.best_iteration_)
            y_val_pred = clf.predict(X_val, num_iteration=clf.best_iteration_)
            best_iter = int(clf.best_iteration_)
        else:
            y_tr_pred  = clf.predict(X_tr)
            y_val_pred = clf.predict(X_val)
            best_iter = clf.n_estimators

        tr_acc = accuracy_score(y_tr, y_tr_pred)
        tr_f1  = f1_score(y_tr, y_tr_pred, average="macro")
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1  = f1_score(y_val, y_val_pred, average="macro")

        fold_scores.append(val_f1)

        log_file.write(
            f"[Trial {trial_no:02d}] Fold {fold_no:02d}: "
            f"best_iter={best_iter} train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={elapsed:.2f}s\n"
        )

        # loss curve (evals_result_ 사용)
        evals_res = getattr(clf, "evals_result_", None) or {}
        train_loss = []
        val_loss = []

        if "training" in evals_res and "multi_logloss" in evals_res["training"]:
            train_loss = evals_res["training"]["multi_logloss"]
        if "valid_1" in evals_res and "multi_logloss" in evals_res["valid_1"]:
            val_loss = evals_res["valid_1"]["multi_logloss"]
        else:
            for k in evals_res.keys():
                if k.startswith("valid") and "multi_logloss" in evals_res[k]:
                    val_loss = evals_res[k]["multi_logloss"]
                    break

        fold_records.append(
            {
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
            }
        )

    avg_val_f1 = float(np.mean(fold_scores))
    FOLD_HISTORY[trial_no] = {
        "folds": fold_records,
        "avg_val_f1": avg_val_f1
    }

    return avg_val_f1

# ============================================================
# Optuna 실행
# ============================================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=4)

# ============================================================
# Best Trial & Best Fold 계산
# ============================================================
best_trial_no = study.best_trial.number + 1
best_trial_info = FOLD_HISTORY[best_trial_no]
fold_records = best_trial_info["folds"]

best_fold_record = max(fold_records, key=lambda fr: fr["val_f1"])
best_fold_idx = best_fold_record["fold"]
best_iteration = best_fold_record["best_iteration"]

avg_val_acc = float(np.mean([fr["val_acc"] for fr in fold_records]))
avg_val_macro_f1 = float(np.mean([fr["val_f1"] for fr in fold_records]))
avg_val_loss = float(np.mean([fr["val_loss"][-1] for fr in fold_records if fr["val_loss"]]))

# ============================================================
# CV Outputs (Best Trial 기준)
#   - Fold별 Confusion Matrix
#   - Fold별 Loss Curve
#   - Fold별 Validation Report
#   - 평균 Confusion Matrix
#   - 평균 Validation Report
# ============================================================
cm_sum = np.zeros((len(labels_int_order), len(labels_int_order)), dtype=np.float32)

for fr in fold_records:
    fold_no = fr["fold"]
    y_true_fold = fr["y_true"]
    y_pred_fold = fr["y_pred"]

    cm_fold = confusion_matrix(
        y_true_fold,
        y_pred_fold,
        labels=labels_int_order
    )
    cm_sum += cm_fold.astype(np.float32)

    # Fold별 CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_fold,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{model_name} Confusion Matrix (Trial {best_trial_no}, Fold {fold_no})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fold_cm_path = os.path.join(
        save_dir_eval,
        f"{model_name}_confusion_matrix_fold{fold_no}.png"
    )
    plt.tight_layout()
    plt.savefig(fold_cm_path)
    plt.close()

    # Fold별 Loss Curve
    train_loss = fr["train_loss"]
    val_loss = fr["val_loss"]

    if train_loss or val_loss:
        plt.figure(figsize=(7, 5))
        if train_loss:
            plt.plot(train_loss, label="Train Loss")
        if val_loss:
            plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("multi_logloss")
        plt.title(
            f"{model_name} Loss Curve (Trial {best_trial_no}, Fold {fold_no})"
        )
        plt.legend()
        plt.grid(True)
        fold_loss_path = os.path.join(
            save_dir_eval,
            f"{model_name}_loss_curve_fold{fold_no}.png"
        )
        plt.tight_layout()
        plt.savefig(fold_loss_path)
        plt.close()

    # Fold별 Validation Report
    val_acc_fold = fr["val_acc"]
    val_f1_fold = fr["val_f1"]
    report_fold = classification_report(
        y_true_fold,
        y_pred_fold,
        labels=labels_int_order,
        target_names=class_names,
        digits=4
    )

    fold_report_path = os.path.join(
        save_dir_eval,
        f"{model_name}_test_report_fold{fold_no}.txt"
    )
    with open(fold_report_path, "w") as f:
        f.write(
            f"[Fold {fold_no}] Validation report (Best trial {best_trial_no})\n"
        )
        f.write(f"Val_Accuracy : {val_acc_fold:.4f}\n")
        f.write(f"Val_Macro-F1 : {val_f1_fold:.4f}\n\n")
        f.write(report_fold)

# 평균 Confusion Matrix (row-normalized)
row_sums = cm_sum.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
cm_avg_norm = cm_sum / row_sums

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_avg_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(
    f"{model_name} Confusion Matrix (Best Trial {best_trial_no} Avg over {N_SPLITS} folds)"
)
plt.xlabel("Predicted")
plt.ylabel("True")
cm_avg_path = os.path.join(
    save_dir_eval,
    f"{model_name}_confusion_matrix_avg.png"
)
plt.tight_layout()
plt.savefig(cm_avg_path)
plt.close()

# 평균 Validation Report
with open(cv_avg_report_path, "w") as f:
    f.write(
        f"[Best trial {best_trial_no}] Cross-validation average over {N_SPLITS} folds\n"
    )
    f.write(f"Avg Val_Accuracy : {avg_val_acc:.4f}\n")
    f.write(f"Avg Val_Macro-F1 : {avg_val_macro_f1:.4f}\n")
    f.write(f"Avg Val_Loss     : {avg_val_loss:.4f}\n")

# Best Fold 대표 Loss Curve (전체용)
best_train_loss = best_fold_record["train_loss"]
best_val_loss = best_fold_record["val_loss"]

if best_train_loss or best_val_loss:
    plt.figure(figsize=(7, 5))
    if best_train_loss:
        plt.plot(best_train_loss, label="Train Loss")
    if best_val_loss:
        plt.plot(best_val_loss, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("multi_logloss")
    plt.title(
        f"{model_name} Best Fold Loss Curve "
        f"(Best trial {best_trial_no}, Fold {best_fold_idx})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_png_path)
    plt.close()

# ============================================================
# Final Model = Best Trial의 Best Fold 모델 재학습
# ============================================================
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

best_model = LGBMClassifier(
    **study.best_params,
    objective="multiclass",
    random_state=SEED,
    n_jobs=-1
)

# 여기서는 다시 한 번 early_stopping 사용 (optional)
best_model.fit(
    X_tr_final, y_tr_final,
    eval_set=[(X_tr_final, y_tr_final), (X_val_final, y_val_final)],
    eval_metric="multi_logloss",
    callbacks=[early_stopping(stopping_rounds=10), log_evaluation(period=0)]
)

final_ckpt_path = os.path.join(
    save_dir_model,
    f"{model_name}_trial{best_trial_no}_fold{best_fold_idx}_best.pkl"
)
joblib.dump(best_model, final_ckpt_path)

# ============================================================
# 테스트 평가 (Hold-out 20%)
# ============================================================
if getattr(best_model, "best_iteration_", None) is not None:
    y_pred_enc = best_model.predict(X_te20, num_iteration=best_model.best_iteration_)
else:
    y_pred_enc = best_model.predict(X_te20)

y_test = y_te20
y_pred = y_pred_enc

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

report = classification_report(
    y_test,
    y_pred,
    labels=labels_int_order,
    target_names=class_names,
    digits=4
)

cm_test = confusion_matrix(
    y_test,
    y_pred,
    labels=labels_int_order
)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(f"{model_name} Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(confusion_png_path)
plt.close()

with open(test_report_path, "w") as f:
    f.write(f"Test Accuracy : {acc:.4f}\n")
    f.write(f"Test Macro-F1 : {macro_f1:.4f}\n\n")
    f.write("[Test] Classification Report\n")
    f.write(report)

# ============================================================
# meta.json 저장
# ============================================================
end_time_total = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
meta = {
    "start_time": start_time_total,
    "end_time": end_time_total,
    "model_name": model_name,
    "seed": SEED,
    "n_trials": N_TRIALS,
    "n_splits": N_SPLITS,
    "best_trial": int(best_trial_no),
    "best_fold": int(best_fold_idx),
    "best_iteration": int(best_iteration),
    "avg_val_macro_f1": round(avg_val_macro_f1, 4),
    "avg_val_accuracy": round(avg_val_acc, 4),
    "avg_val_loss": round(avg_val_loss, 4),
    "class_names": class_names,
    "num_features": len(feat_cols),
    "feature_cols": feat_cols,
    "dataset": os.path.basename(CSV_PATH),
    "early_stopping": {
        "patience": 10,
        "note": "LightGBM early_stopping(stopping_rounds=10) 사용"
    },
    "best_params": study.best_params,
    "cv_folds": [
        {
            "fold": int(fr["fold"]),
            "val_accuracy": round(float(fr["val_acc"]), 4),
            "val_macro_f1": round(float(fr["val_f1"]), 4),
            "val_loss_last": round(float(fr["val_loss"][-1]), 4) if fr["val_loss"] else None,
            "best_iteration": int(fr["best_iteration"])
        }
        for fr in fold_records
    ],
    "paths": {
        "final_ckpt": final_ckpt_path,
        "train_log": train_log_path,
        "test_report": test_report_path,
        "cv_report_avg": cv_avg_report_path,
        "confusion_matrix_png_test": confusion_png_path,
        "confusion_matrix_png_cv_avg": cm_avg_path,
        "loss_curve_png_best_fold": loss_curve_png_path,
        "eval_dir": save_dir_eval,
        "model_dir": save_dir_model
    },
    "test_accuracy": round(float(acc), 4),
    "test_macro_f1": round(float(macro_f1), 4),
    "note": "Final model = best fold of best trial (trained on 80% train split, evaluated on 20% hold-out test)"
}
with open(meta_json_path, "w") as f:
    json.dump(meta, f, indent=2)

log_file.write("\n=== SUMMARY ===\n")
log_file.write(f"Optuna study start: {study_start_iso}\n")
log_file.write(f"Optuna study end  : {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}\n")
log_file.write(
    f"Best trial (avg val_macro_f1): {best_trial_no} | value={study.best_value:.4f}\n"
)
log_file.write(
    f"Best fold in best trial: {best_fold_idx} | "
    f"val_macro_f1={best_fold_record['val_f1']:.4f}\n"
)
log_file.write(
    f"CV avg val_acc: {avg_val_acc:.4f}, "
    f"avg val_macro_f1: {avg_val_macro_f1:.4f}, "
    f"avg val_loss: {avg_val_loss:.4f}\n"
)
log_file.write(f"Used model ckpt: {final_ckpt_path}\n")
log_file.close()

print("✅ DONE")
print(f"Best trial : {best_trial_no}, Best fold : {best_fold_idx}")
print(f"Final model saved at: {final_ckpt_path}")
print(f"Test report         : {test_report_path}")
print(f"Train log           : {train_log_path}")
print(f"CV avg report       : {cv_avg_report_path}")
print(f"Meta                : {meta_json_path}")