#!/usr/bin/env python3

"""
⚡ Stacking Ensemble
Base models: XGBoost + LightGBM + CatBoost
Meta model: LogisticRegression

출력:
  📁 model 저장       → OUT_BASE_MODEL / MODEL_NAME
  📁 evaluation 저장 → OUT_BASE_EVAL  / MODEL_NAME
"""

import os
import csv
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ============================================
# ⚙ SETTINGS
# ============================================
SEED = 42
N_SPLITS = 5
N_CLASSES = 3
MODEL_NAME = "lgbm_xg_cat__"

CSV_PATH = os.path.join(
    "/proj/home/ibs/spaceai_2025/ai2271056/data/simulation_data/final_12_datasetPhase_complete.csv"
)

OUT_BASE_MODEL = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/model/stacking/final_12_datasetPhase_complete"
OUT_BASE_EVAL  = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/evaluation/stacking/final_12_datasetPhase_complete"

model_dir = os.path.join(OUT_BASE_MODEL, MODEL_NAME)
eval_dir  = os.path.join(OUT_BASE_EVAL, MODEL_NAME)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

labels_int_order = [0, 1, 2]
class_names = ["non", "pre", "post"]

# ============================================
# LOAD DATA
# ============================================
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
    raise ValueError("Phase column not found")
label_col = phase_candidates[0]

PHASE_MAP_ANY = {
    "-1": 1, -1: 1, "pre": 1, "PRE": 1, "Pre": 1,
     "0": 0,  0: 0, "non": 0, "NON": 0, "Non": 0,
     "1": 2,  1: 2, "post": 2, "POST": 2, "Post": 2
}
lab = df[label_col].astype(str).str.strip()
y_all = lab.map(PHASE_MAP_ANY)
if y_all.isna().any():
    raise ValueError(f"Unexpected phase values: {sorted(df.loc[y_all.isna(), label_col].unique())}")
y_all = y_all.astype(int).to_numpy()

noise_cols = {"SubHaloID", "Snapshot", "phase_5"}
exclude = {label_col, "ID"} | {c for c in noise_cols if c in df.columns}
feat_cols = [c for c in df.columns if c not in exclude]

df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
nan_mask = ~df[feat_cols].isna().any(axis=1)
df = df.loc[nan_mask].reset_index(drop=True)
y_all = y_all[nan_mask.values]
X_all = df[feat_cols].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)

# ============================================
# BEST PARAMS
# ============================================
best_params_xgb = {
  "n_estimators": 995,
  "learning_rate": 0.08840664837782788,
  "max_depth": 12,
  "subsample": 0.9504585302990415,
  "colsample_bytree": 0.7234885282041882
}
best_params_lgbm = {
  "n_estimators": 274,
  "learning_rate": 0.06209782969418156,
  "num_leaves": 57,
  "max_depth": 10,
  "subsample": 0.7224874306281966,
  "colsample_bytree": 0.7945153120290173,
  "reg_lambda": 0.007388340048077971
}
best_params_cat = {
  "iterations": 632,
  "depth": 7,
  "learning_rate": 0.15990688863531063,
  "l2_leaf_reg": 0.17188204857378184
}

# ============================================
# STEP 1 — OOF 생성
# ============================================
oof_pred = np.zeros((len(X_train), N_CLASSES * 3), dtype=float)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    print(f"[STACK] Fold {fold}")

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=SEED,
        **best_params_xgb
    ).fit(X_tr, y_tr)
    oof_pred[val_idx, 0:3] = xgb.predict_proba(X_val)

    lgbm = LGBMClassifier(
        random_state=SEED,
        **best_params_lgbm
    ).fit(X_tr, y_tr)
    oof_pred[val_idx, 3:6] = lgbm.predict_proba(X_val)

    cat = CatBoostClassifier(
        loss_function="MultiClass",
        verbose=0,
        random_state=SEED,
        **best_params_cat
    ).fit(X_tr, y_tr)
    oof_pred[val_idx, 6:9] = cat.predict_proba(X_val)

# ============================================
# STEP 2 — META MODEL 학습
# ============================================
meta = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=200,     
    learning_rate=0.05,   
    max_depth=3,          
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,       
    random_state=SEED
)
meta.fit(oof_pred, y_train)

oof_pred_label = meta.predict(oof_pred)
oof_acc = accuracy_score(y_train, oof_pred_label)
oof_f1 = f1_score(y_train, oof_pred_label, average="macro")
print(f"[OOF] Acc={oof_acc:.4f}  MacroF1={oof_f1:.4f}")

# ============================================
# STEP 3 — TEST 예측
# ============================================
xgb_full = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=SEED,
    **best_params_xgb
).fit(X_train, y_train)

lgbm_full = LGBMClassifier(
    random_state=SEED,
    **best_params_lgbm
).fit(X_train, y_train)

cat_full = CatBoostClassifier(
    loss_function="MultiClass",
    verbose=0,
    random_state=SEED,
    **best_params_cat
).fit(X_train, y_train)

proba_test = np.hstack([
    xgb_full.predict_proba(X_test),
    lgbm_full.predict_proba(X_test),
    cat_full.predict_proba(X_test)
])
test_pred = meta.predict(proba_test)

test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average="macro")
print(f"[TEST] Acc={test_acc:.4f}  MacroF1={test_f1:.4f}")

# ============================================
# STEP 4 — SAVE EVAL (CM + REPORT + LOG)
# ============================================
cm = confusion_matrix(y_test, test_pred, labels=labels_int_order)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Stacking Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "stacking_confusion_matrix.png"))
plt.close()

# ⭐ 변경된 부분 — digits=4 추가
report = classification_report(
    y_test,
    test_pred,
    target_names=class_names,
    digits=4           # 👈 소수점 네 자리
)

with open(os.path.join(eval_dir, "stacking_classification_report.txt"), "w") as f:
    f.write(report)

with open(os.path.join(eval_dir, "stacking_train_log.txt"), "w") as f:
    f.write(f"[OOF]  Acc={oof_acc:.4f}  MacroF1={oof_f1:.4f}\n")
    f.write(f"[TEST] Acc={test_acc:.4f} MacroF1={test_f1:.4f}\n\n")
    f.write(report)

# ============================================
# STEP 5 — SAVE MODELS (BASE + META)
# ============================================
xgb_path  = os.path.join(model_dir, "xgb_full.pkl")
lgbm_path = os.path.join(model_dir, "lgbm_full.pkl")
cat_path  = os.path.join(model_dir, "cat_full.pkl")
meta_path = os.path.join(model_dir, "meta_logreg.pkl")

joblib.dump(xgb_full,  xgb_path)
joblib.dump(lgbm_full, lgbm_path)
joblib.dump(cat_full,  cat_path)
joblib.dump(meta,      meta_path)

# ============================================
# STEP 6 — SAVE META.JSON
# ============================================
meta_json = {
    "model_name": MODEL_NAME,
    "dataset": os.path.basename(CSV_PATH),
    "feature_count": X_all.shape[1],
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "classes": class_names,
    "oof": {
        "accuracy": round(oof_acc, 4),
        "macro_f1": round(oof_f1, 4)
    },
    "test": {
        "accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4)
    },
    "best_params": {
        "xgboost": best_params_xgb,
        "lightgbm": best_params_lgbm,
        "catboost": best_params_cat
    },
    "model_paths": {
        "xgb": xgb_path,
        "lgbm": lgbm_path,
        "cat": cat_path,
        "meta": meta_path
    },
    "eval_paths": {
        "confusion_matrix": os.path.join(eval_dir, "stacking_confusion_matrix.png"),
        "classification_report": os.path.join(eval_dir, "stacking_classification_report.txt"),
        "train_log": os.path.join(eval_dir, "stacking_train_log.txt")
    }
}

with open(os.path.join(model_dir, "meta.json"), "w") as f:
    json.dump(meta_json, f, indent=2)

print("\n====================================")
print("🚀 STACKING FINISHED")
print("📁 Model dir:", model_dir)
print("📁 Eval dir :", eval_dir)
print("====================================\n")