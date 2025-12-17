# -*- coding: utf-8 -*-
"""
FT-Transformer (Tabular-only) 5-Fold Training
Dataset: final_12_datasetPhase_complete.csv
Features: 12 physics-related columns
Label: Phase (-1,0,1 → mapped to 1,0,2)
Outputs:
- FT-Transformer_confusion_matrix_foldX.png / _avg.png
- FT-Transformer_loss_curve_foldX.png
- FT-Transformer_test_report_foldX.txt / _avg.txt
- FT-Transformer_train_log_foldX.txt
- FT-Transformer_meta.json
"""

import os, json, time, csv, random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rtdl_revisiting_models import FTTransformer


# =============================================================
# Config
# =============================================================
CSV_PATH = "./data/Illustris/Illustris_preprocess_SFR_-1.csv"

MODEL_NAME = "FT-Transformer"
SAVE_DIR = f"./model/deepLearning/SFR_inf_-1/{MODEL_NAME}"
EVAL_DIR = f"./evaluation/deepLearning/SFR_inf_-1/{MODEL_NAME}"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

CLASS_NAMES = ["non", "pre", "post"]  # 0,1,2
DEVICE = torch.device("cpu")
SEED = 42
EPOCHS = 200
BATCH_SIZE = 256
PATIENCE = 15
LR_FACTOR = 0.5

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed()


# =============================================================
# Dataset
# =============================================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# =============================================================
# Load dataset
# =============================================================
# CSV_PATH가 비어있을 경우 예외처리 혹은 더미 데이터 처리 필요
if not CSV_PATH or not os.path.exists(CSV_PATH):
    print(f"[Warning] CSV_PATH not found: {CSV_PATH}. Please check the path.")

df = pd.read_csv(CSV_PATH, engine="python", sep=None, quoting=csv.QUOTE_MINIMAL)
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

feature_cols = [
    "StellarMass", "AbsMag_g", "AbsMag_r", "AbsMag_i", "AbsMag_z",
    "color_gr", "color_gi", "SFR", "BulgeMass", "EffectiveRadius",
    "VelocityDispersion", "Metallicity"
]

label_col = "Phase"
PHASE_MAP = {-1: 1, 0: 0, 1: 2}

df = df.dropna(subset=feature_cols + [label_col]).copy()

df[label_col] = df[label_col].map(PHASE_MAP)
y_all = df[label_col].astype(int).to_numpy()
X_all = df[feature_cols].astype(np.float32).to_numpy()

print(f"[info] Loaded {len(df)} rows, {len(feature_cols)} features")


# =============================================================
# Train/test split
# =============================================================
X_tr80, X_test, y_tr80, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# =============================================================
# EarlyStopping
# =============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.best_epoch = 0
        self.path = path
        self.stopped = False

    def __call__(self, val_loss, model, epoch):
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.count = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stopped = True


# =============================================================
# Helper function for Report Averaging
# =============================================================
def average_classification_reports(report_list):
    """
    List of classification_report dicts -> Averaged dict
    """
    if not report_list:
        return {}
    
    # Initialize with 0
    avg_dict = {}
    keys = list(report_list[0].keys()) # 'non', 'pre', 'post', 'accuracy', 'macro avg', 'weighted avg'
    
    # Structure for aggregation
    # For classes and avgs: store precision, recall, f1-score, support
    # For accuracy: just score
    
    for key in keys:
        if key == 'accuracy':
            avg_dict[key] = 0.0
        else:
            avg_dict[key] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0}

    n_folds = len(report_list)
    
    for r in report_list:
        for key in keys:
            if key == 'accuracy':
                avg_dict[key] += r[key]
            else:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    avg_dict[key][metric] += r[key][metric]
                    
    # Divide by n_folds
    for key in keys:
        if key == 'accuracy':
            avg_dict[key] /= n_folds
        else:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                avg_dict[key][metric] /= n_folds
                
    return avg_dict

def dict_to_report_text(d, class_names):
    """
    Reconstruct text report from dict
    """
    # Header
    width = 55
    text = f"{'':>14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n\n"
    
    # Rows
    for name in class_names:
        r = d[name]
        text += f"{name:>14} {r['precision']:9.4f} {r['recall']:9.4f} {r['f1-score']:9.4f} {int(r['support']):9}\n"
    
    text += "\n"
    # Accuracy
    acc = d['accuracy']
    total_support = int(d['macro avg']['support']) # support is sum, so avg support * folds is weird, but for per-fold avg, it's avg support
    text += f"{'accuracy':>14} {'':>19} {acc:9.4f} {total_support:9}\n"
    
    # Macro avg
    m = d['macro avg']
    text += f"{'macro avg':>14} {m['precision']:9.4f} {m['recall']:9.4f} {m['f1-score']:9.4f} {int(m['support']):9}\n"
    
    # Weighted avg
    w = d['weighted avg']
    text += f"{'weighted avg':>14} {w['precision']:9.4f} {w['recall']:9.4f} {w['f1-score']:9.4f} {int(w['support']):9}\n"
    
    return text


# =============================================================
# K-Fold
# =============================================================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_results = []
fold_report_dicts = []
total_confusion_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

# Meta info collection
meta_val_losses = []
meta_best_epochs = []

start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_tr80, y_tr80), 1):
    print(f"\n===== Fold {fold}/5 =====")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    ckpt_path = os.path.join(fold_dir, "best.pt")
    
    # Fold Train Log File
    train_log_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{fold}.txt")
    train_log_file = open(train_log_path, "w", encoding="utf-8")

    X_tr, X_val = X_tr80[tr_idx], X_tr80[val_idx]
    y_tr, y_val = y_tr80[tr_idx], y_tr80[val_idx]

    train_loader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = FTTransformer(
        n_cont_features=X_tr.shape[1],
        cat_cardinalities=[],
        d_out=len(CLASS_NAMES),
        **FTTransformer.get_default_kwargs()
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = model.make_default_optimizer()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=LR_FACTOR)
    early_stopping = EarlyStopping(patience=PATIENCE, path=ckpt_path)

    train_losses, val_losses = [], []
    fold_start_time = time.time() # For total training time calculation

    # -----------------------------
    # Train Loop
    # -----------------------------
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        tr_loss_total = 0
        tr_preds, tr_labels = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, None)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss_total += loss.item() * xb.size(0)
            tr_preds.append(logits.argmax(1).cpu())
            tr_labels.append(yb.cpu())

        tr_preds = torch.cat(tr_preds).numpy()
        tr_labels = torch.cat(tr_labels).numpy()
        tr_loss = tr_loss_total / len(train_loader.dataset)
        tr_acc = accuracy_score(tr_labels, tr_preds)
        tr_f1 = f1_score(tr_labels, tr_preds, average="macro")

        # ---- Validation ----
        model.eval()
        val_loss_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb, None)
                loss = criterion(logits, yb)
                val_loss_total += loss.item() * xb.size(0)
                val_preds.append(logits.argmax(1).cpu())
                val_labels.append(yb.cpu())
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_loss = val_loss_total / len(valid_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        scheduler.step(val_loss)
        early_stopping(val_loss, model, epoch)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        log_line = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
            f"epoch_time={epoch_time:.2f}s"
        )
        print(f"[Fold{fold}]{log_line}")
        train_log_file.write(log_line + "\n")

        if early_stopping.stopped:
            stop_msg = f"Early stopping at epoch {epoch} (best epoch={early_stopping.best_epoch})"
            print(f"→ {stop_msg}")
            train_log_file.write(stop_msg + "\n")
            break
            
    # End of Fold Loop Logging
    total_train_time = time.time() - fold_start_time
    train_log_file.write(f"Total training time: {total_train_time:.2f}s\n")
    train_log_file.close()
    
    meta_val_losses.append(early_stopping.best)
    meta_best_epochs.append(early_stopping.best_epoch)

    # ----------------------------------------------------------
    # Load best checkpoint & evaluate on test set
    # ----------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    preds, labels = [], []
    start_test = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE), None)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    test_time = time.time() - start_test

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    # 1. Confusion Matrix (Fold)
    cm = confusion_matrix(labels, preds)
    total_confusion_matrix += cm  # Accumulate for average

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_fold{fold}.png")
    plt.savefig(cm_path, dpi=160)
    plt.close()

    # 2. Loss Curve (Fold)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.title(f"Loss Curve Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{fold}.png")
    plt.savefig(loss_path, dpi=160)
    plt.close()

    # 3. Test Report (Fold)
    report_dict = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)
    fold_report_dicts.append(report_dict)
    
    report_text = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    report_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_fold{fold}.txt")
    with open(report_path, "w") as f:
        f.write(f"Test time (s): {test_time:.4f}\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro-F1     : {f1_macro:.4f}\n\n")
        f.write("[Test] Classification Report\n")
        f.write(report_text)

    fold_results.append({"fold": fold, "acc": acc, "f1": f1_macro})


# =============================================================
# Post-Fold Processing: Averages
# =============================================================

# 1. Average Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))

# [수정] plot() 내부 옵션 대신, 데이터 자체를 먼저 정규화(Normalize)합니다.
# 각 행(Row)의 합으로 나누어 Recall 비율을 계산
row_sums = total_confusion_matrix.sum(axis=1)[:, np.newaxis]
# 0으로 나누는 에러 방지
normalized_cm = np.divide(total_confusion_matrix, row_sums, out=np.zeros_like(total_confusion_matrix, dtype=float), where=row_sums != 0)

# [수정] 정규화된 데이터(normalized_cm)를 Display 객체에 전달
disp_avg = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=CLASS_NAMES)

# [수정] normalize='true' 옵션 제거
disp_avg.plot(ax=ax, cmap="Blues", colorbar=False, values_format='.2f') 

plt.title(f"Confusion Matrix Average")
plt.tight_layout()
cm_avg_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_avg.png")
plt.savefig(cm_avg_path, dpi=160)
plt.close()

# 2. Average Test Report
avg_report_dict = average_classification_reports(fold_report_dicts)
avg_report_text = dict_to_report_text(avg_report_dict, CLASS_NAMES)
avg_acc = avg_report_dict['accuracy']
avg_f1 = avg_report_dict['macro avg']['f1-score']

avg_report_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_avg.txt")
with open(avg_report_path, "w") as f:
    f.write(f"Accuracy     : {avg_acc:.4f}\n")
    f.write(f"Macro-F1     : {avg_f1:.4f}\n\n")
    f.write("[Average Test] Classification Report\n")
    f.write(avg_report_text)


# =============================================================
# META JSON
# =============================================================
end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Determine Best Fold (by Macro-F1)
best_fold_idx = np.argmax([r["f1"] for r in fold_results])
best_fold_num = fold_results[best_fold_idx]["fold"]
best_fold_epoch = meta_best_epochs[best_fold_idx]

meta = {
    "start_time": start_time_str,
    "end_time": end_time_str,

    "best_fold": int(best_fold_num),
    "best_epoch": int(best_fold_epoch),
    "avg_macro_f1": float(avg_f1),
    "avg_accuracy": float(avg_acc),
    "avg_val_loss": float(np.mean(meta_val_losses)),
    "epochs_run": EPOCHS, # Max epochs config
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "device": str(DEVICE),
    "class_names": CLASS_NAMES,

    "num_features": len(feature_cols),
    "feature_cols": feature_cols,
    "dataset": CSV_PATH,
    "preprocessing": {
        "standard_scaler": True,
        "knn": True,
        "note": "already applied in dataset (MICE)"
    },

    "optimizer": "AdamW (model.make_default_optimizer())",
    "lr_scheduler": f"ReduceLROnPlateau(mode='min', patience=3, factor={LR_FACTOR})",
    "early_stopping": {"patience": PATIENCE, "min_delta": 0.0},
    
    "paths": {
        "best_ckpt": os.path.abspath(os.path.join(SAVE_DIR, f"fold{best_fold_num}", "best.pt")),
        "train_log": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{best_fold_num}.txt")), # Represents best fold
        "test_report": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_avg.txt")),
        "confusion_matrix_png": os.path.abspath(cm_avg_path),
        "loss_curve_png": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{best_fold_num}.png"))
    },
    
    "fold_results": fold_results
}

with open(os.path.join(EVAL_DIR, f"{MODEL_NAME}_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n===== Training Completed =====")
print(f"Mean Accuracy : {meta['avg_accuracy']:.4f}")
print(f"Mean Macro-F1 : {meta['avg_macro_f1']:.4f}")
print(f"Saved outputs → {EVAL_DIR}")