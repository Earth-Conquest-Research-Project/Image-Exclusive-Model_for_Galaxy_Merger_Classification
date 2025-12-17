# -*- coding: utf-8 -*-
"""
MLP (Tabular-only) Training with 5-Fold CV
Dataset: final_12_datasetPhase_complete.csv

Features (12):
StellarMass, AbsMag_g, AbsMag_r, AbsMag_i, AbsMag_z,
color_gr, color_gi, SFR, BulgeMass, EffectiveRadius,
VelocityDispersion, Metallicity

Label: Phase (-1,0,1) → mapped to (1,0,2) → final classes: {0:non, 1:pre, 2:post}

Outputs (saved under EVAL_DIR):
- MLP_confusion_matrix_foldX.png / _avg.png
- MLP_loss_curve_foldX.png
- MLP_test_report_foldX.txt / _avg.txt
- MLP_train_log_foldX.txt
- MLP_meta.json
"""

import os, time, json, random, csv
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
CSV_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/Illustris/Illustris_preprocess_SFR_no.csv" 

MODEL_NAME = "MLP"
MODEL_DIR = f"/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/model/deepLearning/SFR_inf_remove/{MODEL_NAME}"
EVAL_DIR  = f"/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/evaluation/deepLearning/SFR_inf_remove/{MODEL_NAME}"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

CLASS_NAMES = ["non", "pre", "post"]
SEED = 42
DEVICE = torch.device("cpu")

EPOCHS = 200
BATCH_SIZE = 256
PATIENCE = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(SEED)

# ============================================================
# Dataset class
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ============================================================
# EarlyStopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, path=None):
        self.patience = patience
        self.best = None
        self.count = 0
        self.path = path
        self.stopped = False
        self.best_epoch = 0
        self.min_delta = min_delta


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

# ============================================================
# Helpers for Report Averaging
# ============================================================
def average_classification_reports(report_list):
    if not report_list: return {}
    avg_dict = {}
    keys = list(report_list[0].keys()) 
    # Initialize
    for key in keys:
        if key == 'accuracy':
            avg_dict[key] = 0.0
        else:
            avg_dict[key] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0}
    
    # Sum
    n = len(report_list)
    for r in report_list:
        for key in keys:
            if key == 'accuracy':
                avg_dict[key] += r[key]
            else:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    avg_dict[key][metric] += r[key][metric]
    
    # Average
    for key in keys:
        if key == 'accuracy':
            avg_dict[key] /= n
        else:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                avg_dict[key][metric] /= n
    return avg_dict

def dict_to_report_text(d, class_names):
    text = f"{'':>14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n\n"
    for name in class_names:
        r = d[name]
        text += f"{name:>14} {r['precision']:9.4f} {r['recall']:9.4f} {r['f1-score']:9.4f} {int(r['support']):9}\n"
    text += "\n"
    text += f"{'accuracy':>14} {'':>19} {d['accuracy']:9.4f} {int(d['macro avg']['support']):9}\n"
    m = d['macro avg']
    text += f"{'macro avg':>14} {m['precision']:9.4f} {m['recall']:9.4f} {m['f1-score']:9.4f} {int(m['support']):9}\n"
    w = d['weighted avg']
    text += f"{'weighted avg':>14} {w['precision']:9.4f} {w['recall']:9.4f} {w['f1-score']:9.4f} {int(w['support']):9}\n"
    return text

# ============================================================
# Model (MLP)
# ============================================================
class MLPPhase(nn.Module):
    def __init__(self, in_dim, num_classes=3, hidden=[512,256,128], p=0.25):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [
                nn.Linear(last, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(p)
            ]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Load Dataset
# ============================================================
if not CSV_PATH:
    print("[Warning] CSV_PATH is empty. Please set the path to your dataset.")
    # For execution safety, we might need to handle this, but assuming valid path provided:
    
df = pd.read_csv(CSV_PATH, engine="python", sep=None, quoting=csv.QUOTE_MINIMAL)
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

feature_cols = [
    "StellarMass", "AbsMag_g", "AbsMag_r", "AbsMag_i", "AbsMag_z",
    "color_gr", "color_gi", "SFR", "BulgeMass",
    "EffectiveRadius", "VelocityDispersion", "Metallicity"
]

label_col = "Phase"
PHASE_MAP = {-1: 1, 0: 0, 1: 2}

df = df.dropna(subset=feature_cols + [label_col]).copy()

df[label_col] = df[label_col].map(PHASE_MAP)
y_all = df[label_col].astype(int).to_numpy()
X_all = df[feature_cols].astype(np.float32).to_numpy()

# 80/20 fixed test split
X_tr80, X_test, y_tr80, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
)

test_loader = DataLoader(
    TabularDataset(X_test, y_test),
    batch_size=BATCH_SIZE, shuffle=False
)

# ============================================================
# 5-Fold cross validation
# ============================================================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

fold_results = []
fold_report_dicts = []
total_confusion_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
meta_val_losses = []
meta_best_epochs = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tr80, y_tr80), 1):
    print(f"\n===== Fold {fold}/5 =====")
    
    # Path setups
    ckpt_path = os.path.join(MODEL_DIR, f"fold{fold}_best.pt")
    log_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{fold}.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    
    X_tr, X_val = X_tr80[train_idx], X_tr80[val_idx]
    y_tr, y_val = y_tr80[train_idx], y_tr80[val_idx]

    train_loader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MLPPhase(in_dim=X_tr.shape[1], num_classes=3).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE, path=ckpt_path)

    train_losses, val_losses = [], []
    fold_start_time = time.time()

    # ------------------ Training Loop ----------------------
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss_total = 0
        tr_preds, tr_labels = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
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

        # Validation
        model.eval()
        val_loss_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
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

        epoch_time = time.time() - t0

        log_line = (f"[Epoch {epoch:03d}] "
                    f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
                    f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                    f"epoch_time={epoch_time:.2f}s")
        print(f"[Fold{fold}]{log_line}")
        log_file.write(log_line + "\n")

        if early_stopping.stopped:
            stop_msg = f"Early stopping at epoch {epoch} (best epoch={early_stopping.best_epoch})"
            print(f"→ {stop_msg}")
            log_file.write(stop_msg + "\n")
            break
            
    # End of fold training
    total_train_time = time.time() - fold_start_time
    log_file.write(f"Total training time: {total_train_time:.2f}s\n")
    log_file.close()

    meta_val_losses.append(early_stopping.best)
    meta_best_epochs.append(early_stopping.best_epoch)

    # -------------------------------------------------------
    # Evaluation on Test Set (Fold)
    # -------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    fold_preds, fold_labels = [], []
    start_test = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            fold_preds.extend(logits.argmax(1).cpu().numpy())
            fold_labels.extend(yb.numpy())
    test_time = time.time() - start_test

    acc = accuracy_score(fold_labels, fold_preds)
    f1_macro = f1_score(fold_labels, fold_preds, average="macro")
    
    fold_results.append({"fold": fold, "acc": acc, "f1": f1_macro})

    # 1. Confusion Matrix (Fold)
    cm = confusion_matrix(fold_labels, fold_preds)
    total_confusion_matrix += cm

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_fold{fold}.png"), dpi=160)
    plt.close()

    # 2. Loss Curve (Fold)
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss Curve Fold {fold}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{fold}.png"), dpi=160)
    plt.close()

    # 3. Test Report (Fold)
    report_dict = classification_report(fold_labels, fold_preds, target_names=CLASS_NAMES, output_dict=True)
    fold_report_dicts.append(report_dict)
    
    report_text = classification_report(fold_labels, fold_preds, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_fold{fold}.txt"), "w") as f:
        f.write(f"Test time (s): {test_time:.4f}\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro-F1     : {f1_macro:.4f}\n\n")
        f.write("[Test] Classification Report\n")
        f.write(report_text)


# ============================================================
# Post-Fold Processing: Averages
# ============================================================

# 1. Average Confusion Matrix (Normalized by True Labels)
fig, ax = plt.subplots(figsize=(5, 4))

# [수정됨] 직접 비율 계산
row_sums = total_confusion_matrix.sum(axis=1)[:, np.newaxis]
normalized_cm = np.divide(total_confusion_matrix, row_sums, out=np.zeros_like(total_confusion_matrix, dtype=float), where=row_sums != 0)

disp_avg = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=CLASS_NAMES)
# [수정됨] normalize 인자 제거
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


# ============================================================
# META JSON
# ============================================================
end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
best_fold_idx = np.argmax([r["f1"] for r in fold_results])
best_fold_num = fold_results[best_fold_idx]["fold"]

meta = {
    "start_time": start_time_str,
    "end_time": end_time_str,

    "best_fold": int(best_fold_num),
    "best_epoch": int(meta_best_epochs[best_fold_idx]),
    "avg_macro_f1": float(avg_f1),
    "avg_accuracy": float(avg_acc),
    "avg_val_loss": float(np.mean(meta_val_losses)),
    "epochs_run": EPOCHS,
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
        "note": "already applied before CSV creation"
    },

    "optimizer": "AdamW",
    "lr_scheduler": f"ReduceLROnPlateau(factor=0.5, patience=3)",
    "early_stopping": {
        "patience": PATIENCE,
        "min_delta": 0.0
    },

    "paths": {
        "best_ckpt": os.path.abspath(os.path.join(MODEL_DIR, f"fold{best_fold_num}_best.pt")),
        "train_log": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{best_fold_num}.txt")),
        "test_report": os.path.abspath(avg_report_path),
        "confusion_matrix_png": os.path.abspath(cm_avg_path),
        "loss_curve_png": os.path.abspath(os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{best_fold_num}.png"))
    },
    
    "fold_results": fold_results
}

with open(os.path.join(EVAL_DIR, f"{MODEL_NAME}_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n===== TRAINING COMPLETE =====")
print(f"Mean Accuracy  : {meta['avg_accuracy']:.4f}")
print(f"Mean Macro-F1  : {meta['avg_macro_f1']:.4f}")
print(f"Saved results → {EVAL_DIR}")