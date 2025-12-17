# -*- coding: utf-8 -*-
"""
TabM (Tabular-only) model training with 5-Fold CV (CPU-only)
Dataset: final_12_datasetPhase_complete.csv

Outputs saved under EVAL_DIR:
- TabM_confusion_matrix_foldX.png / _avg.png
- TabM_loss_curve_foldX.png
- TabM_test_report_foldX.txt / _avg.txt
- TabM_train_log_foldX.txt
- TabM_meta.json
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

from tabm import TabM


# ============================================================
# Config
# ============================================================
CSV_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/Illustris/Illustris_preprocess_SFR_no.csv"

MODEL_NAME = "TabM"
MODEL_DIR = f"/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/model/deepLearning/SFR_inf_remove/{MODEL_NAME}"
EVAL_DIR  = f"/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/evaluation/deepLearning/SFR_inf_remove/{MODEL_NAME}"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

CLASS_NAMES = ["non", "pre", "post"]
SEED = 42
DEVICE = torch.device("cpu")

EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
K = 32   # TabM parameter k


# ============================================================
# Utils
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


class TabMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best = None
        self.count = 0
        self.stopped = False
        self.best_epoch = 0

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

# Helper to average classification reports
def average_reports(reports):
    avg_report = {}
    if not reports: return avg_report

    # Initialize keys
    keys = list(reports[0].keys()) # classes + accuracy, macro avg, weighted avg
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
    
    # Divide
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

X_all = df[feature_cols].astype(np.float32).to_numpy()
y_all = df[label_col].astype(int).to_numpy()

# 80/20 split
X_tr80, X_test, y_tr80, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
)

test_loader = DataLoader(TabMDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Train Loop (5-Fold)
# ============================================================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

fold_results = []
fold_reports = []
total_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=float)
meta_val_losses = []
meta_best_epochs = []

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_tr80, y_tr80), 1):
    print(f"\n===== Fold {fold}/5 =====")
    
    # Fold-specific paths
    ckpt_path = os.path.join(MODEL_DIR, f"fold{fold}_best.pt")
    log_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_train_log_fold{fold}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    X_tr, X_val = X_tr80[tr_idx], X_tr80[val_idx]
    y_tr, y_val = y_tr80[tr_idx], y_tr80[val_idx]

    train_loader = DataLoader(TabMDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TabMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    num_features = X_tr.shape[1]
    model = TabM.make(
        n_num_features=num_features,
        cat_cardinalities=[],
        d_out=len(CLASS_NAMES),
        k=K
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    early_stopping = EarlyStopping(patience=PATIENCE, path=ckpt_path)

    train_losses, val_losses = [], []
    fold_start_time = time.time()

    # ------------------ Epoch Loop ------------------
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss_total = 0
        tr_preds, tr_labels = [], []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb.to(DEVICE), None).mean(dim=1)
            loss = criterion(logits, yb.to(DEVICE))
            loss.backward()
            optimizer.step()

            tr_loss_total += loss.item() * xb.size(0)
            tr_preds.append(logits.argmax(1).cpu())
            tr_labels.append(yb.cpu())

        tr_loss = tr_loss_total / len(train_loader.dataset)
        tr_preds = torch.cat(tr_preds).numpy()
        tr_labels = torch.cat(tr_labels).numpy()
        tr_acc = accuracy_score(tr_labels, tr_preds)
        tr_f1 = f1_score(tr_labels, tr_preds, average="macro")

        # Validation
        model.eval()
        val_loss_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                logits = model(xb.to(DEVICE), None).mean(dim=1)
                loss = criterion(logits, yb.to(DEVICE))
                val_loss_total += loss.item() * xb.size(0)
                val_preds.append(logits.argmax(1).cpu())
                val_labels.append(yb.cpu())

        val_loss = val_loss_total / len(valid_loader.dataset)
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        scheduler.step(val_loss)
        early_stopping(val_loss, model, epoch)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - t0

        log_line = (
            f"[Fold {fold}][Epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
            f"epoch_time={epoch_time:.2f}s"
        )
        print(log_line)
        log_file.write(log_line + "\n")

        if early_stopping.stopped:
            stop_msg = f"Early stopping at epoch {epoch} (best={early_stopping.best_epoch})"
            print(f"→ {stop_msg}")
            log_file.write(stop_msg + "\n")
            break

    total_train_time = time.time() - fold_start_time
    log_file.write(f"Total training time: {total_train_time:.2f}s\n")
    log_file.close()

    meta_val_losses.append(early_stopping.best)
    meta_best_epochs.append(early_stopping.best_epoch)

    # -------------------------------------------------------
    # Evaluate on Test Set with Best Model of This Fold
    # -------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    fold_preds, fold_labels = [], []
    start_test = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE), None).mean(dim=1)
            fold_preds.extend(logits.argmax(1).cpu().numpy())
            fold_labels.extend(yb.numpy())
    test_time = time.time() - start_test

    fold_preds = np.array(fold_preds)
    fold_labels = np.array(fold_labels)

    acc = accuracy_score(fold_labels, fold_preds)
    f1_macro = f1_score(fold_labels, fold_preds, average="macro")
    
    fold_results.append({"fold": fold, "acc": acc, "f1": f1_macro})
    
    # 1. Confusion Matrix (Fold)
    cm = confusion_matrix(fold_labels, fold_preds)
    total_cm += cm # Accumulate for average
    
    fig, ax = plt.subplots(figsize=(5,4))
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
    plt.title(f"Loss Curve Fold {fold}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"{MODEL_NAME}_loss_curve_fold{fold}.png"), dpi=160)
    plt.close()

    # 3. Test Report (Fold)
    report_dict = classification_report(fold_labels, fold_preds, target_names=CLASS_NAMES, output_dict=True)
    fold_reports.append(report_dict)
    
    report_text = classification_report(fold_labels, fold_preds, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(EVAL_DIR, f"{MODEL_NAME}_test_report_fold{fold}.txt"), "w") as f:
        f.write(f"Test time (s): {test_time:.4f}\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro-F1     : {f1_macro:.4f}\n\n")
        f.write("[Test] Classification Report\n")
        f.write(report_text)


# ============================================================
# Average Outputs
# ============================================================

# 1. Average Confusion Matrix (Normalized)
# [Image of confusion matrix calculation]

fig, ax = plt.subplots(figsize=(5,4))

# Manually normalize the confusion matrix
row_sums = total_cm.sum(axis=1)[:, np.newaxis]
normalized_cm = np.divide(total_cm, row_sums, out=np.zeros_like(total_cm), where=row_sums != 0)

disp_avg = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=CLASS_NAMES)
# Removed 'normalize' argument to fix TypeError
disp_avg.plot(ax=ax, cmap="Blues", colorbar=False, values_format='.2f')

plt.title("Confusion Matrix Average")
plt.tight_layout()
cm_avg_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_confusion_matrix_avg.png")
plt.savefig(cm_avg_path, dpi=160)
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
        "note": "already applied"
    },

    "optimizer": "AdamW",
    "lr_scheduler": "ReduceLROnPlateau",

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

json_path = os.path.join(EVAL_DIR, f"{MODEL_NAME}_meta.json")
with open(json_path, "w") as f:
    json.dump(meta, f, indent=2)

print("\n===== TRAINING COMPLETE =====")
print(f"Mean Accuracy : {avg_acc:.4f}")
print(f"Mean Macro-F1 : {avg_f1:.4f}")
print(f"Saved results → {EVAL_DIR}")