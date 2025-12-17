import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ======================================================
# 0. 경로 설정
# ======================================================
SEED=42
SAVE_DIR = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/SHAP/randomForest"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH  = "/proj/home/ibs/spaceai_2025/ai2271056/data/simulation_data/final_12_datasetPhase_complete.csv"

# ======================================================
# 1. 모델 로드
# ======================================================
model = joblib.load(
    "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/classicalMachineLearning/phase3/RandomForest/RandomForest_phase3.pkl"
)
print("✅ Model loaded.")


# ======================================================
# 2. 데이터 로드 (SHAP 전용: train 데이터만 사용)
# ======================================================
print(f"Loading CSV: {CSV_PATH}")
import csv
df = pd.read_csv(
    CSV_PATH,
    engine="python",
    sep=None,               # 자동 구분자 감지
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)

# -----------------------------
# 1) 칼럼 이름 정리
# -----------------------------
df.columns = [c.strip().strip("\ufeff") for c in df.columns]

# -----------------------------
# 2) 라벨 컬럼 찾기
# -----------------------------
phase_candidates = [c for c in df.columns if c.strip().lower() == "phase"]
if not phase_candidates:
    raise ValueError(f"'Phase' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)[:20]}")
label_col = phase_candidates[0]

# -----------------------------
# 3) 라벨 매핑
# -----------------------------
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

# -----------------------------
# 4) Feature 선택 (ID/Phase/불필요 컬럼 제외)
# -----------------------------
noise_cols = {"ID", "phase_5", "SubHaloID","Snapshot"}
present_noise = [c for c in noise_cols if c in df.columns]
exclude = {label_col} | set(present_noise)
feat_cols = [c for c in df.columns if c not in exclude]

# -----------------------------
# 5) 수치형 변환 및 NaN 제거
# -----------------------------
for c in feat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

X_all = df[feat_cols]
nan_rows = X_all.isna().any(axis=1)
if nan_rows.any():
    print(f"[warn] removing {nan_rows.sum()} rows with NaN in features")
    X_all = X_all.loc[~nan_rows].copy()
    y_all = y_all[~nan_rows.values]

X_all = X_all.values.astype(np.float32)

print(f"[info] n_samples={X_all.shape[0]}, n_features={X_all.shape[1]}")
print(f"[info] features: {feat_cols[:8]}{'...' if len(feat_cols)>8 else ''}")

# -----------------------------
# ✅ 6) 80/20 Split — SHAP에는 train 데이터만 사용
# -----------------------------
X_tr80, X_te20, y_tr80, y_te20 = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)

print(f"[split] train80 (for SHAP): {X_tr80.shape}, test20 (unused for SHAP): {X_te20.shape}")

# -----------------------------
# ✅ 7) SHAP용 데이터만 반환/사용
# -----------------------------
X_train_for_shap = X_tr80
y_train_for_shap = y_tr80

print(f"[SHAP] Using only training data: {X_train_for_shap.shape}")


# ======================================================
# 3. KernelExplainer baseline + sample 데이터 (train-only)
# ======================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ✅ train-only 데이터 사용
X_full = pd.DataFrame(X_train_for_shap, columns=feat_cols)

# ✅ baseline 50개, SHAP sample 300개
background = X_full.sample(50, random_state=0)
sample_X = X_full.sample(300, random_state=42)

print("✅ Sampling ready:", background.shape, sample_X.shape)


# ======================================================
# 4. 클래스별 KernelExplainer SHAP 계산 (핵심)
# ======================================================
class_names = ["Non-merger (0)", "Pre-merger (1)", "Post-merger (2)"]
colors = ["#78F5E8", "#78F594", "#78B2F5"]

shap_values = []  # 각 클래스 SHAP 저장

print("✅ Starting class-wise SHAP computation...")

for class_idx in range(3):
    print(f"\n=== Computing SHAP for class {class_idx}: {class_names[class_idx]} ===")

    # ✅ 특정 클래스 확률 반환 함수
    f = lambda X: model.predict_proba(X)[:, class_idx]

    # ✅ KernelExplainer (train 기반 baseline)
    explainer = shap.KernelExplainer(f, background)

    # ✅ SHAP 계산
    sv = explainer.shap_values(sample_X, nsamples=200)
    shap_values.append(np.array(sv))

    print(f" -> Class {class_idx} SHAP shape:", np.array(sv).shape)

print("✅ All 3 class SHAP computed.")


# ======================================================
# 5. Global Feature Importance
# ======================================================
global_importance = []

for class_idx in range(3):
    df_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": np.abs(shap_values[class_idx]).mean(axis=0),
        "class": class_names[class_idx]
    })
    global_importance.append(df_imp)

global_df = pd.concat(global_importance)

pivot_df = global_df.pivot(index="feature", columns="class", values="importance")
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]


# ======================================================
# 6. Global Importance Bar Plot 저장
# ======================================================
plt.figure(figsize=(10, 8))
pivot_df.plot(kind="barh", stacked=True, color=colors)
plt.gca().invert_yaxis()
plt.xlabel("mean(|SHAP value|)")
plt.title("Global Feature Importance — Kernel SHAP (RandomForest)")
plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "RandomForest_bar.png")
plt.savefig(save_path, dpi=300)
plt.close()
print("✅ Saved:", save_path)


# ======================================================
# 7. Beeswarm Plot — 클래스별
# ======================================================
for c in range(3):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[c],
        sample_X,
        feature_names=feat_cols,
        plot_type="dot",
        show=False
    )
    plt.title(f"Beeswarm — {class_names[c]} (Kernel SHAP)")
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"RandomForest_beeswarm_phase{c}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("✅ Saved:", save_path)

print("\n✅✅ All SHAP PNG files saved successfully in:")
print(SAVE_DIR)