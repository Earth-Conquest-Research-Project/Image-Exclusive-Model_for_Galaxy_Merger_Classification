#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd

# ============================
# 경로
# ============================
MODEL_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/classicalMachineLearning/phase3/RandomForest/RandomForest_phase3.pkl"

IMPUTER_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/preprocess_train_model/KNNImputer_Illustris.pkl"
SCALER_PATH  = "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/preprocess_train_model/StandardScaler_Illustris.pkl"

# ============================
# 클래스 인덱스 → 라벨 매핑
# ============================
CLASS_LABEL = {
    0: "non-merger",
    1: "pre-merger",
    2: "post-merger"
}

# ============================
# 피처 순서 (반드시 학습과 동일)
# ============================
FEATURE_NAMES = [
    "STELLARMASS", "ABSMAG_G", "ABSMAG_R", "ABSMAG_I", "ABSMAG_Z",
    "COLOR_GR", "COLOR_GI", "SFR", "BULGEMASS",
    "VELOCITYDISPERSION", "METALLICITY", "EFFECTIVERADIUS"
]

# ============================
# 전처리 + 예측 함수
# ============================
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, imputer, scaler

def preprocess_one(values, imputer, scaler):
    """
    values: list[float|np.nan] length=12
    -> imputer.transform -> scaler.transform
    """
    # 1) DataFrame으로 만들어 feature name 유지 (경고 방지)
    X_df = pd.DataFrame([values], columns=FEATURE_NAMES)

    # 2) impute / scale (fit 금지)
    X_imp = pd.DataFrame(imputer.transform(X_df), columns=FEATURE_NAMES)
    X_scl = pd.DataFrame(scaler.transform(X_imp), columns=FEATURE_NAMES)

    return X_scl.to_numpy(dtype=float)

def predict_merger(values, model, imputer, scaler):
    """
    values: list length=12 (float or np.nan)
    """
    X = preprocess_one(values, imputer, scaler)  # shape (1, 12)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    label = CLASS_LABEL[pred]
    p_nomerger = float(proba[0])  # 클래스 0 확률
    return label, p_nomerger, proba

# ============================
# CLI
# ============================
def parse_input(s: str):
    """
    빈 입력/NA/None -> NaN 처리해서 imputer가 채울 수 있게 함
    """
    s = s.strip()
    if s == "" or s.lower() in ["na", "nan", "none", "null"]:
        return np.nan
    return float(s)

def main():
    print("\n🌌 Galaxy Merger Prediction Demo (with Imputer+Scaler)\n")
    print("[INFO] Model  :", MODEL_PATH)
    print("[INFO] Imputer:", IMPUTER_PATH)
    print("[INFO] Scaler :", SCALER_PATH)

    model, imputer, scaler = load_artifacts()

    print("\n👉 아래 피처 값을 입력하세요 (Enter).")
    print("   - 비우거나(Enter) NA/nan 입력하면 결측으로 처리되고, imputer가 채웁니다.")
    print("   순서:", ", ".join(FEATURE_NAMES), "\n")

    values = []
    for name in FEATURE_NAMES:
        v = input(f"  {name}: ")
        values.append(parse_input(v))

    label, p_nm, proba = predict_merger(values, model, imputer, scaler)

    print("\n🔮 Prediction Result")
    print("---------------------------")
    print(f"  Predicted class : {label}")
    print(f"  P_NOMERGER      : {p_nm:.4f}")
    print(f"  Probabilities   : non={proba[0]:.4f}, pre={proba[1]:.4f}, post={proba[2]:.4f}")
    print("---------------------------")

if __name__ == "__main__":
    main()
