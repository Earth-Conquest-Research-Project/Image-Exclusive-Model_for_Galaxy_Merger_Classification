import pandas as pd
import numpy as np
import joblib
import os

# ======================================
# 경로 설정
# ======================================
MODEL_PATH = "./model/classicalMachineLearning/SFR_inf_-1/RandomForest/RandomForest_best_fullTrain.pkl"  # 모델 pkl 경로
DATA_PATH  = "./data/DESI/DESI_raw.csv"  # 추론 입력 CSV
SAVE_PATH  = "./inference/randomforest_final12_inference.csv"  # 결과 저장 경로

# ======================================
# 불러올 학습 피처 (훈련 당시 사용한 것)
# ======================================
train_features = [
    "StellarMass","AbsMag_g","AbsMag_r","AbsMag_i","AbsMag_z",
    "color_gr","color_gi","SFR","BulgeMass","EffectiveRadius",
    "VelocityDispersion","Metallicity"
]

# ======================================
# 추론 데이터 컬럼 → 학습 데이터 컬럼명으로 매핑
# ======================================
inference_to_train_map = {
    "STELLARMASS":"StellarMass",
    "ABSMAG_G":"AbsMag_g",
    "ABSMAG_R":"AbsMag_r",
    "ABSMAG_I":"AbsMag_i",
    "ABSMAG_Z":"AbsMag_z",
    "COLOR_GR":"color_gr",
    "COLOR_GI":"color_gi",
    "SFR":"SFR",
    "BULGEMASS":"BulgeMass",
    "EFFECTIVERADIUS":"EffectiveRadius",
    "VELOCITYDISPERSION":"VelocityDispersion",
    "METALLICITY":"Metallicity",
}

# ======================================
# ➊ 모델 + 데이터 로드
# ======================================
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ======================================
# ➋ Feature 매핑
# ======================================
df_renamed = df.rename(columns=inference_to_train_map)

# 추론용 feature matrix 준비
X = df_renamed[train_features]

# ======================================
# ➌ 예측 + 확신도 계산
# ======================================
pred = model.predict(X)
proba = model.predict_proba(X)

# 예측 확신도 = 예측한 class의 확률
conf = proba.max(axis=1)

# ======================================
# ➍ 결과 CSV 구성
# ======================================
out = pd.DataFrame({
    "RA": df["RA"],
    "DEC": df["DEC"],
    "REDSHIFT": df["REDSHIFT"],
    "pred_class": pred,
    "confidence": conf,
    "P_NOMERGER": df["P_NOMERGER"]
})

# ======================================
# ➎ 저장
# ======================================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
out.to_csv(SAVE_PATH, index=False)

print("🔥 저장 완료:", SAVE_PATH)