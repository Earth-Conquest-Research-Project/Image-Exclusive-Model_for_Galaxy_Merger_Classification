import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import joblib



# ============================================================
# 1) 경로 설정
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Illustris", "Illustris_raw.csv")
SAVE_PATH_NO = os.path.join(PROJECT_ROOT, "data", "Illustris", "Illustris_preprocess_SFR_no.csv")
SAVE_PATH_MINUS1 = os.path.join(PROJECT_ROOT,"data", "Illustris", "Illustris_preprocess_SFR_-1.csv")

PREPROCESS_DIR = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/model/preprocess_train_model"
os.makedirs(PREPROCESS_DIR, exist_ok=True)

IMPUTER_PATH_NO = os.path.join(PREPROCESS_DIR, "KNNImputer_SFR_no.pkl")
SCALER_PATH_NO  = os.path.join(PREPROCESS_DIR, "StandardScaler_SFR_no.pkl")

IMPUTER_PATH_M1 = os.path.join(PREPROCESS_DIR, "KNNImputer_SFR_-1.pkl")
SCALER_PATH_M1  = os.path.join(PREPROCESS_DIR, "StandardScaler_SFR_-1.pkl")

# ============================================================
# 2) 데이터 로드
# ============================================================
print("[INFO] Loading raw data from:", DATA_DIR)
df = pd.read_csv(DATA_DIR)

print("\n=== Raw Dataset Info ===")
print(df.shape)
print(df.columns)


# ============================================================
# 3) SFR = -4 개수 확인
# ============================================================
if "SFR" not in df.columns:
    raise KeyError("ERROR: 데이터에 'SFR' 컬럼이 없습니다. 컬럼명을 확인하세요.")

num_minus4 = (df["SFR"] == -4).sum()
print(f"\n[INFO] SFR = -4 count: {num_minus4}")


# ============================================================
# 전처리에 사용될 공통 함수 정의
# ============================================================
def preprocess_df(input_df, imputer_save_path, scaler_save_path):
    """
    KNN → StandardScaler 적용 (지정된 컬럼 제외) + imputer/scaler 저장
    """
    df_copy = input_df.copy()

    exclude_cols = ["SubHaloID", "Snapshot", "Phase", "ID"]
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [c for c in numeric_cols if c not in exclude_cols]

    print("\n[INFO] Scaling target columns:", len(scale_cols))

    # KNN Imputer
    imputer = KNNImputer(n_neighbors=10)
    df_copy[scale_cols] = imputer.fit_transform(df_copy[scale_cols])
    print("[INFO] KNN Imputer 완료")

    # StandardScaler
    scaler = StandardScaler()
    df_copy[scale_cols] = scaler.fit_transform(df_copy[scale_cols])
    print("[INFO] StandardScaler 완료")

    # ✅ 저장
    joblib.dump(imputer, imputer_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"[SAVE] Imputer saved to: {imputer_save_path}")
    print(f"[SAVE] Scaler  saved to: {scaler_save_path}")

    return df_copy



# ============================================================
# 4) Version A: SFR = -4 제거 후 전처리
# ============================================================
print("\n========================")
print(" Version A: SFR -4 제거 ")
print("========================")

df_no = df[df["SFR"] != -4].reset_index(drop=True)
print("[INFO] After removing SFR=-4:", df_no.shape)

df_no_processed = preprocess_df(df_no, IMPUTER_PATH_NO, SCALER_PATH_NO)

# 저장
df_no_processed.to_csv(SAVE_PATH_NO, index=False)
print(f"[SAVE] Version A saved to: {SAVE_PATH_NO}")


# ============================================================
# 5) Version B: SFR = -4 → -1로 치환 후 전처리
# ============================================================
print("\n===========================")
print(" Version B: SFR -4 → -1 처리 ")
print("===========================")

df_minus1 = df.copy()
df_minus1.loc[df_minus1["SFR"] == -4, "SFR"] = -1

df_minus1_processed = preprocess_df(df_minus1, IMPUTER_PATH_M1, SCALER_PATH_M1)

# 저장
df_minus1_processed.to_csv(SAVE_PATH_MINUS1, index=False)
print(f"[SAVE] Version B saved to: {SAVE_PATH_MINUS1}")


print("\n=== ALL DONE ===")