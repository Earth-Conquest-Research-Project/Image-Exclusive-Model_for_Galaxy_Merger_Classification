import pandas as pd

# 경로 수정해서 사용
infer_path = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/inference/randomforest_final12_inference.csv"
raw_path   = "/proj/home/ibs/spaceai_2025/ai2271056/SYNERGI/data/DESI/DESI_raw.csv"

# CSV 로드
infer_df = pd.read_csv(infer_path)
raw_df   = pd.read_csv(raw_path)

# 인덱스 유지 (매칭을 위해)
infer_df = infer_df.reset_index().rename(columns={"index": "data_index"})
raw_df   = raw_df.reset_index().rename(columns={"index": "data_index"})

# 조건별 row 선택
row_0 = infer_df[infer_df["pred_class"] == 0].sort_values("P_NOMERGER", ascending=False).head(1)
row_1 = infer_df[infer_df["pred_class"] == 1].sort_values("P_NOMERGER", ascending=True).head(1)
row_2 = infer_df[infer_df["pred_class"] == 2].sort_values("P_NOMERGER", ascending=True).head(1)

# 선택된 index만 따기
indices = pd.concat([row_0["data_index"], row_1["data_index"], row_2["data_index"]]).tolist()

# 원본 raw 데이터에서 동일 index row 가져오기
raw_selected = raw_df[raw_df["data_index"].isin(indices)]

# inference 결과 + raw 정보 merge
merged = pd.merge(
    infer_df[infer_df["data_index"].isin(indices)],
    raw_selected,
    on="data_index",
    how="inner",
    suffixes=("_infer", "_raw")
)

print("===== 🔥 최종 선택된 3개의 row (inference + raw 정보) =====")
print(merged.to_string())