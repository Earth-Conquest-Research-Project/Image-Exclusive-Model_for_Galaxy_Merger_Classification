import streamlit as st
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/classicalMachineLearning/phase3/RandomForest/RandomForest_phase3.pkl"
IMPUTER_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/preprocess_train_model/KNNImputer_Illustris.pkl"
SCALER_PATH  = "/proj/home/ibs/spaceai_2025/ai2271056/imageExclusiveModel/model/preprocess_train_model/StandardScaler_Illustris.pkl"

CLASS_LABEL = {0: "non-merger", 1: "pre-merger", 2: "post-merger"}

# ✅ UI에 보여줄 이름(대문자)
DISPLAY_FEATURES = [
    "STELLARMASS", "ABSMAG_G", "ABSMAG_R", "ABSMAG_I", "ABSMAG_Z",
    "COLOR_GR", "COLOR_GI", "SFR", "BULGEMASS",
    "VELOCITYDISPERSION", "METALLICITY", "EFFECTIVERADIUS"
]

# ✅ fit 당시(Illustris) 컬럼명으로 매핑
DISPLAY_TO_FIT = {
    "STELLARMASS": "StellarMass",
    "ABSMAG_G": "AbsMag_g",
    "ABSMAG_R": "AbsMag_r",
    "ABSMAG_I": "AbsMag_i",
    "ABSMAG_Z": "AbsMag_z",
    "COLOR_GR": "color_gr",
    "COLOR_GI": "color_gi",
    "SFR": "SFR",
    "BULGEMASS": "BulgeMass",
    "VELOCITYDISPERSION": "VelocityDispersion",
    "METALLICITY": "Metallicity",
    "EFFECTIVERADIUS": "EffectiveRadius",
}

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ✅ 전처리기가 "진짜로" 기대하는 컬럼명(순서 포함)
    fit_features = list(getattr(imputer, "feature_names_in_", DISPLAY_TO_FIT.values()))
    return model, imputer, scaler, fit_features

def preprocess_one(display_values, imputer, scaler, fit_features):
    """
    display_values: DISPLAY_FEATURES 순서의 값들
    -> fit_features 순서로 DataFrame 생성 -> imputer/scaler transform
    """
    row = {}
    for disp_name, v in zip(DISPLAY_FEATURES, display_values):
        row[DISPLAY_TO_FIT[disp_name]] = v

    # ✅ fit_features 순서로 정렬해서 DataFrame 만듦 (순서/이름 모두 확정)
    X_df = pd.DataFrame([[row.get(f, np.nan) for f in fit_features]], columns=fit_features)

    X_imp = pd.DataFrame(imputer.transform(X_df), columns=fit_features)
    X_scl = pd.DataFrame(scaler.transform(X_imp), columns=fit_features)
    return X_scl.to_numpy(dtype=float)

def predict(display_values):
    model, imputer, scaler, fit_features = load_artifacts()
    X = preprocess_one(display_values, imputer, scaler, fit_features)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return CLASS_LABEL[pred], float(proba[0]), proba, fit_features

def parse_val(s: str):
    s = (s or "").strip()
    if s == "" or s.lower() in ["na", "nan", "none", "null"]:
        return np.nan
    return float(s)

# ============================
# UI
# ============================
st.set_page_config(page_title="Galaxy Merger Predictor", page_icon="🌌", layout="centered")
st.title("🌌 Galaxy Merger Predictor")
st.caption("KNNImputer + StandardScaler (Illustris-fitted) → RandomForest 추론")

with st.expander("모델/전처리 파일 경로", expanded=False):
    st.code(f"MODEL_PATH   = {MODEL_PATH}\nIMPUTER_PATH = {IMPUTER_PATH}\nSCALER_PATH  = {SCALER_PATH}\n")

model, imputer, scaler, fit_features = load_artifacts()
with st.expander("디버그: 전처리기가 기대하는 컬럼명(feature_names_in_)", expanded=False):
    st.write(fit_features)

st.markdown("### 1) 물리량 입력")
st.info("빈칸/NA 는 결측치로 처리되고, KNNImputer가 채웁니다.", icon="ℹ️")

col1, col2 = st.columns(2, gap="large")
left_group = DISPLAY_FEATURES[:6]
right_group = DISPLAY_FEATURES[6:]

inputs = {}
with col1:
    for name in left_group:
        inputs[name] = st.text_input(name, value="")
with col2:
    for name in right_group:
        inputs[name] = st.text_input(name, value="")

st.markdown("### 2) 추론")
run = st.button("🔮 추론하기", type="primary", use_container_width=True)

if run:
    try:
        display_values = [parse_val(inputs[name]) for name in DISPLAY_FEATURES]
        label, p_nm, proba, _ = predict(display_values)

        st.success("추론 완료!", icon="✅")
        st.markdown("### 결과")
        st.metric("Predicted class", label)
        st.metric("P_NOMERGER", f"{p_nm:.4f}")

        prob_df = pd.DataFrame({
            "class": ["non-merger", "pre-merger", "post-merger"],
            "prob": [float(proba[0]), float(proba[1]), float(proba[2])]
        }).sort_values("prob", ascending=False)

        st.markdown("#### Class probabilities")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("class")["prob"])

    except ValueError as e:
        st.error(f"ValueError: {e}")
        st.info("만약 feature names mismatch가 나오면, 위 '디버그: feature_names_in_'와 매핑이 맞는지 확인하세요.")
    except FileNotFoundError as e:
        st.error(f"파일 경로를 찾지 못했어요: {e}")
    except Exception as e:
        st.error(f"에러 발생: {type(e).__name__}: {e}")
