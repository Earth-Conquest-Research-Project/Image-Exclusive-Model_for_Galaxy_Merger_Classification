import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 경로 설정
# ==========================================================
SIM_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/data/simulation_data/final_12_dataset.csv"
REAL_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/data/real_data/DESI_EHWA.csv"
SAVE_DIR = "/proj/home/ibs/spaceai_2025/ai2271056/data/distribution"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# 데이터 로드
# ==========================================================
sim_df = pd.read_csv(SIM_PATH)
real_df = pd.read_csv(REAL_PATH)

# ==========================================================
# 컬럼명 통일
# ==========================================================
rename_map = {
    "STELLARMASS": "StellarMass",
    "ABSMAG_G": "AbsMag_g",
    "ABSMAG_R": "AbsMag_r",
    "ABSMAG_I": "AbsMag_i",
    "ABSMAG_Z": "AbsMag_z",
    "COLOR_GR": "color_gr",
    "COLOR_GI": "color_gi",
    "SFR": "SFR",
    "BULGEMASS": "BulgeMass",
    "EFFECTIVERADIUS": "EffectiveRadius",
    "VELOCITYDISPERSION": "VelocityDispersion",
    "METALLICITY": "Metallicity",
}
real_df = real_df.rename(columns=rename_map)

# ==========================================================
# feature 리스트
# ==========================================================
features = [
    "StellarMass", "AbsMag_g", "AbsMag_r", "AbsMag_i", "AbsMag_z",
    "color_gr", "color_gi", "SFR", "BulgeMass", "EffectiveRadius",
    "VelocityDispersion", "Metallicity"
]

#==========================================================
# 진단 코드
#==========================================================
cols = ["StellarMass", "BulgeMass", "SFR"]

for c in cols:
    print(f"\n====== {c} ======")
    print("Simulation valid count:", sim_df[c].dropna().shape[0])
    print("Real valid count:", real_df[c].dropna().shape[0])
    print("Simulation nunique:", sim_df[c].nunique())
    print("Real nunique:", real_df[c].nunique())

#==========================================================
# P_NONMERGER
#==========================================================  

PLOT_SAVE = "/proj/home/ibs/spaceai_2025/ai2271056/data/distribution/P_NOMERGER.png"

plt.figure(figsize=(7,5))
sns.histplot(real_df["P_NOMERGER"].dropna(), kde=True, bins=40, color="purple", alpha=0.6)
plt.title("Distribution of P_NOMERGER (DESI)", fontsize=16)
plt.xlabel("P_NOMERGER", fontsize=13)
plt.ylabel("Density", fontsize=13)
plt.grid(alpha=0.3)

plt.savefig(PLOT_SAVE, dpi=300, bbox_inches="tight")
plt.close()

print(f"P_NOMERGER plot saved: {PLOT_SAVE}")


# ==========================================================
# 히스토그램 + KDE 함수
# ==========================================================
def plot_hist_kde(feature):
    sim_vals = sim_df[feature].dropna()
    real_vals = real_df[feature].dropna()

    # 데이터가 없을 경우 스킵
    if len(sim_vals) == 0 and len(real_vals) == 0:
        print(f"⚠ {feature}: both datasets empty. Skipping.")
        return
    
    plt.figure(figsize=(8, 6))

    if len(sim_vals) > 0:
        sns.histplot(sim_vals, kde=True, label="Simulation", color="blue", stat="density", alpha=0.4)
    if len(real_vals) > 0:
        sns.histplot(real_vals, kde=True, label="Real (DESI)", color="orange", stat="density", alpha=0.4)

    plt.title(f"Distribution Comparison (with KDE): {feature}", fontsize=16)
    plt.xlabel(feature, fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join(SAVE_DIR, f"{feature}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================================
# 반복 실행
# ==========================================================
for f in features:
    if f in sim_df.columns and f in real_df.columns:
        plot_hist_kde(f)
    else:
        print(f"⚠ {f} not found in one of the datasets.")

print("🎉 KDE 포함 히스토그램 저장 완료!")