import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------
# 1) Load data
# ---------------------------
file_path = "20240213_Er170_Ti50_Th215_f33.txt"
column_names = ["event", "time", "strip", "position", "energy", "channel", "beam"]
df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)

# Keep only BEAM == 0
df = df[df["beam"] == 0].copy()
print("✅ Data loaded:", len(df), "rows with BEAM==0")

# ---------------------------
# 2) Calibration
# ---------------------------
calibration_file = "fitted_peaks_per_strip.csv"
calib_df = pd.read_csv(calibration_file, usecols=["Strip", "a_lin", "b_lin"])
df = df.merge(calib_df, left_on="strip", right_on="Strip", how="left")

missing = df["a_lin"].isna().sum()
if missing:
    print(f"⚠️ {missing} rows missing calibration; dropping them.")
df = df.dropna(subset=["a_lin", "b_lin"]).copy()

df["energy_calibrated"] = df["a_lin"] * df["channel"] + df["b_lin"]

# ---------------------------
# 3) Gates
# ---------------------------
MOTHER_E = 7500
MOTHER_WIN = 500
DAU_E = 6755
DAU_WIN = 150

POS_TOL = 0.5  # mm
T_MIN_S = 0
T_MAX_S = 100
T_MIN_US = T_MIN_S * 1e6
T_MAX_US = T_MAX_S * 1e6

# ---------------------------
# 4) Select mothers
# ---------------------------
mothers = df[
    df["energy_calibrated"].between(MOTHER_E - MOTHER_WIN, MOTHER_E + MOTHER_WIN)
].copy()

pairs = []
multi_daughter_list = []

for _, mom in mothers.iterrows():
    strip = mom["strip"]
    pos0 = mom["position"]
    t0 = mom["time"]

    # Candidate daughters
    daughters = df[
        (df["strip"] == strip) &
        (np.abs(df["position"] - pos0) <= POS_TOL) &
        (df["time"] >= t0 + T_MIN_US) &
        (df["time"] <= t0 + T_MAX_US) &
        (df["energy_calibrated"].between(DAU_E - DAU_WIN, DAU_E + DAU_WIN))
    ].copy()

    if len(daughters) == 1:
        dau = daughters.iloc[0]
        pairs.append({
            "mother_event": int(mom["event"]),
            "mother_strip": int(mom["strip"]),
            "mother_pos_mm": float(mom["position"]),
            "mother_E_keV": float(mom["energy_calibrated"]),
            "mother_time_s": t0 / 1e6,
            "daughter_event": int(dau["event"]),
            "daughter_strip": int(dau["strip"]),
            "daughter_pos_mm": float(dau["position"]),
            "daughter_E_keV": float(dau["energy_calibrated"]),
            "daughter_time_s": dau["time"] / 1e6,
            "Δt_s": (dau["time"] - t0) / 1e6
        })
    elif len(daughters) > 1:
        multi_daughter_list.append({
            "mother_event": int(mom["event"]),
            "mother_strip": int(mom["strip"]),
            "mother_pos_mm": float(mom["position"]),
            "mother_E_keV": float(mom["energy_calibrated"]),
            "mother_time_s": t0 / 1e6,
            "n_daughters": len(daughters)
        })

pairs_df = pd.DataFrame(pairs)
multi_df = pd.DataFrame(multi_daughter_list)

print(f"✅ Found {len(pairs_df)} mother–daughter pairs (exactly 1 daughter)")
print(f"ℹ️ Mothers with multiple daughters: {len(multi_df)}")

# ---------------------------
# 5) Print results
# ---------------------------
if not pairs_df.empty:
    print("\n🔎 Mother–Daughter pairs (1 daughter):")
    print(pairs_df.to_string(index=False))

if not multi_df.empty:
    print("\n⚠️ Mothers with multiple daughters:")
    print(multi_df.to_string(index=False))

# ---------------------------
# 6) Δt histogram + fit
# ---------------------------
def exp_plus_const(t, A, tau, C):
    return A * np.exp(-t / tau) + C

if not pairs_df.empty:
    dts = pairs_df["Δt_s"].to_numpy()

    hist_range = (0, T_MAX_S)
    bins = 50
    hist, edges = np.histogram(dts, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p0 = [hist.max(), 50, min(hist)]
    try:
        popt, pcov = curve_fit(exp_plus_const, centers, hist, p0=p0,
                               bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        A_fit, tau_fit, C_fit = popt
        tau_err = np.sqrt(np.diag(pcov))[1]
        print(f"\n📏 Lifetime fit: τ = {tau_fit:.2f} ± {tau_err:.2f} s")
    except Exception as e:
        print(f"⚠️ Fit failed: {e}")
        popt = None

        plt.figure(figsize=(8,5))
    # Histogram in light pink
    plt.hist(dts, bins=bins, range=hist_range,
             edgecolor='black', color="#ffb6c1", label="Data")
    if popt is not None:
        # Fit curve in deep/hot pink
        t_fit = np.linspace(0, T_MAX_S, 500)
        plt.plot(t_fit, exp_plus_const(t_fit, *popt),
                 color="#ff1493", lw=2, label=f"Fit: τ = {tau_fit:.1f} s")
    plt.xlabel("Δt (s) [daughter − mother]")
    plt.ylabel("Counts")
    plt.title("Mother → Daughter Δt Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()