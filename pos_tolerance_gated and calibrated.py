import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1) Load data
# -----------------------------
file_path = "20240213_Er170_Ti50_Th215_f33.txt"
column_names = ["event", "time", "strip", "position", "energy", "channel", "beam"]
df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)

# Keep only BEAM == 0
df = df[df["beam"] == 0].copy()
print(f"✅ Data loaded: {len(df)} rows with BEAM==0")

# -----------------------------
# 2) Load calibration and apply
# -----------------------------
calibration_file = "fitted_peaks_per_strip.csv"
calib_df = pd.read_csv(calibration_file, usecols=["Strip","a_lin","b_lin"])

df = df.merge(calib_df, left_on="strip", right_on="Strip", how="left")
missing = int(df["a_lin"].isna().sum())
if missing:
    print(f"⚠️ {missing} rows missing calibration coefficients; dropping them.")
df = df.dropna(subset=["a_lin","b_lin"]).copy()

df["energy_calibrated"] = df["a_lin"] * df["channel"] + df["b_lin"]

# -----------------------------
# 3) Parameters
# -----------------------------
mother_energy_keV   = 7700.0
mother_window_keV   = 100.0
daughter_energy_keV = 6755.0
daughter_window_keV = 150.0

pos_tol_mm = 2  # position tolerance in mm
T_MIN_S    = 0        # min time diff in seconds
T_MAX_S    = 20      # max time diff in seconds
T_MIN_US   = T_MIN_S * 1e6
T_MAX_US   = T_MAX_S * 1e6

print(f"Mother gate:   {mother_energy_keV:.0f} ± {mother_window_keV:.0f} keV")
print(f"Daughter gate: {daughter_energy_keV:.0f} ± {daughter_window_keV:.0f} keV")
print(f"Position gate: |Δx| ≤ {pos_tol_mm:.1f} mm")
print(f"Time gate:     {T_MIN_S}–{T_MAX_S} s\n")

# -----------------------------
# 4) Select mothers
# -----------------------------
mothers = df[
    df["energy_calibrated"].between(mother_energy_keV - mother_window_keV,
                                    mother_energy_keV + mother_window_keV)
].copy()
print(f"✅ Mothers in gate: {len(mothers)}")

# -----------------------------
# 5) Find exactly one daughter per mother
# -----------------------------
delta_positions = []
chains = []
rejected_zero  = 0
rejected_multi = 0

# Pre-filter daughters by energy gate
daughters_all = df[
    df["energy_calibrated"].between(daughter_energy_keV - daughter_window_keV,
                                    daughter_energy_keV + daughter_window_keV)
][["event","time","strip","position","energy_calibrated"]].copy()

# Group daughters by strip for fast search
d_by_strip = {s: g.sort_values("time").reset_index(drop=True)
              for s, g in daughters_all.groupby("strip")}

for _, mom in mothers.iterrows():
    s    = int(mom["strip"])
    t0   = float(mom["time"])
    x0   = float(mom["position"])
    E0   = float(mom["energy_calibrated"])

    g = d_by_strip.get(s)
    if g is None or g.empty:
        rejected_zero += 1
        continue

    # Apply time and position gate
    cand = g[
        (g["time"] >= t0 + T_MIN_US) &
        (g["time"] <= t0 + T_MAX_US) &
        (np.abs(g["position"] - x0) <= pos_tol_mm)
    ]

    n = len(cand)
    if n == 1:
        dau = cand.iloc[0]
        dx  = float(dau["position"]) - x0
        dt_s = (float(dau["time"]) - t0) / 1e6
        chains.append({
            "mother_event": int(mom["event"]),
            "mother_strip": s,
            "mother_pos_mm": x0,
            "mother_E_keV": E0,
            "mother_time_s": t0 / 1e6,
            "daughter_event": int(dau["event"]),
            "daughter_pos_mm": float(dau["position"]),
            "daughter_E_keV": float(dau["energy_calibrated"]),
            "daughter_time_s": float(dau["time"]) / 1e6,
            "Δt_s": dt_s,
            "Δx_mm": dx
        })
        delta_positions.append(dx)
    elif n == 0:
        rejected_zero += 1
    else:
        rejected_multi += 1

# -----------------------------
# 6) Print results
# -----------------------------
print(f"\n✅ Accepted pairs (exactly one daughter): {len(chains)}")
print(f"   Rejected (0 daughters):  {rejected_zero}")
print(f"   Rejected (>1 daughters): {rejected_multi}")

if chains:
    df_pairs = pd.DataFrame(chains).sort_values(["mother_strip","mother_time_s"])
    print("\n🔎 Mother–Daughter pairs:")
    print(df_pairs.to_string(index=False))

# -----------------------------
# 7) Fit Gaussian to Δposition + clean legend
# -----------------------------
def gaussian_const(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C

delta_positions = np.asarray(delta_positions, dtype=float)

if delta_positions.size > 0:
    n_bins = 50
    hist, edges = np.histogram(delta_positions, bins=n_bins)
    centers = 0.5*(edges[:-1] + edges[1:])

    A0 = max(hist.max() - np.median(hist), 1.0)
    mu0 = 0.0
    sigma0 = max(np.std(delta_positions), 0.8)
    C0 = max(float(np.min(hist)), 0.0)

    try:
        popt, pcov = curve_fit(
            gaussian_const, centers, hist,
            p0=[A0, mu0, sigma0, C0],
            bounds=([0, centers.min(), 1e-6, 0],
                    [np.inf, centers.max(), np.inf, np.inf]),
            maxfev=20000
        )
        mu_fit, sigma_fit = float(popt[1]), float(popt[2])

        # Print to terminal
        print(f"μ (mean)   = {mu_fit:.3f} mm")
        print(f"σ (sigma) = {sigma_fit:.3f} mm")

        xfit = np.linspace(centers.min(), centers.max(), 800)
        yfit = gaussian_const(xfit, *popt)
    except Exception as e:
        print(f"\n⚠️ Fit did not converge: {e}")
        xfit, yfit = None, None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_positions, bins=n_bins, edgecolor='black', color="#ffb6c1", label="Data")
    if xfit is not None:
        ax.plot(xfit, yfit, color="#ff1493", lw=2.2, label="Fit")

    ax.set_xlabel("Δposition Δx (mm)", fontsize=18, labelpad=10, color="black")
    ax.set_ylabel("Counts", fontsize=18, labelpad=10, color="black")
    ax.set_title("Δposition for accepted pairs", fontsize=20, weight="bold", pad=12, color="black")
    ax.tick_params(axis='both', labelsize=16, colors="black")
    ax.grid(True, alpha=0.3)

    ax.legend(loc="upper right", fontsize=14, frameon=True)
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ No Δposition samples to fit/plot.")