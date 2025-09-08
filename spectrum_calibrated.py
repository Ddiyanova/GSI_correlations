import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------
# 1) Load data
# ---------------------------
file_path = "20240213_Er170_Ti50_Th215_f33.txt"  # raw data
column_names = ["event", "time", "strip", "position", "energy", "channel", "beam"]
df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)

# Keep only beam-off events
df = df[df["beam"] == 0].copy()
print(f"✅ Data loaded: {len(df)} beam-off events")

# ---------------------------
# 2) Load calibration & apply
# ---------------------------
calibration_file = "fitted_peaks_per_strip.csv"
calib_df = pd.read_csv(calibration_file, usecols=["Strip", "a_lin", "b_lin"])
df = df.merge(calib_df, left_on="strip", right_on="Strip", how="left")
df = df.dropna(subset=["a_lin", "b_lin"]).copy()

# Apply linear calibration
df["E_keV"] = df["a_lin"] * df["channel"] + df["b_lin"]

# ---------------------------
# 3) Histogram
# ---------------------------
bin_width_keV = 20
emin, emax = 5570, 8000
bins = np.arange(emin, emax + bin_width_keV, bin_width_keV)
counts, edges = np.histogram(df["E_keV"], bins=bins)

centers = 0.5 * (edges[:-1] + edges[1:])
y_at = interp1d(centers, counts, kind="nearest", bounds_error=False, fill_value=0.0)

# ---------------------------
# 4) Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.step(edges[:-1], counts, where="post", lw=1.8, color="#FF69B4")
ax.set_xlabel("Energy (keV)", fontsize=18, labelpad=12, color="black")
ax.set_ylabel("Counts", fontsize=18, labelpad=12, color="black")
ax.set_title("Calibrated Energy Spectrum", fontsize=20, weight="bold", pad=16, color="black")
ax.tick_params(axis='both', labelsize=16, colors="black")
ax.grid(True, alpha=0.3, color="#FFC0CB")

# ---------------------------
# 5) Peaks + legend info
# ---------------------------
labels = [
    {"E": 6000, "y": 116, "name": r"$^{207}$Rn+$^{208}$Rn", "t12": r"$t_{1/2}(^{207}\mathrm{Rn})=9.25\,\mathrm{m},\ t_{1/2}(^{208}\mathrm{Rn})=24.35\,\mathrm{m}$"},
    {"E": 6367, "y":  50, "name": r"$^{211}$Fr",             "t12": r"$t_{1/2}(^{211}Fr)=3.1\ \mathrm{m}$"},
    {"E": 6720, "y": 242, "name": r"$^{211}$Ra+$^{212}$Ra",  "t12": r"$t_{1/2}(^{211}\mathrm{Ra})=13\ \mathrm{s},\ t_{1/2}(^{212}\mathrm{Ra})=13\ \mathrm{s}$"},
    {"E": 7228, "y":  40, "name": r"$^{215}$Th",             "t12": r"$t_{1/2}(^{215}Th)=1.2\ \mathrm{s}$"},
    {"E": 7391, "y":  77, "name": r"$^{215}$Ac",             "t12": r"$t_{1/2}(^{215}Ac)=0.17\ \mathrm{s}$"},
    {"E": 7734, "y":  77, "name": r"$^{216}$Th",             "t12": r"$t_{1/2}(^{216}Th)=26\ \mathrm{ms}$"},
]

legend_entries = [lab['t12'] for lab in labels]  # only half-life info

for lab in labels:
    E = lab["E"]
    y_tip = lab.get("y", float(y_at(E)))
    ax.annotate(
        lab["name"],
        xy=(E, y_tip), xycoords="data",
        xytext=(0, 80), textcoords="offset pixels",
        ha="center", va="bottom", fontsize=14, color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="#ff99c8"),
        arrowprops=dict(arrowstyle="->", lw=1.6, color="#C71585", shrinkA=0, shrinkB=0, mutation_scale=12)
    )

# ---------------------------
# 6) Add legend with only lifetimes
# ---------------------------
legend_text = "\n".join(legend_entries)
ax.text(
    0.98, 0.98, legend_text,
    transform=ax.transAxes,
    fontsize=12, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ff99c8")
)
# ---------------------------
# 7) Finalize
# ---------------------------
ax.set_ylim(0, max(counts) * 1.35 + 20)
plt.tight_layout()
plt.show()