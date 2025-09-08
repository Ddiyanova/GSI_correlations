import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# 1. Load data

data_file = "20250723_calibration.txt"  # Raw data file
column_names = ["event", "time", "strip", "position", "energy", "channel", "beam"]
df = pd.read_csv(data_file, delim_whitespace=True, names=column_names)

# Filter for BEAM == 0
df_filtered = df[df["beam"] == 0]
print("✅ Data loaded:", len(df_filtered), "rows")


# 2. Load calibration coefficients

calibration_file = "fitted_peaks_per_strip.csv"
calib_df = pd.read_csv(calibration_file)[["Strip", "a_lin", "b_lin"]]


# 3. Merge calibration with event data

df_calibrated = df_filtered.merge(calib_df, left_on="strip", right_on="Strip", how="left")

# Compute calibrated energy for each event
df_calibrated["energy_calibrated"] = (
    df_calibrated["a_lin"] * df_calibrated["channel"] + df_calibrated["b_lin"]
)

print(
    "✅ Calibration applied. Energy range:",
    df_calibrated["energy_calibrated"].min(), "to",
    df_calibrated["energy_calibrated"].max(), "keV"
)

# 4. Define bins
strip_bins  = np.arange(0, 17)                 # 16 strips: 0..15
energy_min  = 4400                              # zoom start
energy_max  = 6400                              # zoom end
energy_bins = np.linspace(energy_min, energy_max, 200)

hist, xedges, yedges = np.histogram2d(
    df_calibrated["strip"],
    df_calibrated["energy_calibrated"],
    bins=[strip_bins, energy_bins]
)


# 5. Light pink colormap (matching your style)

light_pink_cmap = LinearSegmentedColormap.from_list(
    "light_pink_cmap",
    [
        "#4d0026",  # darkest (low counts)
        "#cc0066",  # strong pink
        "#ff66b2",  # bright light pink
        "#ffb6d9",  # pastel pink
        "#ffe6f2"   # very light pink / near-white
    ],
    N=256
)


# 6. Plot calibrated data with bigger fonts

plt.figure(figsize=(10, 6))
im = plt.imshow(
    hist.T, origin="lower", aspect="auto",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap=light_pink_cmap, vmin=0
)

# Labels & title
plt.xlabel("Strip number", fontsize=16, labelpad=10)
plt.ylabel("Energy [keV]", fontsize=16, labelpad=10)
plt.title("Calibrated Data", fontsize=18, weight="bold", pad=14)

# Tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Colorbar with larger labels/ticks
cbar = plt.colorbar(im, pad=0.02)
cbar.set_label("Counts", fontsize=16, labelpad=10)
cbar.ax.tick_params(labelsize=14)

plt.ylim(energy_min, energy_max)
plt.tight_layout()
plt.show()

print("✅ Plot generated successfully.")