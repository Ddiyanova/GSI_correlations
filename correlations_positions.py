import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# 1) Load coincidences (your TSV/whitespace file)
# -----------------------------
coinc_file = "coincidences_filtered.txt"   # change if needed
coinc_df = pd.read_csv(coinc_file, sep=r"\s+", engine="python")
if coinc_df.empty:
    raise ValueError("No coincidences found in file.")

coinc_df.columns = [c.strip() for c in coinc_df.columns]

# Reconstruct daughter_position_mm if not present
if "daughter_position_mm" not in coinc_df.columns:
    if "delta_position_mm" in coinc_df.columns:
        coinc_df["daughter_position_mm"] = coinc_df["mother_position_mm"] + coinc_df["delta_position_mm"]
    else:
        raise ValueError("Need 'daughter_position_mm' or 'delta_position_mm' to plot daughters.")

# Convert & clean
for c in ["strip", "mother_position_mm", "daughter_position_mm"]:
    coinc_df[c] = pd.to_numeric(coinc_df[c], errors="coerce")
coinc_df = coinc_df.dropna(subset=["strip", "mother_position_mm", "daughter_position_mm"])

# Normalize strip indexing to 0–15 if file is 1–16
smin, smax = coinc_df["strip"].min(), coinc_df["strip"].max()
if smin == 1 and smax == 16:
    coinc_df["strip"] = coinc_df["strip"] - 1

# -----------------------------
# 2) Detector geometry & styles
# -----------------------------
num_strips   = 16
strip_length = 35.0   # mm (y)
strip_width  = 5.0    # mm (x)

inactive_strips = {0, 11}            # <<< mark these as non-active (light pink)
inactive_color  = "#FFE6F0"          # very light pink
active_color    = "lightgrey"

mother_color    = "#FF4D94"          # darker pink
daughter_color  = "#FF9ECF"          # lighter pink
mother_size     = 36                 # smaller than before
daughter_size   = 20                 # smaller than before

def strip_center_x(s): return s * strip_width + strip_width / 2.0

# Clip positions to detector range
coinc_df["mother_position_mm"]   = coinc_df["mother_position_mm"].clip(0, strip_length)
coinc_df["daughter_position_mm"] = coinc_df["daughter_position_mm"].clip(0, strip_length)

# -----------------------------
# 3) Draw
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Strips
for s in range(num_strips):
    x0 = s * strip_width
    face = inactive_color if s in inactive_strips else active_color
    ax.add_patch(
        patches.Rectangle((x0, 0), strip_width, strip_length,
                          linewidth=1, edgecolor='black', facecolor=face,
                          alpha=0.8 if s in inactive_strips else 0.35)
    )

# Coincidences
for _, row in coinc_df.iterrows():
    s  = int(row["strip"])
    ym = float(row["mother_position_mm"])
    yd = float(row["daughter_position_mm"])
    x  = strip_center_x(s)

    # connector
    ax.plot([x, x], [ym, yd], lw=1.0, color="tab:gray", alpha=0.7, zorder=2)

    # mother (star) + daughter (dot), both pink but distinct + smaller
    ax.scatter(x, ym, marker='*', s=mother_size, color=mother_color,
               edgecolor='black', linewidth=0.5, zorder=3)
    ax.scatter(x, yd, marker='o', s=daughter_size, color=daughter_color,
               edgecolor='black', linewidth=0.4, zorder=3)

# -----------------------------
# 4) Axes & legend
# -----------------------------
ax.set_xlim(0, strip_width * num_strips)
ax.set_ylim(0, strip_length)

centers = [strip_center_x(s) for s in range(num_strips)]
ax.set_xticks(centers)
ax.set_xticklabels([str(s) for s in range(num_strips)])
ax.set_xlabel("Strip index")
ax.set_ylabel("Position along strip (mm)")
ax.set_title("α–α Coincidence Locations", fontsize=14, weight='bold', pad=12)

ax.set_aspect('equal', adjustable='box')
ax.grid(True, axis='y', alpha=0.25, linestyle='--')

# Legend
mother_handle = plt.Line2D([0], [0], marker='*', color='w', label='Mother',
                           markerfacecolor=mother_color, markeredgecolor='black', markersize=8)
daughter_handle = plt.Line2D([0], [0], marker='o', color='w', label='Daughter',
                             markerfacecolor=daughter_color, markeredgecolor='black', markersize=6)
inactive_patch = patches.Patch(facecolor=inactive_color, edgecolor='black', label='Inactive strip')
ax.legend(handles=[mother_handle, daughter_handle, inactive_patch], loc="upper right", frameon=True)

plt.tight_layout()
plt.show()
