import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 0) Config

file_path = "20240213_Er170_Ti50_Th215_f33.txt"
calibration_file = "fitted_peaks_per_strip.csv"  # columns: Strip,a_lin,b_lin

# Mother energy gate (keV) — used to find mothers, and later to exclude daughters in this window
mother_min = 7600
mother_max = 7880

# Spatial gate (mm)
position_tolerance = 0.5

# Time units & gate 

TIME_SCALE_TO_S = 1e-6
T_MIN_S = 0.0          # lower bound in seconds
T_MAX_S = 20.0         # upper bound in seconds

# Spectrum settings
BIN_WIDTH_KEV = 25
XMIN_KEV, XMAX_KEV = 2000, 9000   # desired x-range for plotting


# 1) Load data

column_names = ["event", "time", "strip", "position", "energy", "channel", "beam"]
df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)

# Keep only beam-off events
df = df[df["beam"] == 0].copy()
print(f"✅ Data loaded: {len(df)} beam-off events")

# Convert time to seconds (keep raw in µs for convenience)
df["time_s"] = df["time"] * TIME_SCALE_TO_S       # seconds
df["time_us"] = df["time"]                         # microseconds (as in file)


# 2) Load calibration & apply

calib_df = pd.read_csv(calibration_file, usecols=["Strip", "a_lin", "b_lin"])
df = df.merge(calib_df, left_on="strip", right_on="Strip", how="left")

missing = df["a_lin"].isna().sum()
if missing > 0:
    print(f"⚠️ {missing} rows missing calibration; dropping them.")
df = df.dropna(subset=["a_lin", "b_lin"]).copy()

# Apply linear calibration: E (keV) = a_lin * channel + b_lin
df["E_keV"] = df["a_lin"] * df["channel"] + df["b_lin"]


# 3) Build decay chains with position + time (seconds) gate

events_mother = df[df["E_keV"].between(mother_min, mother_max)]

decay_chains = []
coincidences = []  # flat table of mother→daughter pairs

for _, evt in events_mother.iterrows():
    pos0 = float(evt["position"])
    strip0 = int(evt["strip"])
    t0_s = float(evt["time_s"])

    # Candidates: same strip and within position tolerance
    mask = (
        (df["strip"] == strip0) &
        (df["position"].between(pos0 - position_tolerance, pos0 + position_tolerance))
    )
    cand = df.loc[mask, ["event", "time_s", "time_us", "position", "E_keV", "strip"]].copy()
    cand["dt_s"]  = cand["time_s"] - t0_s
    cand["dt_us"] = cand["dt_s"] * 1e6

    # Later events within the time window (in seconds)
    same_loc_later = cand[
        (cand["dt_s"] > 0.0) &
        (cand["dt_s"] >= T_MIN_S) &
        (cand["dt_s"] <= T_MAX_S)
    ].sort_values("time_s")

    if same_loc_later.empty:
        continue

    # Start chain (mother)
    chain = [{
        "role": "mother",
        "event_id": int(evt["event"]),
        "energy_keV": float(evt["E_keV"]),
        "time_s": t0_s,
        "position_mm": pos0,
        "Δt_s": 0.0,
        "Δt_us": 0.0,
        "Δx_mm_from_mother": 0.0
    }]

    for _, evt2 in same_loc_later.iterrows():
        dt_s  = float(evt2["dt_s"])
        dt_us = float(evt2["dt_us"])
        dx_mm = float(evt2["position"] - pos0)

        chain.append({
            "role": "daughter",
            "event_id": int(evt2["event"]),
            "energy_keV": float(evt2["E_keV"]),
            "time_s": float(evt2["time_s"]),
            "position_mm": float(evt2["position"]),
            "Δt_s": dt_s,
            "Δt_us": dt_us,
            "Δx_mm_from_mother": dx_mm
        })

        coincidences.append({
            "strip": strip0,
            "mother_event": int(evt["event"]),
            "mother_time_s": t0_s,
            "mother_position_mm": pos0,
            "mother_energy_keV": float(evt["E_keV"]),
            "daughter_event": int(evt2["event"]),
            "daughter_time_s": float(evt2["time_s"]),
            "daughter_position_mm": float(evt2["position"]),
            "daughter_energy_keV": float(evt2["E_keV"]),
            "delta_t_s": dt_s,
            "delta_t_us": dt_us,
            "delta_position_mm": dx_mm
        })

    decay_chains.append({
        "strip": strip0,
        "position_mm": pos0,
        "chain_length": len(chain),
        "events": chain
    })


# 4) Filter daughters & keep only single-daughter chains

if coincidences:
    coinc_df = pd.DataFrame(coincidences)

    # (a) Drop daughters inside the mother energy gate
    n_before = len(coinc_df)
    coinc_df = coinc_df[~coinc_df["daughter_energy_keV"].between(mother_min, mother_max)].copy()
    dropped_same_window = n_before - len(coinc_df)

    # (b) Keep only mothers with exactly ONE remaining daughter
    dcount = coinc_df.groupby("mother_event").size()
    keep_mothers = dcount[dcount == 1].index
    coinc_df_single = coinc_df[coinc_df["mother_event"].isin(keep_mothers)].copy()

    print(f"\n🔧 Filtering summary:")
    print(f"  - Daughters removed in mother window [{mother_min}, {mother_max}] keV: {dropped_same_window}")
    print(f"  - Mothers kept (exactly one daughter): {coinc_df_single['mother_event'].nunique()}")
    print(f"  - Surviving daughter pairs: {len(coinc_df_single)}")
else:
    coinc_df_single = pd.DataFrame(columns=[
        "strip","mother_event","mother_time_s","mother_position_mm","mother_energy_keV",
        "daughter_event","daughter_time_s","daughter_position_mm","daughter_energy_keV",
        "delta_t_s","delta_t_us","delta_position_mm"
    ])
    print("\n⚠️ No coincidences to filter.")


# 4c) Build & print filtered chains (mother + exactly one daughter)

decay_chains_filtered = []

if not coinc_df_single.empty:
    # Stable print order
    g = (coinc_df_single.sort_values(["strip", "mother_time_s", "delta_t_s"])
                      .groupby("mother_event", sort=False))

    for mother_event, grp in g:
        r = grp.iloc[0]  # exactly one row per group after filtering

        chain = {
            "strip": int(r["strip"]),
            "position_mm": float(r["mother_position_mm"]),
            "chain_length": 2,
            "events": [
                {
                    "role": "mother",
                    "event_id": int(mother_event),
                    "energy_keV": float(r["mother_energy_keV"]),
                    "time_s": float(r["mother_time_s"]),
                    "position_mm": float(r["mother_position_mm"]),
                    "Δt_s": 0.0,
                    "Δt_us": 0.0,
                    "Δx_mm_from_mother": 0.0,
                },
                {
                    "role": "daughter",
                    "event_id": int(r["daughter_event"]),
                    "energy_keV": float(r["daughter_energy_keV"]),
                    "time_s": float(r["daughter_time_s"]),
                    "position_mm": float(r["daughter_position_mm"]),
                    "Δt_s": float(r["delta_t_s"]),
                    "Δt_us": float(r["delta_t_us"]),
                    "Δx_mm_from_mother": float(r["delta_position_mm"]),
                },
            ],
        }
        decay_chains_filtered.append(chain)

    # ---- Pretty print ----
    print(f"\n✅ Filtered chains (single-daughter only): {len(decay_chains_filtered)}")
    for ch in decay_chains_filtered:
        print(f"\n--- Chain at strip {ch['strip']}, "
              f"mother position {ch['position_mm']:.3f} mm (length={ch['chain_length']}) ---")
        for ev in ch["events"]:
            print(
                f"  [{ev['role']}] id={ev['event_id']}, "
                f"E = {ev['energy_keV']:.1f} keV, "
                f"pos = {ev['position_mm']:.3f} mm, "
                f"Δt = {ev['Δt_s']:.6g} s ({ev['Δt_us']:.3g} µs), "
                f"Δx(mother) = {ev['Δx_mm_from_mother']:+.3f} mm"
            )
else:
    print("\n⚠️ No chains to print after filtering.")

# (Optional) Save filtered coincidences
if not coinc_df_single.empty:
    coinc_df_single.to_csv("coincidences_filtered.txt", sep="\t", index=False)
    print("💾 Saved filtered coincidences to 'coincidences_filtered.txt'.")


# 5) Daughters-only spectrum (filtered)

if not coinc_df_single.empty:
    # unique daughters by event id
    daughters_df = coinc_df_single[["daughter_event", "daughter_energy_keV"]].drop_duplicates()
    energies = daughters_df["daughter_energy_keV"].to_numpy()

    bins = np.arange(XMIN_KEV, XMAX_KEV + BIN_WIDTH_KEV, BIN_WIDTH_KEV)
    counts, edges = np.histogram(energies, bins=bins)

    plt.figure(figsize=(7, 5))
    plt.step(edges[:-1], counts, where="post", lw=1.5)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title("Gated Daughter Spectrum (filtered: no mother-window daughters; single-daughter chains only)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ No daughters left after filtering — nothing to plot.")
