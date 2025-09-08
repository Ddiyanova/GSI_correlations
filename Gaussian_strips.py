import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# --------------------------
# Load data
# --------------------------
file_path = "20250723_calibration.txt"  # Update path if needed
column_names = ["event", "time", "strip", "position", "energy", "channel", "beam"]
df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)
df_filtered = df[df["beam"] == 0]

# Histogram bins
channel_bins = np.linspace(3000, 4500, 200)

# Gaussian pieces
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# NEW: triple Gaussian with a *shared* sigma
def triple_gaussian_shared_sigma(x, a1, m1, a2, m2, a3, m3, s):
    return (gaussian(x, a1, m1, s) +
            gaussian(x, a2, m2, s) +
            gaussian(x, a3, m3, s))

# -----------------------------
# User-defined guesses and bounds
# -----------------------------
initial_means  = [3413, 3630, 3840]  # expected peak centers
initial_sigma_shared = 25            # one shared sigma initial guess
initial_amps   = [250, 250, 250]     # expected peak heights

# Bounds for params: [a1, m1, a2, m2, a3, m3, s]
lower_bounds = [0,   3300,   0,   3550,   0,   3750,  15]
upper_bounds = [1e6, 3500, 1e6,  3700,  1e6,  3900,  35]

# Known energies for calibration (keV)
known_energies = [5157, 5486, 5805]

# Store results
results = []

# -----------------------------
# Create 4x4 subplot figure
# -----------------------------
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()

for strip in range(16):
    data_strip = df_filtered[df_filtered["strip"] == strip]["channel"].to_numpy()
    ax = axes[strip]

    # Handle empty strips
    if data_strip.size < 1:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center',
                fontsize=10, transform=ax.transAxes)
        results.append([strip, np.nan, np.nan, np.nan,  # peaks
                        np.nan, np.nan, np.nan,        # a_lin, b_lin, sigma_shared
                        np.nan, np.nan, np.nan])       # HWHM, FWHM, calib quality flag (optional)
        ax.set_title(f"Strip {strip}", fontsize=10)
        ax.set_xlim(3000, 4500)
        continue

    # Histogram
    counts, bin_edges = np.histogram(data_strip, bins=channel_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Build initial guesses: [a1, m1, a2, m2, a3, m3, s]
    guess = []
    for mean_guess, amp_guess in zip(initial_means, initial_amps):
        idx = np.argmin(np.abs(bin_centers - mean_guess))
        a0 = max(counts[idx], amp_guess)
        guess.extend([a0, mean_guess])
    guess.append(initial_sigma_shared)  # shared sigma at the end

    # Clamp guesses into bounds
    guess = np.maximum(guess, lower_bounds)
    guess = np.minimum(guess, upper_bounds)

    # Fit triple with shared sigma
    try:
        popt, pcov = curve_fit(
            triple_gaussian_shared_sigma,
            bin_centers, counts,
            p0=guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )
        # Unpack parameters
        a1, m1, a2, m2, a3, m3, s_shared = [float(v) for v in popt]
        # Sort peaks by mean (for mapping to known energies)
        means_sorted = sorted([m1, m2, m3])
        sigma_shared = float(s_shared)

        # Convert shared sigma to HWHM/FWHM
        hwhm = np.sqrt(2.0*np.log(2.0)) * sigma_shared
        fwhm = 2.0 * hwhm

        # Linear calibration: E = a*Ch + b
        channels = np.array(means_sorted, dtype=float)
        energies = np.array(known_energies, dtype=float)
        a_lin, b_lin = np.polyfit(channels, energies, 1)

        # (Optional) quick calib quality: rms of energy residuals after linear fit
        E_fit = a_lin*channels + b_lin
        calib_rms = float(np.sqrt(np.mean((E_fit - energies)**2)))

        # Save
        results.append([strip,
                        means_sorted[0], means_sorted[1], means_sorted[2],
                        a_lin, b_lin, sigma_shared,
                        hwhm, fwhm, calib_rms])

        # Plot histogram and fit
        ax.bar(bin_centers, counts, width=np.diff(bin_edges), color='lightblue', alpha=0.6)
        ax.plot(bin_centers, triple_gaussian_shared_sigma(bin_centers, *popt), 'r-', label="Fit")

        for peak in means_sorted:
            ax.axvline(peak, linestyle='--', color='blue', alpha=0.8)

        ax.text(0.04, 0.96,
                f"σ={sigma_shared:.1f}\nFWHM={fwhm:.1f}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.6))

    except Exception as e:
        # Fit failed
        ax.bar(bin_centers, counts, width=np.diff(bin_edges), color='lightblue', alpha=0.6)
        ax.text(0.5, 0.5, "Fit failed", ha='center', va='center',
                fontsize=10, transform=ax.transAxes)
        results.append([strip, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan])

    ax.set_title(f"Strip {strip}", fontsize=10)
    ax.set_xlim(3000, 4500)

    # Hide repetitive labels for a cleaner grid
    if strip % 4 != 0:
        ax.set_ylabel("")
    if strip < 12:
        ax.set_xlabel("")

# Shared labels
fig.text(0.5, 0.04, "Channel", ha='center', fontsize=14)
fig.text(0.04, 0.5, "Counts", va='center', rotation='vertical', fontsize=14)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.suptitle("Triple-Gaussian Fit per Strip with Shared σ (common HWHM/FWHM)", fontsize=16, y=1.02)
plt.show()

# -----------------------------
# Save results to CSV
# -----------------------------
df_results = pd.DataFrame(results, columns=[
    "Strip",
    "Peak1_channel", "Peak2_channel", "Peak3_channel",
    "a_lin", "b_lin", "sigma_shared",
    "HWHM_shared", "FWHM_shared", "calib_rms_keV"
])
df_results.to_csv("fitted_peaks_per_strip.csv", index=False)
print("✅ Saved fitted_peaks_per_strip.csv (shared σ → shared HWHM/FWHM per strip)")