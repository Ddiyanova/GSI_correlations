
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#inputs:
file_path = "20250723_calibration.txt"      # your calibration run
cols = ["event","time","strip","position","energy","channel","beam"]

# Rough expected peak centers (channels) for this setup
expected_means = [3413, 3633, 3841]
init_sigma = 25.0         # shared σ initial guess (channels)
halfwin_guess = 80        # look ± this around each expected mean
amp_fallback = 200.0

# Histogram binning (channels)
channel_bins = np.linspace(3000, 4500, 200)

# Model:
def gaussian(x, A, mu, s):
    return A * np.exp(-0.5*((x - mu)/s)**2)

def triple_gaussian_shared_sigma(x, A1, m1, A2, m2, A3, m3, s):
    return (gaussian(x, A1, m1, s) +
            gaussian(x, A2, m2, s) +
            gaussian(x, A3, m3, s))

# Bounds (attempt 1 then a wider retry)
lb1 = [0,   3300,  0,   3550,  0,   3800,  15]
ub1 = [1e6, 3500, 1e6,  3700, 1e6,  3900,  35]
lb2 = [0,   3200,  0,   3450,  0,   3700,  10]
ub2 = [1e6, 3600, 1e6,  3900, 1e6,  4100,  60]

# Load:
df = pd.read_csv(file_path, delim_whitespace=True, names=cols)
df = df[df["beam"] == 0].copy()

# Helper: local seed near expected mean 
def local_peak_guess(xc, yc, center, halfwin=80, amp_fb=200):
    m = (xc >= center - halfwin) & (xc <= center + halfwin)
    if not np.any(m):
        i = np.argmin(np.abs(xc - center))
        return float(xc[i]), max(float(yc[i]), amp_fb)
    subx, suby = xc[m], yc[m]
    i = int(np.argmax(suby))
    return float(subx[i]), max(float(suby[i]), amp_fb)

# Fit per strip
results = []

fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()

for strip in range(16):
    ax = axes[strip]
    ch = df.loc[df["strip"] == strip, "channel"].to_numpy()

    if ch.size < 10:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        results.append([strip, *(np.nan,)*12])  # fill NAs
        ax.set_title(f"Strip {strip}"); ax.set_xlim(3000, 4500)
        continue

    counts, edges = np.histogram(ch, bins=channel_bins)
    centers = 0.5*(edges[:-1] + edges[1:])

    m1g, A1g = local_peak_guess(centers, counts, expected_means[0], halfwin_guess, amp_fallback)
    m2g, A2g = local_peak_guess(centers, counts, expected_means[1], halfwin_guess, amp_fallback)
    m3g, A3g = local_peak_guess(centers, counts, expected_means[2], halfwin_guess, amp_fallback)
    p0 = [A1g, m1g, A2g, m2g, A3g, m3g, init_sigma]

    fitted = False
    for LB, UB, fev in [(lb1, ub1, 20000), (lb2, ub2, 40000)]:
        try:
            popt, pcov = curve_fit(triple_gaussian_shared_sigma, centers, counts,
                                   p0=p0, bounds=(LB, UB), maxfev=fev)
            fitted = True
            break
        except Exception:
            p0[-1] = max(p0[-1]*1.2, LB[-1])  # bump σ and retry

    if not fitted:
        ax.bar(centers, counts, width=np.diff(edges), color="lightblue", alpha=0.6)
        ax.text(0.5, 0.5, "Fit failed", ha="center", va="center", transform=ax.transAxes)
        results.append([strip, *(np.nan,)*12])
        ax.set_title(f"Strip {strip}"); ax.set_xlim(3000, 4500)
        continue

    A1, m1, A2, m2, A3, m3, s = [float(v) for v in popt]
    means_sorted = sorted([m1, m2, m3])
    sigma_shared = float(s)
    hwhm = float(np.sqrt(2*np.log(2)) * sigma_shared)
    fwhm = 2*hwhm

    # Linear & Quadratic calibration from the three peak positions
    channels = np.array(means_sorted, dtype=float)
    energies  = np.array([5157.0, 5486.0, 5805.0], dtype=float)

    a_lin,  b_lin              = np.polyfit(channels, energies, 1)
    a_quad, b_quad, c_quad     = np.polyfit(channels, energies, 2)
    E_lin_fit  = a_lin*channels + b_lin
    calib_rms_lin = float(np.sqrt(np.mean((E_lin_fit - energies)**2)))

    results.append([strip,
                    means_sorted[0], means_sorted[1], means_sorted[2],
                    a_lin, b_lin, a_quad, b_quad, c_quad,
                    sigma_shared, hwhm, fwhm, calib_rms_lin])

    # Plot
    ax.bar(centers, counts, width=np.diff(edges), color="lightpink", alpha=0.6)
    ax.plot(centers, triple_gaussian_shared_sigma(centers, *popt), color="#C71585")
    for pk in means_sorted:
        ax.axvline(pk, ls="--", color="#CF9EB7", alpha=0.8)
    ax.text(0.04, 0.96, f"σ={sigma_shared:.1f}\nFWHM={fwhm:.1f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.6))
    ax.set_title(f"Strip {strip}"); ax.set_xlim(3000, 4500)
    if strip % 4 != 0: ax.set_ylabel("")
    if strip < 12:     ax.set_xlabel("")

fig.text(0.5, 0.04, "Channel", ha="center", fontsize=14)
fig.text(0.04, 0.5, "Counts", va="center", rotation="vertical", fontsize=14)
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.suptitle("Triple-Gaussian per Strip (shared σ) — calibration", fontsize=16, y=1.02)
plt.show()

# Save CSV: 
df_out = pd.DataFrame(results, columns=[
    "Strip",
    "Peak1_channel","Peak2_channel","Peak3_channel",
    "a_lin","b_lin","a_quad","b_quad","c_quad",
    "sigma_shared","HWHM_shared","FWHM_shared","calib_rms_lin_keV"
])
df_out.to_csv("fitted_peaks_per_strip.csv", index=False)
print("✅ Saved fitted_peaks_per_strip.csv (linear + quadratic + widths)")
