"""
Multi-dimensional multiplicative-noise SDE with staggered collapse thresholds (mu <= zeta^2/2).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "grid.alpha": 0.4,
})

NUM_REALIZATIONS = 500
STEPS = 5000
dt = 0.01

# mu = 2.5 stays active (mu > zeta^2/2 = 1.28); mu = 1.2 and mu = 0.5 collapse to zero.
mu = np.array([2.5, 1.2, 0.5])
zeta = 1.6
D = len(mu)

trajectories = np.zeros((NUM_REALIZATIONS, STEPS, D))
np.random.seed(42)
trajectories[:, 0, :] = np.random.normal(loc=1.5, scale=0.2, size=(NUM_REALIZATIONS, D))

for t in range(1, STEPS):
    theta = trajectories[:, t - 1, :]
    drift = -(theta ** 3 - theta * mu)
    dW = np.random.normal(0, np.sqrt(dt), size=(NUM_REALIZATIONS, D))
    diffusion = zeta * theta * dW
    trajectories[:, t, :] = theta + drift * dt + diffusion

order_param = np.linalg.norm(trajectories, axis=2)
mean_O = np.mean(order_param, axis=0)
var_O = np.var(order_param, axis=0)
Rv_raw = var_O / (mean_O ** 2 + 1e-8)
Rv_smooth = gaussian_filter1d(Rv_raw, sigma=40.0)

time_x = np.arange(STEPS) * dt

peaks_idx, properties = find_peaks(Rv_smooth, prominence=np.max(Rv_smooth) * 0.01, distance=200)
early_mask = time_x[peaks_idx] < 15.0
early_peaks_idx = peaks_idx[early_mask]
early_prominences = properties["prominences"][early_mask]
if len(early_peaks_idx) > 2:
    top_indices = np.argsort(early_prominences)[-2:]
    precursory_peaks = np.sort(early_peaks_idx[top_indices])
else:
    precursory_peaks = early_peaks_idx

c_main, c_peak = "#1f77b4", "#d62728"
fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_x, mean_O, lw=3, color=c_main)
ax1.set_title(r"SDE: Mean Distance from Origin $\langle \mathcal{O}(t) \rangle$")
ax1.set_xlabel("Time $t$")
ax1.grid(True, linestyle=":")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_x, Rv_raw, color="#8c564b", alpha=0.3, lw=1.0, label="Raw Variance")
ax2.plot(time_x, Rv_smooth, color="#8c564b", lw=2.5, label=r"Smoothed $\mathcal{R}_v(t)$")
for i, idx in enumerate(precursory_peaks):
    tp = time_x[idx]
    ax2.scatter(tp, Rv_smooth[idx], color=c_peak, s=90, zorder=5)
    ax2.annotate(f"$p_{i+1}$", (tp, Rv_smooth[idx]), textcoords="offset points",
                 xytext=(0, 10), ha="center", color=c_peak, fontweight="bold", fontsize=12)
ax2.set_title(r"SDE: Sequential Collapse Peaks $\mathcal{R}_v(t)$")
ax2.set_xlabel("Time $t$")
ax2.grid(True, linestyle=":")
ax2.legend(loc="lower right")

ax3 = fig.add_subplot(gs[0, 2])
for i in range(D):
    ax3.plot(time_x, trajectories[0, :, i], alpha=0.85, lw=1.5, label=rf"Dim {i + 1} ($\mu={mu[i]}$)")
ax3.set_title("Single Particle Trajectory")
ax3.set_xlabel("Time $t$")
ax3.grid(True, linestyle=":")
ax3.legend(loc="upper right")

fig.suptitle("Experiment 4: SDE Simulation with Staggered Multiplicative Collapse",
             y=1.02, fontsize=15, fontweight="bold")
plt.savefig("exp4_sde_multiplicative_smoothed.png", bbox_inches="tight", dpi=300)
plt.show()
