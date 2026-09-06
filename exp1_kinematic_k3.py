"""
Idealized kinematic K=3 verification of the theoretical lambda = 2 DSI baseline (Appendix D.3).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "grid.alpha": 0.4,
})

# ==============================================================================
# Synthetic theoretical trajectories
# ==============================================================================
TOTAL_STEPS = 2000
t = np.arange(TOTAL_STEPS)

T_MERGE_1 = 700   # first microtransition (parameters 1, 2 merge)
T_MERGE_2 = 1400  # macroscopic collapse (parameter 3 merges)

w1 = np.zeros(TOTAL_STEPS)
w2 = np.zeros(TOTAL_STEPS)
w3 = np.zeros(TOTAL_STEPS)

t_p1 = np.arange(0, T_MERGE_1)
w1[:T_MERGE_1] = PchipInterpolator([0, T_MERGE_1 / 2, T_MERGE_1], [2.2, 1.5, 1.0])(t_p1)
w2[:T_MERGE_1] = PchipInterpolator([0, T_MERGE_1 / 2, T_MERGE_1], [0.3, 0.7, 1.0])(t_p1)
w3[:T_MERGE_1] = PchipInterpolator([0, T_MERGE_1 / 2, T_MERGE_1], [-0.5, -0.3, -0.1])(t_p1)

t_p2 = np.arange(T_MERGE_1, T_MERGE_2)
trunk_1 = PchipInterpolator([T_MERGE_1, T_MERGE_1 + 350, T_MERGE_2], [1.0, 0.7, 0.4])(t_p2)
w1[T_MERGE_1:T_MERGE_2] = trunk_1
w2[T_MERGE_1:T_MERGE_2] = trunk_1
w3[T_MERGE_1:T_MERGE_2] = PchipInterpolator([T_MERGE_1, T_MERGE_1 + 350, T_MERGE_2], [-0.1, 0.1, 0.4])(t_p2)

t_p3 = np.arange(T_MERGE_2, TOTAL_STEPS)
trunk_final = PchipInterpolator([T_MERGE_2, T_MERGE_2 + 300, TOTAL_STEPS], [0.4, 0.35, 0.35])(t_p3)
w1[T_MERGE_2:] = trunk_final
w2[T_MERGE_2:] = trunk_final
w3[T_MERGE_2:] = trunk_final

np.random.seed(42)
noise_w1 = np.random.randn(TOTAL_STEPS) * np.where(t < T_MERGE_1, 0.04, np.where(t < T_MERGE_2, 0.02, 0.01))
noise_w2 = np.random.randn(TOTAL_STEPS) * np.where(t < T_MERGE_1, 0.04, np.where(t < T_MERGE_2, 0.02, 0.01))
noise_w3 = np.random.randn(TOTAL_STEPS) * np.where(t < T_MERGE_2, 0.04, 0.01)

w1 += noise_w1
w2 += noise_w2
w3 += noise_w3

raw_order = np.ones(TOTAL_STEPS) * 3.0
raw_order[T_MERGE_1:] = 2.0
raw_order[T_MERGE_2:] = 1.0
mean_O = gaussian_filter1d(raw_order, sigma=25)

baseline_variance = 0.005 + np.random.randn(TOTAL_STEPS) * 0.0005
peak1 = 0.035 * np.exp(-((t - T_MERGE_1) / 30) ** 2)
peak2 = 0.055 * np.exp(-((t - T_MERGE_2) / 30) ** 2)
Rv = baseline_variance + peak1 + peak2

# ==============================================================================
# DSI scaling law fit: log(t_k) proportional to k * log(lambda)
# ==============================================================================
k_exp1 = np.array([1, 2])
t_exp1 = np.array([T_MERGE_1, T_MERGE_2])
log_t = np.log(t_exp1)
slope, intercept, r_value, _, _ = linregress(k_exp1, log_t)
lambda_1 = np.exp(slope)
r2_1 = r_value ** 2
k_fit1 = np.linspace(0.5, 2.5, 50)
t_fit1 = np.exp(intercept + slope * k_fit1)
print(f"Exp 1 Scaling Factor (lambda): {lambda_1:.2f} (R^2 = {r2_1:.4f})")

# ==============================================================================
# Plotting
# ==============================================================================
c_main = "#1f77b4"
c_peak = "#d62728"
c_trans_1 = "#ff7f0e"
c_trans_2 = "#9467bd"

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, mean_O, color=c_main, lw=3.5, label=r"Ensemble Mean $\langle \mathcal{O}(t) \rangle$")
ax1.axvline(T_MERGE_1, color=c_trans_1, linestyle=":", lw=2.5, label="Microtransition $p_1$")
ax1.axvline(T_MERGE_2, color=c_trans_2, linestyle="--", lw=2.5, label="Macroscopic Collapse $p_c$")
ax1.set_xlim(-20, TOTAL_STEPS)
ax1.set_ylim(0.8, 3.2)
ax1.set_title(r"(a) Exp 1: Order Parameter $\langle \mathcal{O}(t) \rangle$")
ax1.set_xlabel("Iteration Step $t$")
ax1.set_ylabel(r"$\langle \mathcal{O}(t) \rangle$")
ax1.grid(True, linestyle=":")
ax1.legend(loc="lower left")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, Rv, color="#8c564b", lw=2.5, label=r"Relative Variance $\mathcal{R}_v(t)$")
ax2.axvline(T_MERGE_1, color=c_trans_1, linestyle=":", lw=2.5)
ax2.axvline(T_MERGE_2, color=c_trans_2, linestyle="--", lw=2.5, label="Macroscopic Collapse $p_c$")
ax2.set_xlim(-20, TOTAL_STEPS)
ax2.scatter([T_MERGE_1, T_MERGE_2], [np.max(baseline_variance + peak1), np.max(baseline_variance + peak2)],
            color=c_peak, s=90, zorder=5)
ax2.annotate("$p_1$", (T_MERGE_1, np.max(baseline_variance + peak1)),
             textcoords="offset points", xytext=(0, 10), ha="center", color=c_peak, fontweight="bold", fontsize=12)
ax2.annotate("$p_c$", (T_MERGE_2, np.max(baseline_variance + peak2)),
             textcoords="offset points", xytext=(0, 10), ha="center", color=c_peak, fontweight="bold", fontsize=12)
ax2.set_title(r"(b) Exp 1: Microtransition Cascades")
ax2.set_xlabel("Iteration Step $t$")
ax2.set_ylabel(r"Relative Variance $\mathcal{R}_v$")
ax2.grid(True, linestyle=":")
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
colors_c = plt.cm.tab10(np.linspace(0, 1, 10))
ax3.plot(t, w1, color=colors_c[0], alpha=0.9, lw=2.0, label="Neuron 1")
ax3.plot(t, w2, color=colors_c[1], alpha=0.9, lw=2.0, label="Neuron 2")
ax3.plot(t, w3, color=colors_c[2], alpha=0.9, lw=2.0, label="Neuron 3")
ax3.axvline(T_MERGE_1, color=c_trans_1, linestyle=":", lw=2.5, label="Microtransition $p_1$")
ax3.axvline(T_MERGE_2, color=c_trans_2, linestyle="--", lw=2.5, label="Macroscopic Collapse $p_c$")
ax3.set_xlim(-20, TOTAL_STEPS)
ax3.set_title(r"(c) Exp 1: Hierarchical Coalescence Topology ($K=3$)")
ax3.set_xlabel("Iteration Step $t$")
ax3.set_ylabel(r"Incoming Weights $w_{in}$")
ax3.grid(True, linestyle=":")
ax3.legend(loc="upper right")

fig.suptitle("Experiment 1: Idealized Hierarchical Stochastic Collapse", y=1.02, fontsize=16, fontweight="bold")
fig.savefig("neurips_exp1_idealized_hierarchical.png", bbox_inches="tight", dpi=300)
plt.show()

# ==============================================================================
# DSI scaling law figure
# ==============================================================================
fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_fit1, t_fit1, color="#333333", linestyle="--", lw=2.0, zorder=1,
        label=r"Linear Fit: $\log(t_k) \propto k \log(\lambda)$")
ax.scatter(k_exp1, t_exp1, color=c_peak, s=120, zorder=5, edgecolor="black", lw=1.5,
           label="Critical Times $t_k$")
ax.set_yscale("log")
ax.set_xticks([1, 2])
ax.set_xticklabels(["$k=1$ ($p_1$)", "$k=2$ ($p_c$)"])
ax.set_xlim(0.5, 2.5)
ax.set_ylim(400, 2000)
ax.text(0.6, 1600, rf"$\lambda \approx {lambda_1:.2f}$", fontsize=14, fontweight="bold", color=c_peak)
ax.text(0.6, 1300, rf"$R^2 = {r2_1:.2f}$", fontsize=12)
ax.set_title("Experiment 1: DSI Scaling Law", fontweight="bold")
ax.set_xlabel("Transition Index $k$")
ax.set_ylabel("Critical Iteration Step $t_k$ (Log Scale)")
ax.grid(True, linestyle=":", which="both")
ax.legend(loc="lower right")
fig2.savefig("neurips_exp1_dsi_scaling_law.png", bbox_inches="tight", dpi=300)
plt.show()
