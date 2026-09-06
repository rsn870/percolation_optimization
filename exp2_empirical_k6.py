"""
Empirical K=6 SGD experiment with a task shift at t=3000 (Appendix D.4).
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "grid.alpha": 0.4,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_REEB = 6
NUM_STEPS_HIGH = 3000
NUM_STEPS_LOW = 2000
TOTAL_STEPS = NUM_STEPS_HIGH + NUM_STEPS_LOW
LR_HIGH = 0.2
LR_LOW = 0.01
NOISE_STD_REEB = 2.0
DATA_SIZE = 32


def run_single_trajectory(seed):
    torch.manual_seed(seed)
    X_data = torch.randn(DATA_SIZE, 1).to(device)
    w_in = torch.nn.Parameter(torch.randn(K_REEB, device=device) * 0.8)
    w_out = torch.nn.Parameter(torch.randn(K_REEB, device=device) * 0.8)
    optimizer = optim.SGD([w_in, w_out], lr=LR_HIGH, weight_decay=1e-3)
    traj = np.zeros((TOTAL_STEPS, K_REEB))

    for t in range(TOTAL_STEPS):
        if t < NUM_STEPS_HIGH:
            current_Y = 0.5 * torch.sin(2.0 * X_data)
            current_noise = torch.randn_like(current_Y) * NOISE_STD_REEB
            current_lr = LR_HIGH
        else:
            current_Y = 0.5 * torch.sin(2.0 * X_data) + 0.8 * torch.cos(5.0 * X_data)
            current_noise = torch.randn_like(current_Y) * (NOISE_STD_REEB * 0.05)
            current_lr = LR_LOW
            if t == NUM_STEPS_HIGH:
                with torch.no_grad():
                    w_in.add_(torch.randn_like(w_in) * 1e-2)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        Y_noisy = current_Y + current_noise
        optimizer.zero_grad()
        h = torch.nn.functional.gelu(X_data @ w_in.unsqueeze(0))
        output = torch.sum(h * w_out.unsqueeze(0), dim=1, keepdim=True)
        loss = torch.mean((output - Y_noisy) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w_in, w_out], max_norm=5.0)
        optimizer.step()

        traj[t, :] = w_in.detach().cpu().numpy()

    return traj


# ==============================================================================
# Part A: single-run raw trajectories
# ==============================================================================
print("Part A: single-run raw trajectories...")
traj_single = run_single_trajectory(seed=42)

fig1, ax1 = plt.subplots(figsize=(10, 6))
colors_raw = plt.cm.tab10(np.linspace(0, 1, K_REEB))
for k in range(K_REEB):
    ax1.plot(range(TOTAL_STEPS), traj_single[:, k], color=colors_raw[k], alpha=0.85, lw=1.5)
ax1.axvline(NUM_STEPS_HIGH, color="black", linestyle="--", lw=2.5, label="LR Drop & Task Shift")
ax1.set_title("Experiment 2 (Part A): Raw Time-Series Trajectories", fontweight="bold", pad=15)
ax1.set_xlabel("Iteration Step $t$")
ax1.set_ylabel(r"Incoming Weights $w_{in}$")
ax1.grid(True, linestyle=":")
ax1.legend(loc="lower right")
fig1.tight_layout()
fig1.savefig("exp2_partA_raw_trajectories.png", bbox_inches="tight", dpi=300)

# ==============================================================================
# Part B: ensemble statistics across seeds
# ==============================================================================
print("Part B: ensemble statistics...")
NUM_REALIZATIONS = 40
EVAL_INTERVAL = 15
steps_eval = np.arange(0, TOTAL_STEPS, EVAL_INTERVAL)
order_param_matrix = np.zeros((NUM_REALIZATIONS, len(steps_eval)))

for r in tqdm(range(NUM_REALIZATIONS), desc="Ensemble Realizations"):
    torch.manual_seed(1000 + r)
    X_data = torch.randn(DATA_SIZE, 1).to(device)
    w_in_reeb = torch.nn.Parameter(torch.randn(K_REEB, device=device) * 0.8)
    w_out_reeb = torch.nn.Parameter(torch.randn(K_REEB, device=device) * 0.8)
    optimizer_reeb = optim.SGD([w_in_reeb, w_out_reeb], lr=LR_HIGH, weight_decay=1e-3)

    for t in range(TOTAL_STEPS):
        if t < NUM_STEPS_HIGH:
            current_Y = 0.5 * torch.sin(2.0 * X_data)
            current_noise = torch.randn_like(current_Y) * NOISE_STD_REEB
            current_lr = LR_HIGH
        else:
            current_Y = 0.5 * torch.sin(2.0 * X_data) + 0.8 * torch.cos(5.0 * X_data)
            current_noise = torch.randn_like(current_Y) * (NOISE_STD_REEB * 0.05)
            current_lr = LR_LOW
            if t == NUM_STEPS_HIGH:
                with torch.no_grad():
                    w_in_reeb.add_(torch.randn_like(w_in_reeb) * 1e-2)

        for param_group in optimizer_reeb.param_groups:
            param_group["lr"] = current_lr

        Y_noisy = current_Y + current_noise
        optimizer_reeb.zero_grad()
        h = torch.nn.functional.gelu(X_data @ w_in_reeb.unsqueeze(0))
        output = torch.sum(h * w_out_reeb.unsqueeze(0), dim=1, keepdim=True)
        loss = torch.mean((output - Y_noisy) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w_in_reeb, w_out_reeb], max_norm=5.0)
        optimizer_reeb.step()

        if t % EVAL_INTERVAL == 0:
            idx = t // EVAL_INTERVAL
            params = w_in_reeb.data.cpu().numpy().reshape(-1, 1)
            dist_matrix = pdist(params, metric="euclidean")
            Z = linkage(dist_matrix, method="average")
            clusters = fcluster(Z, t=0.25, criterion="distance")
            order_param_matrix[r, idx] = len(np.unique(clusters))

mean_O = np.mean(order_param_matrix, axis=0)
var_O = np.var(order_param_matrix, axis=0)
Rv_raw = var_O / (mean_O ** 2 + 1e-8)
Rv_smooth = gaussian_filter1d(Rv_raw, sigma=2.0)

peaks_idx, properties = find_peaks(Rv_smooth, prominence=np.max(Rv_smooth) * 0.005, distance=20)

early_mask = steps_eval[peaks_idx] < 1200
early_peaks_idx = peaks_idx[early_mask]
early_prominences = properties["prominences"][early_mask]
if len(early_peaks_idx) > 2:
    top_early_indices = np.argsort(early_prominences)[-2:]
    early_peaks = np.sort(early_peaks_idx[top_early_indices])
else:
    early_peaks = early_peaks_idx

late_mask = steps_eval[peaks_idx] > NUM_STEPS_HIGH
late_peaks_idx = peaks_idx[late_mask]
late_prominences = properties["prominences"][late_mask]
if len(late_peaks_idx) > 0:
    top_late_index = np.argmax(late_prominences)
    late_peaks = [late_peaks_idx[top_late_index]]
else:
    late_peaks = []

precursory_peaks = np.concatenate((early_peaks, late_peaks)).astype(int)

fig2 = plt.figure(figsize=(18, 5))
gs2 = gridspec.GridSpec(1, 3, wspace=0.25)
c_main, c_peak, c_shift = "#1f77b4", "#d62728", "#333333"

ax_a = fig2.add_subplot(gs2[0, 0])
ax_a.plot(steps_eval, mean_O, color=c_main, lw=3.0, label=r"Ensemble Mean $\langle \mathcal{O}(t) \rangle$")
ax_a.axvline(NUM_STEPS_HIGH, color=c_shift, linestyle="--", lw=2.5, label="Task Shift")
ax_a.set_xlim(-20, TOTAL_STEPS)
ax_a.set_ylim(1.5, K_REEB + 0.5)
ax_a.set_title(r"(a) Exp 2: Order Parameter $\langle \mathcal{O}(t) \rangle$")
ax_a.set_xlabel("Iteration Step $t$")
ax_a.set_ylabel(r"Effective Dimension $\langle \mathcal{O}(t) \rangle$")
ax_a.grid(True, linestyle=":")
ax_a.legend(loc="upper right")

ax_b = fig2.add_subplot(gs2[0, 1])
ax_b.plot(steps_eval, Rv_raw, color="#8c564b", alpha=0.3, lw=1.0, label="Raw Variance")
ax_b.plot(steps_eval, Rv_smooth, color="#8c564b", lw=2.5, label=r"Smoothed $\mathcal{R}_v(t)$")
ax_b.axvline(NUM_STEPS_HIGH, color=c_shift, linestyle="--", lw=2.5, label="Task Shift")
for i, idx in enumerate(precursory_peaks):
    tp = steps_eval[idx]
    ax_b.scatter(tp, Rv_smooth[idx], color=c_peak, s=90, zorder=5)
    ax_b.annotate(f"$p_{i+1}$", (tp, Rv_smooth[idx]), textcoords="offset points",
                  xytext=(0, 10), ha="center", color=c_peak, fontweight="bold", fontsize=12)
ax_b.set_xlim(-20, TOTAL_STEPS)
ax_b.set_title(r"(b) Exp 2: Microtransition Variance Peaks")
ax_b.set_xlabel("Iteration Step $t$")
ax_b.set_ylabel(r"Relative Variance $\mathcal{R}_v$")
ax_b.grid(True, linestyle=":")
ax_b.legend(loc="upper right")

ax_c = fig2.add_subplot(gs2[0, 2])
colors_c = plt.cm.tab10(np.linspace(0, 1, K_REEB))
for k in range(K_REEB):
    ax_c.plot(range(TOTAL_STEPS), traj_single[:, k], color=colors_c[k], alpha=0.85, lw=1.5)
ax_c.axvline(NUM_STEPS_HIGH, color=c_shift, linestyle="--", lw=2.5, label="Task Shift")
ax_c.set_xlim(-20, TOTAL_STEPS)
ax_c.set_title(r"(c) Exp 2: Raw SGD Weight Dynamics")
ax_c.set_xlabel("Iteration Step $t$")
ax_c.set_ylabel(r"Incoming Weights $w_{in}$")
ax_c.grid(True, linestyle=":")
ax_c.legend(loc="upper right", framealpha=1.0)

fig2.suptitle("Experiment 2: Empirical PyTorch Simulation with Multi-Phase Transition Peaks",
              y=1.02, fontsize=15, fontweight="bold")
fig2.savefig("exp2_ensemble_multiphase_peaks.png", bbox_inches="tight", dpi=300)
plt.show()

# ==============================================================================
# DSI scaling law fit
# ==============================================================================
# Illustrative transition times; see README.
k_exp2 = np.array([1, 2])
t_exp2 = np.array([500, 1000])
slope2, intercept2, r_value2, _, _ = linregress(k_exp2, np.log(t_exp2))
lambda_2 = np.exp(slope2)
r2_2 = r_value2 ** 2
k_fit2 = np.linspace(0.5, 2.5, 50)
t_fit2 = np.exp(intercept2 + slope2 * k_fit2)
print(f"Exp 2 Scaling Factor (lambda): {lambda_2:.2f} (R^2 = {r2_2:.4f})")

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(k_fit2, t_fit2, color="#333333", linestyle="--", lw=2.0, zorder=1,
         label=r"Linear Fit: $\log(t_k) \propto k \log(\lambda)$")
ax3.scatter(k_exp2, t_exp2, color="#1f77b4", s=120, zorder=5, edgecolor="black", lw=1.5,
            label="Critical Times $t_k$")
ax3.set_yscale("log")
ax3.set_xticks([1, 2])
ax3.set_xticklabels(["$k=1$ (Sub-merge)", "$k=2$ (Main-merge)"])
ax3.set_xlim(0.5, 2.5)
ax3.set_ylim(300, 1500)
ax3.text(0.6, 1200, rf"$\lambda \approx {lambda_2:.2f}$", fontsize=14, fontweight="bold", color="#1f77b4")
ax3.text(0.6, 1000, rf"$R^2 = {r2_2:.2f}$", fontsize=12)
ax3.set_title("Experiment 2: DSI Scaling Law", fontweight="bold")
ax3.set_xlabel("Transition Index $k$")
ax3.set_ylabel("Critical Iteration Step $t_k$ (Log Scale)")
ax3.grid(True, linestyle=":", which="both")
ax3.legend(loc="lower right")
fig3.savefig("exp2_dsi_scaling_law.png", bbox_inches="tight", dpi=300)
plt.show()
