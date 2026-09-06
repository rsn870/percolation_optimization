"""
Manifold, vision, and grokking suite with null tests and Adam/AdamW diagnostics (Appendix F.3).
"""
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons, make_swiss_roll
from torchvision import datasets, transforms
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "grid.alpha": 0.4,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "./null_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# Dataset and architecture factory
# ==============================================================================
def get_experiment_setup(exp_name):
    """Returns (train_loader, val_loader_or_None, model, target_layer, epochs, lr)."""
    if exp_name in ["Moons", "Swiss Roll"]:
        n_samples = 2000
        if exp_name == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=0.1)
            in_dim = 2
        else:
            X, _ = make_swiss_roll(n_samples=n_samples, noise=0.1)
            y = (X[:, 0] > 0).astype(int)
            in_dim = 3

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        model = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2),
        ).to(device)
        return train_loader, None, model, model[0].weight, 100, 0.1

    elif exp_name in ["MNIST", "FashionMNIST"]:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        DatasetClass = datasets.MNIST if exp_name == "MNIST" else datasets.FashionMNIST
        dataset = DatasetClass(root="./data", train=True, download=True, transform=transform)
        subset_indices = torch.randperm(len(dataset))[:5000]
        subset = torch.utils.data.Subset(dataset, subset_indices)
        train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 10),
        ).to(device)
        return train_loader, None, model, model[1].weight, 50, 0.05

    elif exp_name == "Modular Arithmetic":
        P = 17
        vocab_size = P + 1
        X_all, y_all = [], []
        for a in range(P):
            for b in range(P):
                X_all.append([a, b, P])
                y_all.append((a + b) % P)
        X_t = torch.tensor(X_all, dtype=torch.long)
        y_t = torch.tensor(y_all, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)

        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

        class UnnormalizedTransformerLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(0.1)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.activation = nn.GELU()

            def forward(self, x):
                attn_out, _ = self.self_attn(x, x, x)
                x = x + attn_out
                ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
                x = x + ff_out
                return x

        class NandaTransformer(nn.Module):
            def __init__(self, vocab_size, d_model=128):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = nn.Parameter(torch.randn(1, 3, d_model))
                self.layer = UnnormalizedTransformerLayer(d_model=d_model, nhead=4, dim_feedforward=512)
                self.out = nn.Linear(d_model, P)

            def forward(self, x):
                h = self.embed(x) + self.pos
                h = self.layer(h)
                return self.out(h[:, 2, :])

        model = NandaTransformer(vocab_size=vocab_size).to(device)
        return train_loader, val_loader, model, model.embed.weight, 30000, 3e-3


# ==============================================================================
# Metrics and shared pipeline logic
# ==============================================================================
def compute_effective_rank(weight_matrix):
    with torch.no_grad():
        s = torch.linalg.svdvals(weight_matrix)
        p = s / torch.sum(s)
        p = p[p > 1e-9]
        entropy = -torch.sum(p * torch.log(p))
        return torch.exp(entropy).item()


def calculate_dsi(peaks_idx):
    if len(peaks_idx) < 3:
        return None, None
    peak_epochs = peaks_idx + 1
    k = np.arange(1, len(peak_epochs) + 1)
    slope, _, r_val, _, _ = linregress(k, np.log(peak_epochs))
    return np.exp(slope), r_val ** 2


def detrend_from_var(var_O, macro_sigma):
    epochs_arr = np.arange(len(var_O))
    log_var = np.log(var_O + 1e-10)
    slope, intercept, _, _, _ = linregress(epochs_arr, log_var)
    linear_baseline = slope * epochs_arr + intercept
    flat_log_var = log_var - linear_baseline
    macro_flat = gaussian_filter1d(flat_log_var, sigma=macro_sigma, mode="reflect")
    macro_trend_log = macro_flat + linear_baseline
    return log_var - macro_trend_log


def run_detection_pipeline(var_detrended_raw, prominence, distance, local_sigma):
    Rv_smooth = gaussian_filter1d(var_detrended_raw, sigma=local_sigma)
    denom = np.max(np.abs(Rv_smooth))
    if denom > 0:
        Rv_smooth = Rv_smooth / denom
    peaks_idx, _ = find_peaks(Rv_smooth, prominence=prominence, distance=distance)
    lam, r2 = calculate_dsi(peaks_idx)
    return Rv_smooth, peaks_idx, lam, r2


# ==============================================================================
# Null model tests
# ==============================================================================
def phase_randomize(x, rng):
    n = len(x)
    f = np.fft.rfft(x)
    amps = np.abs(f)
    phases = rng.uniform(0, 2 * np.pi, size=len(f))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    f_surr = amps * np.exp(1j * phases)
    return np.fft.irfft(f_surr, n=n)


def spectral_null_test(var_detrended_raw, real_r2, detect_kwargs, n_surrogates=1000, seed=0):
    rng = np.random.default_rng(seed)
    null_r2, null_lam = [], []
    for _ in range(n_surrogates):
        surr = phase_randomize(var_detrended_raw, rng)
        _, _, lam_s, r2_s = run_detection_pipeline(surr, **detect_kwargs)
        if r2_s is not None:
            null_r2.append(r2_s)
            null_lam.append(lam_s)
    null_r2 = np.array(null_r2)
    null_lam = np.array(null_lam)

    out = {
        "n_surrogates_requested": n_surrogates,
        "n_surrogates_with_valid_fit": int(len(null_r2)),
        "real_r2": None if real_r2 is None else float(real_r2),
    }
    if real_r2 is not None and len(null_r2) > 0:
        p_val = (1 + np.sum(null_r2 >= real_r2)) / (len(null_r2) + 1)
        out["p_value_r2_ge_real"] = float(p_val)
        out["null_r2_mean"] = float(np.mean(null_r2))
        out["null_r2_median"] = float(np.median(null_r2))
        out["null_r2_90th_pct"] = float(np.percentile(null_r2, 90))
    out["null_r2_raw"] = null_r2.tolist()
    out["null_lambda_raw"] = null_lam.tolist()
    return out


def seed_bootstrap_test(order_param_matrix, macro_sigma, detect_kwargs, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n_real = order_param_matrix.shape[0]
    boot_r2, boot_lam = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_real, size=n_real)
        resampled = order_param_matrix[idx, :]
        var_O_b = np.var(resampled, axis=0)
        var_detrended_b = detrend_from_var(var_O_b, macro_sigma)
        _, _, lam_b, r2_b = run_detection_pipeline(var_detrended_b, **detect_kwargs)
        if r2_b is not None:
            boot_r2.append(r2_b)
            boot_lam.append(lam_b)
    boot_r2, boot_lam = np.array(boot_r2), np.array(boot_lam)
    out = {"n_boot": n_boot, "n_boot_with_valid_fit": int(len(boot_r2))}
    if len(boot_lam) > 0:
        out["lambda_boot_mean"] = float(np.mean(boot_lam))
        out["lambda_boot_ci95"] = [float(np.percentile(boot_lam, 2.5)), float(np.percentile(boot_lam, 97.5))]
    return out


# Adam/AdamW assumption diagnostics (Remark D.28). Only realization 0 is instrumented.
def autocorr_1d(x, max_lag):
    x = x - np.mean(x)
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    return np.array([
        np.dot(x[:n - k], x[k:]) / ((n - k) * var) if k > 0 else 1.0
        for k in range(max_lag + 1)
    ])


class AdamDiagnosticRecorder:
    def __init__(self, i_unit=3, j_unit=7, k_in=5, burnin_steps=20):
        self.i_unit, self.j_unit, self.k_in = i_unit, j_unit, k_in
        self.burnin_steps = burnin_steps
        self.grad_i, self.grad_j = [], []
        self.v_i, self.v_j = [], []
        self.theta_i, self.theta_j = [], []
        self.global_vmin_trace = []
        self.step = 0

    def record(self, model, optimizer):
        lin1 = model.layer.linear1
        self.grad_i.append(lin1.weight.grad[self.i_unit, self.k_in].item())
        self.grad_j.append(lin1.weight.grad[self.j_unit, self.k_in].item())
        self.theta_i.append(lin1.weight.data[self.i_unit, self.k_in].item())
        self.theta_j.append(lin1.weight.data[self.j_unit, self.k_in].item())
        self.step += 1

        st = optimizer.state.get(lin1.weight, None)
        if st is not None and "exp_avg_sq" in st:
            self.v_i.append(st["exp_avg_sq"][self.i_unit, self.k_in].item())
            self.v_j.append(st["exp_avg_sq"][self.j_unit, self.k_in].item())

        if self.step >= self.burnin_steps:
            vmins = [s["exp_avg_sq"].min().item() for p, s in optimizer.state.items() if "exp_avg_sq" in s]
            if vmins:
                self.global_vmin_trace.append(min(vmins))

    def summarize(self, optimizer):
        beta1, beta2 = optimizer.defaults["betas"]
        eps_adam = optimizer.defaults["eps"]
        tau1, tau2 = 1.0 / (1 - beta1), 1.0 / (1 - beta2)

        grad_i, grad_j = np.array(self.grad_i), np.array(self.grad_j)
        v_i, v_j = np.array(self.v_i), np.array(self.v_j)
        theta_i, theta_j = np.array(self.theta_i), np.array(self.theta_j)
        global_vmin_trace = np.array(self.global_vmin_trace) if self.global_vmin_trace else np.array([np.nan])

        v_min_local = float(min(v_i.min() if len(v_i) else np.nan, v_j.min() if len(v_j) else np.nan))
        v_min_global = float(global_vmin_trace.min())

        max_lag = min(200, max(4, len(grad_i) // 4))
        ac_i = autocorr_1d(grad_i ** 2, max_lag)
        ac_j = autocorr_1d(grad_j ** 2, max_lag)
        mean_ac = (ac_i + ac_j) / 2
        below = np.where(mean_ac < 1 / np.e)[0]
        tau_corr = int(below[0]) if len(below) > 0 else max_lag
        e_t_holds = bool(tau_corr < tau2)

        d_theta = np.abs(theta_i - theta_j)
        d_v = np.abs(v_i - v_j)
        nz = d_theta > 1e-8
        ratio = d_v[nz] / d_theta[nz] if nz.any() else np.array([np.nan])
        corr_dv_dtheta = float(np.corrcoef(d_theta, d_v)[0, 1]) if len(d_theta) > 1 else float("nan")

        g_all = np.concatenate([grad_i, grad_j])
        heavy_tail_ratio = float(np.abs(g_all).max() / g_all.std()) if g_all.std() > 0 else float("nan")

        return {
            "tau_1_steps": tau1,
            "tau_2_steps": tau2,
            "n_steps_recorded": int(len(grad_i)),
            "nondegeneracy": {
                "v_min_local_tracked_pair": v_min_local,
                "v_min_global_all_params_post_burnin": v_min_global,
                "eps": eps_adam,
            },
            "correlation_time": {
                "tau_corr_steps": tau_corr,
                "tau_2_steps": tau2,
                "E_t_holds": e_t_holds,
            },
            "block_homogeneity_mechanism": {
                "median_dv_over_dtheta": float(np.median(ratio)),
                "p95_dv_over_dtheta": float(np.percentile(ratio, 95)),
                "corr_dtheta_dv": corr_dv_dtheta,
            },
            "heavy_tail_proxy": {"max_abs_grad_over_std": heavy_tail_ratio},
        }


# ==============================================================================
# Experiment runner
# ==============================================================================
EXPERIMENTS_TO_RUN = ["Modular Arithmetic"]
NUM_REALIZATIONS = 15
RUN_ADAM_DIAGNOSTICS = True  # instrument only realization 0, only for AdamW experiments

for exp_name in EXPERIMENTS_TO_RUN:
    print(f"\n--- Running: {exp_name} ---")
    _, val_loader, _, _, EPOCHS, LR = get_experiment_setup(exp_name)

    order_param_matrix = np.zeros((NUM_REALIZATIONS, EPOCHS))
    train_loss_matrix = np.zeros((NUM_REALIZATIONS, EPOCHS))
    val_loss_matrix = np.zeros((NUM_REALIZATIONS, EPOCHS)) if val_loader is not None else None
    adam_diagnostics = None

    for r in range(NUM_REALIZATIONS):
        print(f"\n[Realization {r+1}/{NUM_REALIZATIONS}] Training for {EPOCHS} epochs...")
        train_loader, val_loader, model, target_layer, _, _ = get_experiment_setup(exp_name)

        if exp_name == "Modular Arithmetic":
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.5)
        else:
            optimizer = optim.SGD(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        run_diagnostics_this_realization = (
            RUN_ADAM_DIAGNOSTICS and r == 0 and isinstance(optimizer, optim.AdamW)
        )
        recorder = AdamDiagnosticRecorder() if run_diagnostics_this_realization else None

        for epoch in tqdm(range(EPOCHS)):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                if recorder is not None:
                    recorder.record(model, optimizer)
                epoch_loss += loss.item()

            train_loss_matrix[r, epoch] = epoch_loss / len(train_loader)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        out = model(batch_X)
                        loss = criterion(out, batch_y)
                        val_loss += loss.item()
                val_loss_matrix[r, epoch] = val_loss / len(val_loader)

            order_param_matrix[r, epoch] = compute_effective_rank(target_layer)

        if recorder is not None:
            adam_diagnostics = recorder.summarize(optimizer)
            print("\n[Adam/AdamW Assumption Diagnostics]")
            print(json.dumps(adam_diagnostics, indent=2))
            diag_path = os.path.join(RESULTS_DIR, f"{exp_name.replace(' ', '_').lower()}_adam_diagnostics_only.json")
            with open(diag_path, "w") as f:
                json.dump(adam_diagnostics, f, indent=2)

        final_train = train_loss_matrix[r, -1]
        final_val = val_loss_matrix[r, -1] if val_loss_matrix is not None else "N/A"
        final_erank = order_param_matrix[r, -1]
        val_str = final_val if isinstance(final_val, str) else f"{final_val:.5f}"
        print(f" -> Realization {r+1} finished. Final Train Loss: {final_train:.5f} | "
              f"Final Val Loss: {val_str} | Final Effective Rank: {final_erank:.2f}")

    macro_sigma = max(1.0, EPOCHS // 5)
    detect_kwargs = {
        "local_sigma": max(1.0, EPOCHS // 50 + 1),
        "distance": max(1, EPOCHS // 20),
        "prominence": 0.05,
    }

    mean_O = np.mean(order_param_matrix, axis=0)
    var_O = np.var(order_param_matrix, axis=0)
    mean_train_loss = np.mean(train_loss_matrix, axis=0)
    mean_val_loss = np.mean(val_loss_matrix, axis=0) if val_loss_matrix is not None else None

    var_detrended_raw = detrend_from_var(var_O, macro_sigma)
    Rv_smooth, peaks_idx, lam, r2 = run_detection_pipeline(var_detrended_raw, **detect_kwargs)
    denom = np.max(np.abs(var_detrended_raw))
    var_detrended = var_detrended_raw / denom if denom > 0 else var_detrended_raw

    print("Running spectral null (1000 surrogates)...")
    spectral_null = spectral_null_test(var_detrended_raw, r2, detect_kwargs, n_surrogates=1000, seed=42)
    print("Running seed-bootstrap null (500 resamples)...")
    seed_null = seed_bootstrap_test(order_param_matrix, macro_sigma, detect_kwargs, n_boot=500, seed=42)
    if "p_value_r2_ge_real" in spectral_null:
        print(f"[{exp_name}] p(R2 >= real) = {spectral_null['p_value_r2_ge_real']:.4f}")

    # --- Plotting: order parameter/loss, detrended cascade, null histogram ---
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    epochs_x = np.arange(1, EPOCHS + 1)
    c_main, c_loss, c_peak = "#1f77b4", "#ff7f0e", "#d62728"

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_x, mean_O, color=c_main, lw=3.0, label=r"Effective Rank $\langle \mathcal{O}(t) \rangle$")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Effective Rank", color=c_main)
    ax1.tick_params(axis="y", labelcolor=c_main)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_x, mean_train_loss, color=c_loss, lw=2.0, linestyle="--", label="Train Loss")
    if mean_val_loss is not None:
        ax1_twin.plot(epochs_x, mean_val_loss, color="green", lw=2.0, linestyle=":", label="Validation Loss")
    ax1_twin.set_ylabel("Loss", color=c_loss)
    ax1_twin.tick_params(axis="y", labelcolor=c_loss)
    if exp_name == "Modular Arithmetic":
        ax1_twin.set_yscale("log")
    ax1.set_title(f"(a) {exp_name}: Collapse & Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_x, var_detrended, color="#8c564b", alpha=0.3, lw=1.0, label="Raw Fluctuations")
    ax2.plot(epochs_x, Rv_smooth, color="#8c564b", lw=2.5, label=r"Detrended Variance $\mathcal{R}_v(t)$")
    for i, idx in enumerate(peaks_idx):
        tp = epochs_x[idx]
        ax2.scatter(tp, Rv_smooth[idx], color=c_peak, s=90, zorder=5)
        ax2.annotate(f"$p_{i+1}$", (tp, Rv_smooth[idx]), textcoords="offset points",
                     xytext=(0, 10), ha="center", color=c_peak, fontweight="bold")
    dsi_title = "(b) Variance Peaks"
    if lam is not None:
        dsi_title += f"\n(DSI $\\lambda={lam:.2f}, R^2={r2:.2f}$)"
    ax2.set_title(dsi_title)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Variance Fluctuation")
    ax2.set_ylim(-1.2, 1.4)
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[0, 2])
    if len(spectral_null["null_r2_raw"]) > 0:
        ax3.hist(spectral_null["null_r2_raw"], bins=30, color="#7f7f7f", alpha=0.7,
                 label=f"Phase-randomized null\n(n={len(spectral_null['null_r2_raw'])})")
        if r2 is not None:
            ax3.axvline(r2, color=c_peak, lw=2.5, linestyle="--", label=f"Real $R^2$={r2:.2f}")
            p_txt = spectral_null.get("p_value_r2_ge_real", None)
            if p_txt is not None:
                ax3.text(0.02, 0.95, f"p = {p_txt:.3f}", transform=ax3.transAxes, va="top", ha="left",
                         fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax3.set_title("(c) Null-Model Control\n(spectral surrogate $R^2$ distribution)")
    ax3.set_xlabel(r"Fitted $R^2$ (log-linear DSI fit)")
    ax3.set_ylabel("Count (surrogates)")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    clean_name = exp_name.replace(" ", "_").lower()
    fig.suptitle(f"Topological Dynamics: {exp_name}", y=1.03, fontsize=15, fontweight="bold")
    plt.savefig(f"suite_{clean_name}_with_null.png", bbox_inches="tight", dpi=300)
    plt.close()

    output_record = {
        "dataset": exp_name,
        "lambda_real": None if lam is None else float(lam),
        "r2_real": None if r2 is None else float(r2),
        "spectral_null": {k: v for k, v in spectral_null.items() if not k.endswith("_raw")},
        "seed_bootstrap_null": {k: v for k, v in seed_null.items() if not k.endswith("_raw")},
    }
    if adam_diagnostics is not None:
        output_record["adam_assumption_diagnostics"] = adam_diagnostics

    with open(os.path.join(RESULTS_DIR, f"{clean_name}_null_tests.json"), "w") as f:
        json.dump(output_record, f, indent=2)

print("\nSuite execution complete.")
