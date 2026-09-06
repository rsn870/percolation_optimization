# Percolation Dynamics in Optimization — Experiment Code

Code for the toy, empirical, and null-model experiments in the paper. Each file is
self-contained and runnable on its own.

## Files

| File | Experiment | Output |
|---|---|---|
| `toy_invariance_demo.py` | Single-neuron sign invariance, two-neuron permutation invariance | `neurips_toy_collapse.png` |
| `exp1_kinematic_k3.py` | Idealized K=3 kinematic verification of λ=2 (Appendix D.3) | `neurips_exp1_idealized_hierarchical.png`, `neurips_exp1_dsi_scaling_law.png` |
| `exp2_empirical_k6.py` | Empirical K=6 SGD with task shift, single run + 40-seed ensemble (Appendix D.4) | `exp2_partA_raw_trajectories.png`, `exp2_ensemble_multiphase_peaks.png`, `exp2_dsi_scaling_law.png` |
| `exp3_uci_tabular.py` | UCI tabular suite with spectral + bootstrap null tests (Table 1) | `exp3_uci_<dataset>_detrended_with_null.png`, JSON results in `null_test_results/` |
| `exp4_sde_multiplicative.py` | 3D multiplicative-noise SDE, staggered collapse thresholds | `exp4_sde_multiplicative_smoothed.png` |
| `exp5_suite_grokking.py` | Moons/Swiss Roll/MNIST/FashionMNIST/grokking, null tests, Adam/AdamW diagnostics (Remark D.28) | `suite_modular_arithmetic_with_null.png`, JSON results in `null_test_results/` |

## Setup

```bash
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib joblib tqdm
```

A CUDA GPU is used automatically if available; all scripts fall back to CPU otherwise.

## Running

Each script runs standalone:

```bash
python toy_invariance_demo.py
python exp1_kinematic_k3.py
python exp2_empirical_k6.py
python exp3_uci_tabular.py     # downloads UCI datasets on first run, caches to ./uci_datasets_cache/
python exp4_sde_multiplicative.py
python exp5_suite_grokking.py  # slowest script, 15 x 30000-epoch runs for the grokking task
```

Figures save to the working directory. `exp3` and `exp5` additionally write per-dataset
JSON results (real λ, R², null-test p-values, bootstrap CIs) to `./null_test_results/`.



## Link to the paper

[arXiv:2609.02373](https://arxiv.org/abs/2609.02373), Percolation Dynamics in Optimization: Variance Cascades and Discrete Scale Invariance

