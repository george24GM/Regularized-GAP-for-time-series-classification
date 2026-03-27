import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, norm

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError as e:
    raise ImportError("Need numpy>=1.20 for sliding_window_view. Please upgrade numpy.") from e


# ============================================================
# 1. Generate AR(1) process
# ============================================================
def generate_ar1(n, theta, sigma=1.0):
    eps = np.random.normal(0, sigma, size=n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = theta * X[t - 1] + eps[t]
    return X

def generate_ma1(n, theta, sigma=1.0):
    eps = np.random.normal(0, sigma, size=n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = theta * eps[t - 1] + eps[t]
    return X

# ============================================================
# 2. Fast residual block (vectorized)
# ============================================================
def resnet_block_identity_fast(U, w1, w2):
    U = np.asarray(U, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    k1 = w1.size
    k2 = w2.size
    K = k1 + k2 - 1
    n = U.size
    out_len = n - K + 1
    if out_len <= 0:
        return np.array([])

    W = sliding_window_view(U, window_shape=k1)  # (n-k1+1, k1)

    W3 = W[:out_len + k2 - 1]
    blocks = sliding_window_view(W3, window_shape=(k2, k1))[:, 0, :, :]  # (out_len, k2, k1)

    s1 = np.tensordot(blocks, w1, axes=([2], [0]))
    a1 = np.maximum(s1, 0.0)

    s2 = a1 @ w2
    nonlinear = np.maximum(s2, 0.0)

    return nonlinear + U[:out_len]


# ============================================================
# 3. Deep ResNet with B blocks
# ============================================================
def deep_resnet_features(X, w1_list, w2_list):
    assert len(w1_list) == len(w2_list)
    Z = X.copy()
    for b in range(len(w1_list)):
        Z = resnet_block_identity_fast(Z, w1_list[b], w2_list[b])
    return Z


# ============================================================
# 4. ResNet output with scaling
# ============================================================
def resnet_output(X, w1_list, w2_list):
    Z = deep_resnet_features(X, w1_list, w2_list)
    n_eff = len(Z)
    if n_eff == 0:
        return np.nan

    # If you want CLT scaling, use sqrt(n_eff). If you want GAP, use n_eff.
    return np.sum(Z) / (n_eff)   # <-- CLT scaling


# ============================================================
# 5. Monte Carlo experiment: generate S sample
# ============================================================
def run_experiment(n, theta, B, n_rep, sigma=1.0):
    w1_full = [
        np.array([0, 0, 0]),
        np.array([0.4, 0.2, -0.6]),
        np.array([-0.7, 0.5, 0.3]),
        np.array([0.6, -0.2, 0.4]),
        np.array([1, 9, -3]),
    ]

    w2_full = [
        np.array([0, 0, 0]),
        np.array([-0.8, 0.6]),
        np.array([0.9, -0.5]),
        np.array([-0.6, 0.7]),
        np.array([-0.6, 7]),
    ]

    if B > len(w1_full):
        raise ValueError(f"B={B} but only {len(w1_full)} blocks of weights are defined.")

    w1_list = w1_full[:B]
    w2_list = w2_full[:B]

    S = np.empty(n_rep)
    for r in range(n_rep):
        #X = generate_ar1(n, theta, sigma=sigma)
        X = generate_ma1(n, theta, sigma=sigma)
        S[r] = resnet_output(X, w1_list, w2_list)
    return S


# ============================================================
# 6. Helper: plot one S sample with fitted normal
# ============================================================
def plot_normal_fit(S, title_prefix="One run"):
    mu_hat = float(np.mean(S))
    sigma_hat = float(np.std(S, ddof=1))

    x = np.linspace(mu_hat - 4 * sigma_hat, mu_hat + 4 * sigma_hat, 400)

    plt.figure(figsize=(7, 4))
    plt.hist(S, bins=40, density=True, alpha=0.6, color="#D67615", edgecolor="k",
             label="ResNet outputs")
    plt.plot(x, norm.pdf(x, loc=mu_hat, scale=sigma_hat), color="k", lw=2,
             label="Normal fit")
    plt.xlabel("GAP",fontsize=18)
    plt.ylabel("Density")
    plt.title("FCN output distribution",fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.gca().set_yticklabels([])   # <-- before show
    plt.savefig("Fig1_density.eps", format="eps", bbox_inches="tight")
    plt.show()


# ============================================================
# 7. Repeat p-value computation N times + plots
# ============================================================
def pvalue_uniformity_study(
    N=200,
    n=1000,
    theta=0.6,
    B=1,
    n_rep=2000,
    sigma=1.0,
    base_seed=1,
    pick_run=0,
    do_normal_plot=True
):
    pvals = np.empty(N)
    S_pick = None

    for i in range(N):
        np.random.seed(base_seed + i)
        S = run_experiment(n=n, theta=theta, B=B, n_rep=n_rep, sigma=sigma)

        if i == pick_run:
            S_pick = S.copy()

        _, p = shapiro(S)
        pvals[i] = p

    # --- Uniformity histogram of p-values ---
    plt.figure(figsize=(7, 4))
    plt.hist(pvals, bins=20, density=True, alpha=0.6, color="#D67615", edgecolor="k",
             label="empirical p-values")
    plt.plot([0, 1], [1, 1], "k-", lw=2, label="Uniform(0,1) density")  # <-- FIX
    plt.xlim(0, 1)
    plt.xlabel("p-value",fontsize=18)
    plt.ylabel("Density",fontsize=18)
    plt.title(f"Shapiro p-values", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.gca().set_yticklabels([])   # <-- before show
    plt.savefig("Fig1_pvalues_hist.eps", format="eps", bbox_inches="tight")
    plt.show()

    # --- QQ-plot for p-values ---
    u = np.sort(pvals)
    q = (np.arange(1, N + 1) - 0.5) / N
    plt.figure(figsize=(5, 5))
    plt.plot(q, u, "ko", markersize=4, alpha=0.8)  # <-- FIX (black points)
    plt.plot([0, 1], [0, 1], "k-", lw=2)            # <-- FIX
    plt.xlabel("Uniform(0,1) theoretical quantiles",fontsize=18)
    plt.ylabel("Empirical p-value quantiles",fontsize=18)
    plt.title("QQ-plot",fontsize=20)
    plt.tight_layout()
    plt.savefig("Fig1_QQ.eps", format="eps", bbox_inches="tight")
    plt.show()

    # --- KS test vs Uniform ---
    ks_stat, ks_p = kstest(pvals, "uniform")
    print("\nUniformity diagnostics")
    print("----------------------")
    print(f"KS test vs Uniform(0,1): statistic={ks_stat:.4f}, p-value={ks_p:.4g}")
    print(f"Mean(p)={pvals.mean():.3f}, Var(p)={pvals.var(ddof=1):.3f} (Uniform: mean 0.5, var 1/12≈0.083)")

    if do_normal_plot and S_pick is not None:
        plot_normal_fit(S_pick, title_prefix=f"Run #{pick_run} (seed={base_seed + pick_run})")

    return pvals


# ============================================================
# MAIN
# ============================================================
def main():
    pvalue_uniformity_study(
        N=1000,  # number of times I repeat (hist of the p-value)
        n=1000, #time series length
        theta=0.4,
        B=4,
        n_rep=1000, #hist of the dist
        sigma=2.0,
        base_seed=1,
        pick_run=0,
        do_normal_plot=True
    )

if __name__ == "__main__":
    main()
