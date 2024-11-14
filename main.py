import numpy as np
import scipy as sp
import polars as pl

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



def load_ehec_data(treated):
    panel = (pl.read_csv("data/ehec_data.csv")
             .with_columns((pl.col("year") >= pl.col("yexp2")).alias("treated")))

    treatment = panel.pivot("year", index="stfips", values="treated").sort("stfips").drop("stfips").fill_null(False).to_numpy()
    observed = treatment == treated
    dins = panel.pivot("year", index="stfips", values="dins").sort("stfips").drop("stfips").to_numpy()

    return observed, dins


def laplacian(O):
    m, n = O.shape
    A = np.zeros((m + n, m + n))
    A[:m, m:] = O
    A[m:, :m] = O.T
    D = np.concatenate((np.sum(O, axis=1), np.sum(O, axis=0)))
    L = np.diag(D) - A
    return L


def effective_resistance(O):
    L = laplacian(O)

    m, n = O.shape
    L_inv = np.linalg.pinv(L)

    if np.linalg.matrix_rank(L) < m + n - 1:
        print("Graph is not connected!")

    L_inv_diag = np.diag(L_inv)
    eff_res = L_inv_diag[:m, np.newaxis] + L_inv_diag[m:] - 2 * L_inv[:m, m:]
    return eff_res


def treatment_effect(C, U):
    p1, p2 = C.shape
    U = np.atleast_3d(U)

    du = np.sum(C, axis=1)
    dv = np.sum(C, axis=0)
    M = C[:, :, np.newaxis] * np.where(C[:, :, np.newaxis] > 0, U, 0)
    yu = np.sum(M, axis=1)
    yv = np.sum(M, axis=0)

    S = np.diag(dv) - C.T @ (C/du[:, np.newaxis])
    SS = S + 1/p2
    c, low = sp.linalg.cho_factor(SS)
    beta_vv_s = sp.linalg.cho_solve((c, low), yv)
    beta_vv = beta_vv_s # - np.mean(beta_vv_s, axis=0, keepdims=True)
    beta_uv = -(C @ beta_vv)/du[:, np.newaxis]
    neg_beta_vu_s = sp.linalg.cho_solve((c, low), C.T @ (yu/du[:, np.newaxis]))
    neg_beta_vu = neg_beta_vu_s # - np.mean(neg_beta_vu_s, axis=0, keepdims=True)

    beta_vu = -neg_beta_vu
    beta_uu = (yu + C @ neg_beta_vu)/du[:, np.newaxis]
    beta_L = beta_uu + beta_uv
    beta_R = beta_vu + beta_vv

    return beta_L, beta_R


def plot_yhat(O, Y, Yhat, prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    masked = np.where(O, Y, np.nan)
    combined_data = np.stack([masked, Yhat])
    min_, max_ = np.nanmin(combined_data), np.nanmax(combined_data)

    cmap = mpl.colormaps.get_cmap("viridis")                                                                                                           
    cmap.set_bad(color="grey")

    sns.heatmap(masked, vmin=min_, vmax=max_, ax=ax1, cmap=cmap)
    ax1.set(title="Observed")
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.heatmap(Yhat, vmin=min_, vmax=max_, ax=ax2, cmap=cmap)
    ax2.set(title="Estimated")
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.savefig(f"figures/{prefix}_estimation.png")
    plt.close(fig)


def plot_eff_res(O, eff_res, prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    sns.heatmap(O, ax=ax1)
    ax1.set(title="Omega")
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.heatmap(eff_res, vmin=0, ax=ax2)
    ax2.set(title="Effective resistance")
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.savefig(f"figures/{prefix}_uncertainty.png")
    plt.close(fig)


def estimate(treated):
    observed, dins = load_ehec_data(treated)

    row_mask = np.flatnonzero(np.any(observed, axis=1))
    col_mask = np.flatnonzero(np.any(observed, axis=0))
    O = observed[np.ix_(row_mask, col_mask)]
    Y = dins[np.ix_(row_mask, col_mask)]

    beta_L, beta_R = treatment_effect(O, Y)
    Yhat = beta_L + np.squeeze(beta_R, -1)
    dins_hat = np.full_like(observed, np.nan, dtype=float)
    dins_hat[np.ix_(row_mask, col_mask)] = Yhat
    plot_yhat(observed, dins, dins_hat, treated)

    eff_res = np.full_like(observed, np.nan, dtype=float)
    eff_res[np.ix_(row_mask, col_mask)] = effective_resistance(O)
    plot_eff_res(observed, eff_res, treated)

    return dins_hat, eff_res


def plot_treatment_effect(treatment_effect_hat, eff_res):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    sns.heatmap(treatment_effect_hat, ax=ax1)
    ax1.set(title="Estimated treatment effect")
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.heatmap(eff_res, vmin=0, ax=ax2)
    ax2.set(title="Effective resistance")
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.savefig(f"figures/treatment_effect.png")
    plt.close(fig)


def regression():
    panel = (pl.read_csv("data/ehec_data.csv")
             .with_columns((pl.col("year") >= pl.col("yexp2")).fill_null(False).alias("treated")))

    X = panel.select(("year", "stfips", "treated")).to_dummies(("year", "stfips")).to_numpy()
    assert X.shape[-1] == panel["year"].n_unique() + panel["stfips"].n_unique() + 1
    Y = panel["dins"].to_numpy()

    beta_hat, *_ = np.linalg.lstsq(X, Y)
    return beta_hat[-1]


def main():
    # Estimate the homogeneous treatment effect directly with linear regression
    # Y_{it} = alpha_i + lambda_t + D_{it} beta + epsilon_{it}
    beta_hat_homogeneous = regression()
    print(f"{beta_hat_homogeneous.item() = }")

    # Estimate the heterogeneous treatment effects with linear regression
    # Y_{it} = alpha_i + lambda_t + D_{it} beta_{ij} + epsilon_{it}
    dins_hat_treated, eff_res_treated = estimate(True)
    dins_hat_control, eff_res_control = estimate(False)
    treatment_effect_hat = dins_hat_treated - dins_hat_control
    eff_res = eff_res_treated + eff_res_control
    plot_treatment_effect(treatment_effect_hat, eff_res)

    # Estimate the homogeneous treatment effect by averaging the heterogeneous treatment effects
    beta_hat_heterogeneous = np.nanmean(treatment_effect_hat)
    print(f"{beta_hat_heterogeneous.item() = }")


if __name__ == "__main__":
    main()
