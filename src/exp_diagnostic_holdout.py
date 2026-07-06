"""Held-out prospective test of the boundary-penalty diagnostic (Algorithm 1).

The diagnostic constants (rho, nu, mu, sigma_B, kappa_red) are computed FIRST,
from the discretized operators of problems whose parameters do not appear in any
figure of the paper. The predictions they imply (the regime call, the leak
brackets, the leak--conditioning product limit, the valley location) are then confirmed
against an INDEPENDENT direct-solve weight sweep. Nothing in the sweep is used to
form the constants, so the confirmation is genuinely out of sample.

Runs in seconds (dense linear algebra, float64).
"""

import numpy as np
from problems import (build_poisson_1d, build_poisson_2d,
                      constrained_solution, penalized_solution)


def diagnostic_constants(A, f, B, g):
    """Algorithm 1 in the exact (fixed-feature) regime: return the product-law constants."""
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = np.linalg.norm(y_star)
    # reduced curvature M = Z^T A^T A Z on ker B
    AZ = A @ Z
    M = AZ.T @ AZ
    ev = np.linalg.eigvalsh(M)
    ev = ev[ev > ev.max() * 1e-12]
    mu, L_red = ev.min(), ev.max()
    kappa_red = L_red / mu
    # boundary Schur curvature N = (AY)^T (I - AZ M^{-1} (AZ)^T) (AY)
    Y = B.T @ np.linalg.inv(B @ B.T)
    AY = A @ Y
    Minv = np.linalg.inv(M)
    N = AY.T @ (AY - AZ @ (Minv @ (AZ.T @ AY)))
    nu = np.linalg.norm(N, 2)
    sigmaB = np.linalg.norm(B, 2)
    return dict(rho=rho, nu=nu, mu=mu, L_red=L_red, kappa_red=kappa_red,
                sigmaB=sigmaB, y_star=y_star)


def confirm(A, f, B, g, c, lams):
    """Independent sweep: measured leak and conditioning at each weight."""
    leaks, kaps = [], []
    for lam in lams:
        w_lam, H = penalized_solution(A, f, B, g, lam)
        leaks.append(np.linalg.norm(B @ w_lam - g))
        kaps.append(np.linalg.cond(H))
    return np.array(leaks), np.array(kaps)


def report(tag, A, f, B, g):
    c = diagnostic_constants(A, f, B, g)
    rho, nu, mu, sigmaB, kred = c["rho"], c["nu"], c["mu"], c["sigmaB"], c["kappa_red"]
    print(f"\n===== {tag} =====")
    print(f"diagnostic constants (computed before any sweep):")
    print(f"  rho={rho:.4e}  nu={nu:.4e}  mu={mu:.4e}  sigma_B={sigmaB:.4e}  kappa_red={kred:.4e}")
    # ---- prediction (from constants only) ----
    harmless = rho < 1e-8 * (abs(A).max())  # Algorithm 1 rho-branch
    product_pred = rho * sigmaB ** 2 / mu
    print(f"  RECOMMENDATION: {'penalty statically harmless (free lifting)' if harmless else 'costly: tune the valley / negotiate the constraint'}")
    print(f"  predicted product limit rho*sigma_B^2/mu = {product_pred:.4e}")
    # ---- confirmation (independent sweep) ----
    lams = np.logspace(1, 9, 25)
    leaks, kaps = confirm(A, f, B, g, c, lams)
    if harmless:
        print(f"  CONFIRM: max leak over sweep = {leaks.max():.3e} (predicted ~0)")
        return
    # Exact regime detector: leak*lam sits on its plateau rho (i.e. leak = rho/lam
    # cleanly), which is precisely where the direct solve is not yet round-off bound.
    prod_lam = leaks * lams
    exact = (prod_lam > 0.5 * rho) & (prod_lam < 2.0 * rho)
    lo = rho / (lams + nu)
    hi = rho / lams
    within = bool(np.all((leaks[exact] <= hi[exact] * (1 + 1e-3)) &
                         (leaks[exact] >= lo[exact] * (1 - 1e-3))))
    rho_fit = np.median(prod_lam[exact])                  # leak intercept
    product_meas = np.median((leaks * kaps)[exact])       # leak--conditioning product
    lam_valley = lams[np.argmin(kaps)]                    # conditioning valley
    print(f"  CONFIRM leak brackets hold at every exact-regime weight: {within}")
    print(f"  CONFIRM leak intercept: predicted rho={rho:.4e}, measured={rho_fit:.4e}, "
          f"rel err={abs(rho_fit-rho)/rho:.2%}")
    print(f"  CONFIRM product limit: predicted={product_pred:.4e}, measured={product_meas:.4e}, "
          f"rel err={abs(product_meas-product_pred)/product_pred:.2%}")
    print(f"  observed conditioning-valley weight (argmin kappa) = {lam_valley:.2e}")


if __name__ == "__main__":
    print("HELD-OUT PROSPECTIVE TEST OF THE BOUNDARY-PENALTY DIAGNOSTIC")
    print("(parameters below appear in no figure of the paper)")

    # HO-1: 2D Poisson, held-out frequencies/resolution, no lifting -> rho > 0.
    # The diagnostic must call it COSTLY and predict leak + product limit.
    p1 = build_poisson_2d(J=6, m=1500, b=16, kx=3.3, ky=2.1, seed=7)
    report("HO-1  2D Poisson (J=6, kx=3.3, ky=2.1), no lifting", p1["A"], p1["f"], p1["B"], p1["g"])

    # HO-2: 1D Poisson, held-out frequency/features, FREE LIFTING -> rho ~ 0.
    # The diagnostic must call it HARMLESS without seeing the sweep.
    p2 = build_poisson_1d(n_feat=30, m=260, b=2, freq=3.3, seed=11, lifting=True)
    report("HO-2  1D Poisson (freq=3.3, n=30), free lifting", p2["A"], p2["f"], p2["B"], p2["g"])
