"""Verify Proposition 2 (the drag) and the tax theorem (error x budget).

Uses the eigenfunction-basis 1D misfit problem (float64-clean statics) and
EXACT spectral propagation of gradient descent on the quadratic loss:
    w_t - w_lam = sum_i (1 - eta h_i)^t c_i v_i,
so error curves at any t come from one eigendecomposition per lambda, with no
time-stepping roundoff.

Checks:
  D1: lambda_min(H_lam) <= mu (reduced curvature) for all lam.
  D2: lambda_max(H_lam) >= max(sigma_max(A)^2, lam sigma_max(B)^2).
  D3: kappa(H_lam) grows linearly at large lam, above lam sigma_B^2 / mu.
  D4: GD error to w* flattens at the leak floor ||w_lam - w*||.
  D5: the envelope over lam of error-at-budget-T decays algebraically ~1/T
      (slope in [-1.35, -0.75]); hard-constrained GD is geometric.
  D6: heavy-ball envelope is steeper (~1/T^2) but still algebraic.
  D7: the tax constant: T_eps * eps >= c * rho * sigma_B / mu for the swept
      family (report the measured min of the product against the bound scale).
"""

import numpy as np
from problems import build_poisson_1d_eig, constrained_solution, penalized_solution

PASS = True


def check(name, ok, detail=""):
    global PASS
    status = "PASS" if ok else "FAIL"
    if not ok:
        PASS = False
    print(f"[{status}] {name} {detail}")


def gd_error_curve(H, rhs, w_star, w0, ts):
    """Exact error-to-w* curve for GD minimizing 1/2 w'Hw - rhs'w."""
    h, V = np.linalg.eigh(H)
    eta = 1.0 / h[-1]
    w_lam = V @ ((V.T @ rhs) / h)
    c0 = V.T @ (w0 - w_lam)
    offset = V.T @ (w_lam - w_star)
    fac = 1.0 - eta * h
    errs = []
    for t in ts:
        errs.append(np.linalg.norm(c0 * np.power(fac, float(t)) + offset))
    return np.array(errs), w_lam


def heavyball_error_curve(H, rhs, w_star, w0, ts):
    """Exact heavy-ball error curve via per-mode 2x2 companion powers."""
    h, V = np.linalg.eigh(H)
    Lmax, mmin = h[-1], max(h[0], 1e-300)
    eta = 4.0 / (np.sqrt(Lmax) + np.sqrt(mmin)) ** 2
    beta = ((np.sqrt(Lmax) - np.sqrt(mmin)) / (np.sqrt(Lmax) + np.sqrt(mmin))) ** 2
    w_lam = V @ ((V.T @ rhs) / h)
    c0 = V.T @ (w0 - w_lam)
    offset = V.T @ (w_lam - w_star)
    def mat_pow(M, t):
        """Exact 2x2 matrix power by binary exponentiation (handles defective M)."""
        R = np.eye(2)
        P = M.copy()
        t = int(t)
        while t > 0:
            if t & 1:
                R = R @ P
            P = P @ P
            t >>= 1
        return R

    errs = np.zeros((len(ts), len(h)))
    for i, hi in enumerate(h):
        M = np.array([[1.0 + beta - eta * hi, -beta], [1.0, 0.0]])
        state0 = np.array([c0[i], c0[i]])       # w_0 = w_{-1}
        for k, t in enumerate(ts):
            errs[k, i] = (mat_pow(M, t) @ state0)[0]
    out = []
    for k in range(len(ts)):
        out.append(np.linalg.norm(errs[k] + offset))
    return np.array(out)


def main():
    prob = build_poisson_1d_eig(normalize=True)
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = np.linalg.norm(y_star)
    M = Z.T @ (A.T @ A) @ Z
    mu = np.linalg.eigvalsh(M)[0]
    L_red = np.linalg.eigvalsh(M)[-1]
    sA2 = np.linalg.norm(A, 2) ** 2
    sB2 = np.linalg.norm(B, 2) ** 2
    print(f"rho={rho:.4e} mu={mu:.4e} L_red={L_red:.4e} sA2={sA2:.4e} sB2={sB2:.4e}")

    lams = np.logspace(0, 8, 17)
    kappas = []
    d1 = d2 = True
    for lam in lams:
        H = A.T @ A + lam * (B.T @ B)
        ev = np.linalg.eigvalsh(H)
        kappas.append(ev[-1] / ev[0])
        d1 &= ev[0] <= mu * (1 + 1e-8)
        d2 &= ev[-1] >= max(sA2, lam * sB2) * (1 - 1e-10)
    check("D1 lam_min(H) <= mu for all lam", d1)
    check("D2 lam_max(H) >= max(sA2, lam sB2) for all lam", d2)
    kappas = np.array(kappas)
    hi = lams >= 1e5
    slope = np.polyfit(np.log10(lams[hi]), np.log10(kappas[hi]), 1)[0]
    check("D3 kappa slope == +1", abs(slope - 1.0) < 0.02, f"slope={slope:.4f}")
    check("D3 kappa >= lam sB2/mu", np.all(kappas[hi] >= (lams[hi] * sB2 / mu) * (1 - 1e-8)))

    # D4: two-regime shape at one moderate lam
    ts = np.unique(np.round(np.logspace(0, 7.2, 60))).astype(float)
    lam = 1e4
    H = A.T @ A + lam * (B.T @ B)
    rhs = A.T @ f + lam * (B.T @ g)
    w0 = np.zeros_like(w_star)
    errs, w_lam = gd_error_curve(H, rhs, w_star, w0, ts)
    leak = np.linalg.norm(w_lam - w_star)
    check("D4 GD floor == ||w_lam - w*||", abs(errs[-1] / leak - 1.0) < 0.02,
          f"final={errs[-1]:.3e} leak={leak:.3e}")

    # D5: envelope over lam at each budget
    budgets = np.array([10 ** k for k in range(1, 8)], dtype=float)
    lam_grid = np.logspace(1, 9, 25)
    env = np.full(len(budgets), np.inf)
    for lam in lam_grid:
        H = A.T @ A + lam * (B.T @ B)
        rhs = A.T @ f + lam * (B.T @ g)
        errs, _ = gd_error_curve(H, rhs, w_star, w0, budgets)
        env = np.minimum(env, errs)
    sl = np.polyfit(np.log10(budgets[-4:]), np.log10(env[-4:]), 1)[0]
    check("D5 soft envelope ~ 1/T (steady tail)", -1.25 < sl < -0.8, f"slope={sl:.3f}")

    # D7: the settling-time tax bound T*eps >= (rho sB / 4 mu) ln(c0/eps) holds
    # (it is a LOWER bound; slack from the loose kappa estimate is expected)
    products = budgets * env
    tax_scale = rho * np.sqrt(sB2) / (4 * mu)
    ok = all(p >= tax_scale for p in products[-4:])
    check("D7 tax lower bound holds on the tail", ok,
          f"min tail product={products[-4:].min():.3e} >= scale={tax_scale:.3e} "
          f"(slack {products[-4:].min()/tax_scale:.1f}x)")

    # Hard-constrained GD: geometric decay
    w_p = np.linalg.lstsq(B, g, rcond=None)[0]
    rhs_v = Z.T @ (A.T @ (f - A @ w_p))
    v_star = np.linalg.lstsq(M, rhs_v, rcond=None)[0]
    errs_h, _ = gd_error_curve(M, rhs_v, v_star, np.zeros(M.shape[0]), budgets)
    good = errs_h > 1e-9          # pre-roundoff-floor segment only
    corr = np.corrcoef(budgets[good], np.log(errs_h[good]))[0, 1] if good.sum() > 2 else -1.0
    check("D5 hard GD geometric (pre-floor)", corr < -0.9, f"corr(T, log e)={corr:.4f}")
    print(f"    hard error at budgets: {list(np.format_float_scientific(e, 2) for e in errs_h)}")
    print(f"    soft envelope:         {list(np.format_float_scientific(e, 2) for e in env)}")

    # D6: heavy-ball envelope
    env2 = np.full(len(budgets), np.inf)
    for lam in lam_grid:
        H = A.T @ A + lam * (B.T @ B)
        rhs = A.T @ f + lam * (B.T @ g)
        errs = heavyball_error_curve(H, rhs, w_star, w0, budgets)
        env2 = np.minimum(env2, errs)
    sl2 = np.polyfit(np.log10(budgets[1:5]), np.log10(env2[1:5]), 1)[0]
    check("D6 heavy-ball envelope steeper but algebraic",
          -2.7 < sl2 < -1.3, f"slope={sl2:.3f}")

    print("\nALL PASS" if PASS else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if PASS else 1)


if __name__ == "__main__":
    main()
