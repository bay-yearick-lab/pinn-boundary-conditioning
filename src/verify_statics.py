"""Verify Proposition 1 (the leak) claim by claim.

Checks, on 1D and 2D Poisson feature models in the misfit regime:
  C1: Delta-formula: w_lam - w* = (A^T A + lam B^T B)^{-1} B^T y*  (exact).
  C2: lam * (B w_lam - g) -> y* as lam -> infinity (ratio -> 1).
  C3: perturbed KKT holds exactly: A^T(A w_lam - f) + B^T y_lam = 0
      with y_lam = lam (B w_lam - g).
  C4: leak norm = rho/lam * (1 + O(1/lam)): fitted log-log slope = -1 and
      fitted intercept matches rho = ||y*|| quantitatively.
  C5: realizable regime: rho ~ 0 and the leak vanishes at every lam.
"""

import numpy as np
from problems import (build_poisson_1d, build_poisson_1d_eig, build_poisson_2d,
                      constrained_solution, penalized_solution)

PASS = True


def check(name, ok, detail=""):
    global PASS
    status = "PASS" if ok else "FAIL"
    if not ok:
        PASS = False
    print(f"[{status}] {name} {detail}")


def run(problem, tag):
    A, f, B, g = problem["A"], problem["f"], problem["B"], problem["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = np.linalg.norm(y_star)
    print(f"\n== {tag}: n={problem['n']} m={problem['m']} b={problem['b']} "
          f"rho={rho:.6e} ||r*||={np.linalg.norm(r_star):.6e}")

    lams = np.logspace(1, 8, 15)
    leaks, ratios = [], []
    for lam in lams:
        w_lam, H = penalized_solution(A, f, B, g, lam)
        # C1 exact Delta formula
        delta_pred = np.linalg.solve(H, B.T @ y_star)
        err_c1 = np.linalg.norm((w_lam - w_star) - delta_pred) / (np.linalg.norm(w_lam - w_star) + 1e-300)
        # C3 perturbed KKT
        y_lam = lam * (B @ w_lam - g)
        kkt = np.linalg.norm(A.T @ (A @ w_lam - f) + B.T @ y_lam)
        scale = np.linalg.norm(A.T @ f) + 1e-300
        leak = np.linalg.norm(B @ w_lam - g)
        leaks.append(leak)
        ratios.append(np.linalg.norm(y_lam - y_star) / (rho + 1e-300))
        # exact-algebra checks at moderate lam only: beyond ~1e6 the penalized
        # solve itself is limited by kappa(H) * eps_mach (which is the point).
        if lam in (lams[0], lams[6]):
            check(f"C1 Delta-formula lam={lam:.1e}", err_c1 < 1e-5, f"relerr={err_c1:.2e}")
            check(f"C3 perturbed KKT lam={lam:.1e}", kkt / scale < 1e-8, f"rel={kkt/scale:.2e}")
    leaks = np.array(leaks)
    # C2: multiplier estimate converges like 1/lam (compare mid-decade values)
    i5 = np.argmin(np.abs(np.log10(lams) - 5.0))
    check("C2 lam*(Bw-g) -> y*", ratios[i5] < 1e-2, f"relerr at lam=1e5: {ratios[i5]:.2e}")
    # C4: leak = rho/lam quantitatively. Fit in the top decades, where the
    # 1/lam asymptote holds (the crossover scale is set by the spectrum of
    # A^T A relative to B^T B and can reach ~1e5); the boundary residual is a
    # difference of O(1) quantities, so float64 resolves it far below 1e-8.
    mask = lams >= 1e6
    slope, intercept = np.polyfit(np.log10(lams[mask]), np.log10(leaks[mask]), 1)
    rho_fit = 10 ** intercept
    check("C4 slope == -1", abs(slope + 1.0) < 0.02, f"slope={slope:.4f}")
    check("C4 intercept == rho", abs(rho_fit / rho - 1.0) < 0.05,
          f"fit={rho_fit:.4e} vs rho={rho:.4e} ratio={rho_fit/rho:.4f}")
    check("C4 lam*leak/rho -> 1 at top", abs(lams[-1] * leaks[-1] / rho - 1.0) < 0.02,
          f"ratio={lams[-1]*leaks[-1]/rho:.4f}")
    return rho


def run_free_lifting():
    # Same misfit target, but the basis now contains {1, x}: free lifting.
    prob = build_poisson_1d(n_feat=24, m=200, b=2, freq=2.5, lifting=True)
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = np.linalg.norm(y_star)
    print(f"\n== free-lifting 1D (misfit target, {{1,x}} in span): "
          f"rho={rho:.3e} ||r*||={np.linalg.norm(r_star):.3e}")
    # The direct test that the shadow price is zero: perturb the boundary data
    # and finite-difference the optimal interior objective. With a free
    # lifting the response is ~0; without one it equals |y*_i|.
    delta = 1e-4
    J0 = 0.5 * np.linalg.norm(r_star) ** 2
    fd = []
    for i in range(len(g)):
        gp = g.copy(); gp[i] += delta
        _, _, r_p, _ = constrained_solution(A, f, B, gp)
        fd.append((0.5 * np.linalg.norm(r_p) ** 2 - J0) / delta)
    fd = np.linalg.norm(fd)
    check("C0 finite-difference shadow price == 0", fd < 1e-4 * np.sqrt(2 * J0),
          f"|dJ/dg|={fd:.2e} vs sqrt(2 J0)={np.sqrt(2*J0):.2e}")
    leaks = []
    for lam in (1e0, 1e2, 1e4):
        w_lam, _ = penalized_solution(A, f, B, g, lam)
        leaks.append(np.linalg.norm(B @ w_lam - g) / np.linalg.norm(g))
    check("C0 relative leak == 0 at every lam", max(leaks) < 1e-8,
          f"max rel leak={max(leaks):.2e}")


def run_realizable():
    # Overparameterized 1D: many features, few points => interpolation possible.
    prob = build_poisson_1d(n_feat=120, m=40, b=2, freq=2.0, seed=1, lifting=True,
                            bump=False)  # integer freq, no bump: target in span
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = np.linalg.norm(y_star)
    print(f"\n== realizable 1D: n={prob['n']} m={prob['m']} rho={rho:.3e} "
          f"||r*||={np.linalg.norm(r_star):.3e}")
    check("C5 rho ~ 0 when realizable", rho < 1e-8, f"rho={rho:.2e}")
    leaks = []
    for lam in (1e1, 1e3, 1e5):
        w_lam, _ = penalized_solution(A, f, B, g, lam)
        leaks.append(np.linalg.norm(B @ w_lam - g))
    check("C5 leak ~ 0 at every lam", max(leaks) < 1e-7, f"max leak={max(leaks):.2e}")


if __name__ == "__main__":
    np.random.seed(0)
    run(build_poisson_1d_eig(), "misfit 1D Poisson (eigenfunction basis)")
    run(build_poisson_2d(), "misfit 2D Poisson (tensor Fourier)")
    run_free_lifting()
    run_realizable()
    print("\nALL PASS" if PASS else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if PASS else 1)
