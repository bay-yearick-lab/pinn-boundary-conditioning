"""Verify the strengthened, exact, unconditional laws before they enter the paper.

All claims below assume only (A1) full-row-rank B and (A2) reduced positive
definiteness. No asymptotics, no algorithm, no genericity.

  X1 (two-sided leak):   rho/(lam+nu) <= ||B w_lam - g|| <= rho/lam, all lam>0,
                         with nu = ||N||_2 the boundary Schur curvature.
  X2 (monotone ledger):  the leak is strictly decreasing in lam; the multiplier
                         estimate ||y_lam|| = lam*leak is strictly increasing
                         and never exceeds rho.
  X3 (Pareto slope):     along the path, d||Aw-f||^2 / d||Bw-g||^2 = -lam
                         exactly; the supported tradeoff curve has slope -lambda.
  X4 (envelope law):     J(lam) = min_w L_lam is concave increasing with
                         J'(lam) = leak^2 / 2 exactly.
  X5 (leak--conditioning product): for every lam >= nu,
                         leak(lam) * kappa(H_lam) >= rho * sigma_B^2 / (2 mu).
  X6 (momentum family):  for every stable constant heavy-ball pair (eta, beta)
                         (GD is beta=0), the slow-mode companion spectral
                         radius is >= 1 - sqrt(eta h_s).
  X7 (inactivity equivalence): rho = 0  iff  A^T r* = 0  iff  leak == 0 at
                         every lam (checked across the three testbeds).
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


def schur_N(A, B, Z):
    """Boundary Schur curvature N = (AY)^T (I - AZ M^{-1} (AZ)^T) AY."""
    Y = B.T @ np.linalg.inv(B @ B.T)
    AZ, AY = A @ Z, A @ Y
    M = AZ.T @ AZ
    P = AY - AZ @ np.linalg.solve(M, AZ.T @ AY)
    return AY.T @ P


def run(prob, tag):
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = np.linalg.norm(y_star)
    N = schur_N(A, B, Z)
    nu = np.linalg.norm(N, 2)
    M = Z.T @ (A.T @ A) @ Z
    mu = np.linalg.eigvalsh(M)[0]
    sB2 = np.linalg.norm(B, 2) ** 2
    print(f"\n== {tag}: rho={rho:.4e} nu={nu:.4e} mu={mu:.4e} sB2={sB2:.4e}")

    lams = np.logspace(-2, 8, 41)
    leaks, ys, Ps, Qs, Js, kaps = [], [], [], [], [], []
    for lam in lams:
        w_lam, H = penalized_solution(A, f, B, g, lam)
        q = np.linalg.norm(B @ w_lam - g)
        leaks.append(q)
        ys.append(lam * q)
        Ps.append(np.linalg.norm(A @ w_lam - f) ** 2)
        Qs.append(q ** 2)
        Js.append(0.5 * Ps[-1] + 0.5 * lam * Qs[-1])
        ev = np.linalg.eigvalsh(H)
        kaps.append(ev[-1] / ev[0])
    leaks, ys = np.array(leaks), np.array(ys)
    Ps, Qs, Js, kaps = map(np.array, (Ps, Qs, Js, kaps))

    # X1 two-sided, every lam
    lo = rho / (lams + nu)
    hi = rho / lams
    ok1 = np.all(leaks <= hi * (1 + 1e-6)) and np.all(leaks >= lo * (1 - 1e-6))
    check("X1 rho/(lam+nu) <= leak <= rho/lam (all lam)", ok1,
          f"max hi-viol={np.max(leaks/hi - 1):.1e} max lo-viol={np.max(lo/leaks - 1):.1e}")

    # X2 monotone ledger
    check("X2 leak strictly decreasing", np.all(np.diff(leaks) < 0))
    check("X2 rho lam/(lam+nu) <= ||y_lam|| <= rho (all lam), nondecreasing",
          np.all(ys >= rho * lams / (lams + nu) * (1 - 1e-6))
          and np.all(ys <= rho * (1 + 1e-6))
          and np.all(np.diff(ys) > -1e-6 * rho),
          f"max ||y_lam||/rho = {ys.max()/rho:.6f}")

    # X3, X4: derivatives by DENSE local central differences (rel. step 1e-3)
    def path_vals(lam):
        w_lam, _ = penalized_solution(A, f, B, g, lam)
        q2 = np.linalg.norm(B @ w_lam - g) ** 2
        p2 = np.linalg.norm(A @ w_lam - f) ** 2
        return p2, q2, 0.5 * p2 + 0.5 * lam * q2

    worst3 = worst4 = 0.0
    for lam0 in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        h = 1e-3 * lam0
        Pp, Qp, Jp = path_vals(lam0 + h)
        Pm, Qm, Jm = path_vals(lam0 - h)
        P0, Q0, J0 = path_vals(lam0)
        slope = (Pp - Pm) / (Qp - Qm)
        worst3 = max(worst3, abs(slope + lam0) / lam0)
        dJ = (Jp - Jm) / (2 * h)
        worst4 = max(worst4, abs(dJ - 0.5 * Q0) / max(0.5 * Q0, 1e-300))
    check("X3 dP/dQ == -lam (local stencils)", worst3 < 1e-4,
          f"max rel err={worst3:.2e}")
    check("X4 J'(lam) == leak^2/2 (local stencils)", worst4 < 1e-4,
          f"max rel err={worst4:.2e}")
    dJg = np.gradient(Js, lams)
    check("X4 J concave increasing", np.all(dJg > 0) and np.all(np.diff(dJg) < 1e-12))

    # X5 leak--conditioning product
    mask = lams >= nu
    prod = leaks[mask] * kaps[mask]
    bound = rho * sB2 / (2 * mu)
    check("X5 leak * kappa >= rho sB^2/(2 mu) for lam >= nu",
          np.all(prod >= bound * (1 - 1e-6)),
          f"min product={prod.min():.4e} bound={bound:.4e} slack={prod.min()/bound:.2f}x")
    # X5b: the product CONVERGES to rho sB^2 / mu (exact large-weight limit)
    limit = rho * sB2 / mu
    check("X5b leak * kappa -> rho sB^2/mu (top of sweep, 1%)",
          abs(leaks[-1] * kaps[-1] / limit - 1.0) < 0.01,
          f"product={leaks[-1]*kaps[-1]:.6e} limit={limit:.6e} "
          f"ratio={leaks[-1]*kaps[-1]/limit:.4f}")
    return dict(rho=rho, nu=nu)


def run_matrix_fence():
    """X9: matrix-weight fence bound. kappa_eff(A'A + B'WB) >= kappa_red for
    every positive-definite W (pointwise/self-adaptive weights included).
    Random PD W's: diagonal (pointwise weights) and full."""
    rng = np.random.default_rng(21)
    viol, total, worst = 0, 0, np.inf
    def kappa_eff(H, tol=1e-11):
        ev = np.linalg.eigvalsh(H)
        nz = ev[ev > ev[-1] * tol]
        return ev[-1] / nz[0]
    for trial in range(120):
        n = rng.integers(8, 50); m = rng.integers(4, 70); b = rng.integers(1, min(n - 2, 8))
        A = rng.normal(size=(m, n)) * np.exp(rng.normal(0, 1.5, size=(1, n)))
        B = rng.normal(size=(b, n))
        _, sv, Vt = np.linalg.svd(B, full_matrices=True)
        Z = Vt[b:].T
        evr = np.linalg.eigvalsh(Z.T @ (A.T @ A) @ Z)
        nzr = evr[evr > evr[-1] * 1e-11]
        kred = evr[-1] / nzr[0]
        for _ in range(10):
            if rng.random() < 0.5:
                W = np.diag(np.exp(rng.normal(0, 2.0, size=b)))       # pointwise
            else:
                G = rng.normal(size=(b, b))
                W = G @ G.T + 1e-6 * np.eye(b)                        # full PD
            k = kappa_eff(A.T @ A + B.T @ W @ B)
            total += 1
            ratio = k / kred
            worst = min(worst, ratio)
            if ratio < 1 - 1e-6:
                viol += 1
    check("X9 matrix-weight fence bound (1200 PD W pairs)", viol == 0,
          f"violations={viol} worst ratio={worst:.6f}")
    # X9b: interior reweighting W_A moves the fence; the bound holds against
    # the correspondingly weighted fence kappa_red(W_A^{1/2} A)
    viol2, total2, worst2 = 0, 0, np.inf
    for trial in range(60):
        n = rng.integers(8, 40); m = rng.integers(6, 60); b = rng.integers(1, min(n - 2, 6))
        A = rng.normal(size=(m, n)); B = rng.normal(size=(b, n))
        _, sv, Vt = np.linalg.svd(B, full_matrices=True)
        Z = Vt[b:].T
        for _ in range(8):
            WA = np.diag(np.exp(rng.normal(0, 1.5, size=m)))
            WB = np.diag(np.exp(rng.normal(0, 1.5, size=b)))
            At = np.sqrt(WA) @ A
            evr = np.linalg.eigvalsh(Z.T @ (At.T @ At) @ Z)
            nzr = evr[evr > evr[-1] * 1e-11]
            kred_w = evr[-1] / nzr[0]
            k = kappa_eff(At.T @ At + B.T @ WB @ B)
            total2 += 1
            ratio = k / kred_w
            worst2 = min(worst2, ratio)
            if ratio < 1 - 1e-6:
                viol2 += 1
    check("X9b interior-reweighted fence bound (480 pairs)", viol2 == 0,
          f"violations={viol2} worst ratio={worst2:.6f}")
    # X9c: scalar-weight stress test at the Appendix B spec. Kernel accounting
    # must be exact: ker(H_lambda) = ker A intersect ker B for every lambda > 0,
    # so its dimension is n - rank([A; B]) independently of lambda; thresholding
    # eigenvalues relative to the top misclassifies interior modes once lambda
    # stretches the spectrum, producing false violations.
    viol3, total3, worst3 = 0, 0, np.inf
    rng3 = np.random.default_rng(37)
    while total3 < 3000:
        n = int(rng3.integers(8, 61)); m = int(rng3.integers(4, 81))
        b = int(rng3.integers(1, 9))
        if b >= n - 1:
            continue
        A = rng3.normal(size=(m, n)) * np.exp(rng3.normal(0, 1.5, size=(1, n)))
        B = rng3.normal(size=(b, n))
        _, sv, Vt = np.linalg.svd(B, full_matrices=True)
        Z = Vt[b:].T
        k0 = n - np.linalg.matrix_rank(np.vstack([A, B]))
        evr = np.linalg.eigvalsh(Z.T @ (A.T @ A) @ Z)
        k0r = Z.shape[1] - np.linalg.matrix_rank(A @ Z)
        kred = evr[-1] / evr[k0r]
        lam = 10.0 ** rng3.uniform(-6, 8)
        ev = np.linalg.eigvalsh(A.T @ A + lam * (B.T @ B))
        keff = ev[-1] / ev[k0]
        total3 += 1
        ratio = keff / kred
        worst3 = min(worst3, ratio)
        if ratio < 1 - 1e-6:
            viol3 += 1
    check("X9c scalar fence stress (3000 pairs, exact kernels)", viol3 == 0,
          f"violations={viol3} worst ratio={worst3:.6f}")


def run_lookahead_momentum_grid(prob):
    """X6b: look-ahead (accelerated) constant momentum obeys the same slow-mode floor
    1 - sqrt(eta h_s). Companion: z^2 - (1+beta)(1-eta h) z + beta(1-eta h)."""
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    lam = 1e4
    _, H = penalized_solution(A, f, B, g, lam)
    ev = np.linalg.eigvalsh(H)
    h_s, h_max = ev[0], ev[-1]
    worst = np.inf
    viol = 0
    for beta in np.linspace(0.0, 0.999, 60):
        for frac in np.linspace(0.02, 1.0, 50):
            eta = frac / h_max                     # stability: eta h <= 1 branch
            x = eta * h_s
            r = np.max(np.abs(np.roots([1.0, -(1 + beta) * (1 - x), beta * (1 - x)])))
            bound = 1 - np.sqrt(x)
            worst = min(worst, r - bound)
            if r < bound - 1e-12:
                viol += 1
    check("X6b look-ahead momentum companion radius >= 1 - sqrt(eta h_s) (3000 pairs)",
          viol == 0, f"min slack={worst:.3e}")


def run_noise_calibrated_penalty():
    """X8: the noise-calibrated penalty. Data model f = A w + noise(sig_r), g = B w + noise(sig_b).

    (a) GLS optimality in the positive semidefinite order: Cov(w_lam) - Cov(w_lam*) is PSD for
        every lam on a grid, with lam* = sig_r^2/sig_b^2, using the exact
        covariance Cov = H^{-1}(sig_r^2 A'A + lam^2 sig_b^2 B'B) H^{-1}.
    (b) Monte Carlo risk curve: E||w_lam - w_true||^2 attains its minimum on
        the grid at lam* (within grid neighbors).
    """
    from problems import build_poisson_1d_eig
    prob = build_poisson_1d_eig(normalize=True)
    A, B = prob["A"], prob["B"]
    rng = np.random.default_rng(11)
    n = A.shape[1]
    sig_r, sig_b = 1e-2, 1e-3
    lam_star = sig_r ** 2 / sig_b ** 2          # = 100
    lams = np.logspace(0, 4, 17)

    def cov(lam):
        H = A.T @ A + lam * (B.T @ B)
        Hi = np.linalg.inv(H)
        return Hi @ (sig_r ** 2 * (A.T @ A) + lam ** 2 * sig_b ** 2 * (B.T @ B)) @ Hi

    C_star = cov(lam_star)
    worst = np.inf
    for lam in lams:
        d = np.linalg.eigvalsh(cov(lam) - C_star)
        worst = min(worst, d[0] / np.trace(C_star))
    check("X8a GLS optimality: Cov(lam) - Cov(lam*) PSD for all lam",
          worst > -1e-10, f"min rel eig={worst:.2e}")

    trials = 400
    w_true = rng.normal(size=n)
    risks = np.zeros(len(lams))
    for _ in range(trials):
        f = A @ w_true + sig_r * rng.normal(size=A.shape[0])
        g = B @ w_true + sig_b * rng.normal(size=B.shape[0])
        for i, lam in enumerate(lams):
            H = A.T @ A + lam * (B.T @ B)
            w = np.linalg.solve(H, A.T @ f + lam * (B.T @ g))
            risks[i] += np.sum((w - w_true) ** 2) / trials
    i_min = int(np.argmin(risks))
    i_star = int(np.argmin(np.abs(np.log10(lams / lam_star))))
    check("X8b Monte Carlo risk minimized at lam* (within one grid step)",
          abs(i_min - i_star) <= 1,
          f"argmin lam={lams[i_min]:.3g} vs lam*={lam_star:g}; "
          f"risk(lam*)/min={risks[i_star]/risks[i_min]:.4f}")


def run_momentum_grid(prob):
    """X6: slow-mode companion radius >= 1 - sqrt(eta h_s) over the family."""
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    lam = 1e4
    _, H = penalized_solution(A, f, B, g, lam)
    ev = np.linalg.eigvalsh(H)
    h_s, h_max = ev[0], ev[-1]
    worst = np.inf
    viol = 0
    for beta in np.linspace(0.0, 0.999, 60):
        for frac in np.linspace(0.02, 1.0, 50):
            eta = frac * 2 * (1 + beta) / h_max      # stability boundary
            x = eta * h_s
            r = np.max(np.abs(np.roots([1.0, -(1 + beta - eta * h_s), beta])))
            bound = 1 - np.sqrt(x)
            worst = min(worst, r - bound)
            if r < bound - 1e-12:
                viol += 1
    check("X6 companion radius >= 1 - sqrt(eta h_s) (3000 pairs)",
          viol == 0, f"min slack={worst:.3e}")


def run_equivalence():
    """X7: rho = 0 iff A^T r* = 0 iff zero leak at every lam."""
    probs = [
        ("misfit (rho>0)", build_poisson_1d_eig(normalize=True), False),
        ("free lifting (rho=0)", build_poisson_1d(n_feat=24, m=200, b=2,
                                                  freq=2.5, lifting=True), True),
    ]
    for tag, prob, expect_zero in probs:
        A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
        w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
        rho = np.linalg.norm(y_star)
        grad = np.linalg.norm(A.T @ r_star)
        scale = np.linalg.norm(A.T @ f) + 1e-300
        leaks = []
        for lam in (1e0, 1e3, 1e6):
            w_lam, _ = penalized_solution(A, f, B, g, lam)
            leaks.append(np.linalg.norm(B @ w_lam - g) / max(np.linalg.norm(g), 1e-300))
        is_zero = (rho / scale < 1e-8, grad / scale < 1e-8, max(leaks) < 1e-8)
        check(f"X7 [{tag}] three conditions agree ({'zero' if expect_zero else 'nonzero'})",
              all(z == expect_zero for z in is_zero),
              f"rho/s={rho/scale:.1e} |A'r*|/s={grad/scale:.1e} leak={max(leaks):.1e}")


if __name__ == "__main__":
    run(build_poisson_1d_eig(normalize=True), "misfit 1D (eig, normalized)")
    run(build_poisson_2d(), "misfit 2D (tensor Fourier)")
    run_momentum_grid(build_poisson_1d_eig(normalize=True))
    run_lookahead_momentum_grid(build_poisson_1d_eig(normalize=True))
    run_matrix_fence()
    run_noise_calibrated_penalty()
    run_equivalence()
    print("\nALL PASS" if PASS else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if PASS else 1)
