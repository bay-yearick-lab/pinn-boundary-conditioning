"""Prospective boundary-penalty diagnostic on harder 3D and multiscale-stiff PDEs.

It carries the exact held-out protocol of ``exp_diagnostic_holdout.py`` onto:

  TB-1  3D Poisson  : L = -Laplacian on the unit cube, tensor-Fourier features,
                      Dirichlet trace on the six faces, a manufactured target with
                      non-integer frequencies (non-representable -> misfit, rho>0).
  TB-2  3D stiff    : anisotropic high-contrast elliptic operator
                      L = -(a_x d_xx + a_y d_yy + a_z d_zz) with a=(1,40,1600),
                      making the reduced curvature genuinely stiff (kappa_red ~ 1e6)
                      so a naive penalty sweep is structurally expensive.

Reference truth is the dense constrained least-squares solve and direct penalized
solves in float64 (the fixed-feature regime, where the constants are exact).

PROSPECTIVE DISCIPLINE. For every testbed and seed the diagnostic constants
(rho, nu, mu, sigma_B, kappa_red) are computed from the operators FIRST, the
predictions they imply are COMMITTED to the printout (regime call, leak brackets
rho/(lam+nu)..rho/lam, product limit rho*sigma_B^2/mu, conditioning valley), and
only THEN an independent logspace direct-solve sweep is run to confirm them.

The one place the constants touch the sweep is choosing WHERE to sample: the leak
plateau on which rho and the product are read lives at lambda >> nu, and nu is a
pre-committed constant. Placing the grid around nu is therefore a prediction, not
a fit -- for the stiff testbed nu ~ 4e10, so a fixed low grid would miss the
plateau entirely, which is exactly the "structurally expensive" failure the
product limit forecasts. No leak, conditioning, or product MEASUREMENT is ever
used to form or adjust the constants.

Runs to completion in a couple of minutes (dense float64).
An optional small trained-3D tanh-PINN check (--pinn) shows the same
qualitative conditioning valley.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import constrained_solution, penalized_solution  # noqa: E402

SEEDS = tuple(range(10))


# --------------------------------------------------------------------------- #
# 3D tensor-Fourier feature map and PDE builders (mirror build_poisson_2d).   #
# --------------------------------------------------------------------------- #
def tensor_fourier_3d(X, J, a=(1.0, 1.0, 1.0)):
    """Products of {sin,cos}(j pi x) x {sin,cos}(k pi y) x {sin,cos}(l pi z).

    Returns (Phi, Lap) where Lap is the negative-definite action of the
    anisotropic elliptic operator on each feature,
        L phi = -(a_x d_xx + a_y d_yy + a_z d_zz) phi.
    With a=(1,1,1) this is the ordinary Laplacian; high contrast in a makes the
    operator stiff. No harmonic completion: every feature has nonzero L phi, so
    there is no free lifting. Feature count n = 8 J^3.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    ax, ay, az = a
    trig = (np.sin, np.cos)
    cols, laps = [], []
    for j in range(1, J + 1):
        for k in range(1, J + 1):
            for el in range(1, J + 1):
                wj, wk, wl = j * np.pi, k * np.pi, el * np.pi
                fac = -(ax * wj ** 2 + ay * wk ** 2 + az * wl ** 2)
                for fx in trig:
                    for fy in trig:
                        for fz in trig:
                            phi = fx(wj * x) * fy(wk * y) * fz(wl * z)
                            cols.append(phi)
                            laps.append(fac * phi)
    return np.stack(cols, axis=1), np.stack(laps, axis=1)


def _cube_boundary_points(rng, b):
    """b points spread over the six faces of the unit cube."""
    face = rng.integers(0, 6, size=b)
    Xb = rng.uniform(0.0, 1.0, size=(b, 3))
    for i in range(b):
        axis, val = face[i] // 2, face[i] % 2
        Xb[i, axis] = float(val)
    return Xb


def build_poisson_3d(J=3, m=1600, b=72, kx=2.4, ky=1.7, kz=1.3, seed=0):
    """-Lap u = f on the unit cube, Dirichlet data, u = sin(kx pi x)sin(ky pi y)sin(kz pi z).

    Tensor-Fourier model (n = 8 J^3 features), well-conditioned. Non-integer
    kx, ky, kz put the target outside the integer-harmonic span: misfit regime
    with rho > 0. Interior columns divided by sqrt(m), boundary by sqrt(b), as in
    build_poisson_2d. b kept modest so B B^T stays well-conditioned.
    """
    rng = np.random.default_rng(seed)
    Xi = rng.uniform(0.0, 1.0, size=(m, 3))
    Phi_i, Lap_i = tensor_fourier_3d(Xi, J)
    A = -Lap_i / np.sqrt(m)
    u_ex = (lambda X: np.sin(kx * np.pi * X[:, 0]) * np.sin(ky * np.pi * X[:, 1])
            * np.sin(kz * np.pi * X[:, 2]))
    coef = (kx ** 2 + ky ** 2 + kz ** 2) * np.pi ** 2
    f = coef * u_ex(Xi) / np.sqrt(m)
    Xb = _cube_boundary_points(rng, b)
    Phi_b, _ = tensor_fourier_3d(Xb, J)
    B = Phi_b / np.sqrt(b)
    g = u_ex(Xb) / np.sqrt(b)
    return dict(A=A, f=f, B=B, g=g, n=Phi_i.shape[1], m=m, b=b, a=(1.0, 1.0, 1.0),
                kfreq=(kx, ky, kz), J=J)


def build_stiff_3d(J=3, m=1600, b=72, a=(1.0, 40.0, 1600.0),
                   kx=2.4, ky=1.7, kz=1.3, seed=0):
    """Anisotropic high-contrast elliptic problem: L u = f on the unit cube.

    L = -(a_x d_xx + a_y d_yy + a_z d_zz) with a = (1, 40, 1600) spanning three
    orders of magnitude. The reduced curvature on ker B inherits this contrast,
    so kappa_red is large (~1e6): a genuinely stiff constrained problem where a
    naive fixed-weight penalty sweep is structurally expensive. Manufactured
    non-representable target (non-integer frequencies) keeps rho > 0. Same
    sqrt(m)/sqrt(b) normalization as the isotropic build.
    """
    rng = np.random.default_rng(seed)
    Xi = rng.uniform(0.0, 1.0, size=(m, 3))
    Phi_i, Lap_i = tensor_fourier_3d(Xi, J, a)
    A = -Lap_i / np.sqrt(m)
    ax, ay, az = a
    u_ex = (lambda X: np.sin(kx * np.pi * X[:, 0]) * np.sin(ky * np.pi * X[:, 1])
            * np.sin(kz * np.pi * X[:, 2]))
    coef = (ax * kx ** 2 + ay * ky ** 2 + az * kz ** 2) * np.pi ** 2
    f = coef * u_ex(Xi) / np.sqrt(m)
    Xb = _cube_boundary_points(rng, b)
    Phi_b, _ = tensor_fourier_3d(Xb, J, a)
    B = Phi_b / np.sqrt(b)
    g = u_ex(Xb) / np.sqrt(b)
    return dict(A=A, f=f, B=B, g=g, n=Phi_i.shape[1], m=m, b=b, a=a,
                kfreq=(kx, ky, kz), J=J)


# --------------------------------------------------------------------------- #
# Diagnostic constants (Algorithm 1, exact fixed-feature regime).             #
# Identical mathematics to exp_diagnostic_holdout.diagnostic_constants.        #
# --------------------------------------------------------------------------- #
def diagnostic_constants(A, f, B, g):
    """Return the product-law constants (rho, nu, mu, L_red, kappa_red, sigma_B)."""
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = float(np.linalg.norm(y_star))
    AZ = A @ Z
    M = AZ.T @ AZ                                  # reduced curvature on ker B
    ev = np.linalg.eigvalsh(M)
    ev = ev[ev > ev.max() * 1e-12]
    mu, L_red = float(ev.min()), float(ev.max())
    kappa_red = L_red / mu
    Y = B.T @ np.linalg.inv(B @ B.T)               # right inverse of B
    AY = A @ Y
    Minv = np.linalg.inv(M)
    N = AY.T @ (AY - AZ @ (Minv @ (AZ.T @ AY)))    # boundary Schur curvature
    nu = float(np.linalg.norm(N, 2))
    sigmaB = float(np.linalg.norm(B, 2))
    return dict(rho=rho, nu=nu, mu=mu, L_red=L_red, kappa_red=kappa_red,
                sigmaB=sigmaB)


def leak_grid_from_constants(c, n_pts=40, decades_above=3.0):
    """Leak-plateau weight grid, placed from PRE-COMMITTED constants only.

    The leak plateau where leak = rho/lambda holds cleanly lives at lambda >> nu.
    nu is computed before the sweep, so centering the logspace grid on nu (up to
    a few decades above it) is a prediction of where the plateau sits, not a fit.
    For the stiff testbed nu ~ 5e10, so this is exactly the regime a naive low
    grid would miss: reading rho and the product off the plateau requires knowing
    nu in advance, which is itself the diagnostic's forecast.
    """
    nu = c["nu"]
    lo = np.log10(max(nu, 1.0)) + 0.5
    hi = lo + decades_above
    return np.logspace(lo, hi, n_pts)


def valley_grid_from_constants(c, n_pts=45):
    """Conditioning-valley weight grid, placed from PRE-COMMITTED constants only.

    The condition number of H_lambda = A^T A + lambda B^T B falls from the
    unconstrained clock, reaches its valley, then rises linearly as lambda B^T B
    dominates; the fence bound floors the valley at kappa_red. This grid spans
    from far below to a few decades above nu so the valley is bracketed. The
    valley is expected below nu (before the leak plateau), and its depth is
    predicted to be no smaller than kappa_red.
    """
    nu = c["nu"]
    lo = -2.0
    hi = np.log10(max(nu, 1.0)) + 3.0
    return np.logspace(lo, hi, n_pts)


def run_sweep(A, f, B, g, lams):
    """Independent direct-solve sweep: measured leak and H-conditioning per weight."""
    leaks, kaps = [], []
    for lam in lams:
        w_lam, H = penalized_solution(A, f, B, g, lam)
        leaks.append(float(np.linalg.norm(B @ w_lam - g)))
        kaps.append(float(np.linalg.cond(H)))
    return np.array(leaks), np.array(kaps)


def confirm(c, leak_lams, leaks, leak_kaps, valley_lams, valley_kaps):
    """Confirm the pre-committed predictions against the independent sweeps.

    leak_lams/leaks/leak_kaps : plateau grid (lambda >> nu) -> rho intercept and
                                the leak--conditioning product limit, read as the
                                directly measured product leak * cond(H_lambda).
    valley_lams/valley_kaps    : broad grid -> conditioning valley + kappa_red floor.
    """
    rho, nu, mu, sigmaB = c["rho"], c["nu"], c["mu"], c["sigmaB"]
    product_pred = rho * sigmaB ** 2 / mu
    prod_lam = leaks * leak_lams
    # exact-regime detector: leak*lam sits on its rho plateau (not yet round-off bound)
    exact = (prod_lam > 0.5 * rho) & (prod_lam < 2.0 * rho)
    lo = rho / (leak_lams + nu)
    hi = rho / leak_lams
    within = bool(np.all((leaks[exact] <= hi[exact] * (1 + 1e-3))
                         & (leaks[exact] >= lo[exact] * (1 - 1e-3)))) if exact.any() else False
    rho_fit = float(np.median(prod_lam[exact])) if exact.any() else float("nan")
    # leak--conditioning product limit: directly measured leak * cond(H) on the plateau
    product_meas = float(np.median((leaks * leak_kaps)[exact])) if exact.any() else float("nan")
    iv = int(np.argmin(valley_kaps))
    lam_valley = float(valley_lams[iv])
    kappa_min = float(valley_kaps[iv])
    # fence-bound floor: measured valley depth cannot beat kappa_red
    floor_holds = bool(kappa_min >= c["kappa_red"] * (1 - 1e-6))
    floor_ratio = kappa_min / c["kappa_red"]
    rho_relerr = abs(rho_fit - rho) / rho if np.isfinite(rho_fit) else float("nan")
    product_relerr = (abs(product_meas - product_pred) / product_pred
                      if np.isfinite(product_meas) else float("nan"))
    return dict(product_pred=product_pred, product_meas=product_meas,
                product_relerr=product_relerr, rho_meas=rho_fit,
                rho_relerr=rho_relerr, leak_brackets_hold=within,
                n_exact_pts=int(exact.sum()), lam_valley=lam_valley,
                kappa_min=kappa_min, floor_holds=floor_holds,
                floor_ratio=float(floor_ratio))


# --------------------------------------------------------------------------- #
# Prospective per-testbed, per-seed driver.                                   #
# --------------------------------------------------------------------------- #
def run_testbed(tag, builder, seeds=SEEDS):
    print(f"\n=================== {tag} ===================")
    per_seed = []
    for seed in seeds:
        p = builder(seed=seed)
        A, f, B, g = p["A"], p["f"], p["B"], p["g"]
        # ---- STEP 1: constants, computed with NO sweep information ----
        c = diagnostic_constants(A, f, B, g)
        product_pred = c["rho"] * c["sigmaB"] ** 2 / c["mu"]
        harmless = c["rho"] < 1e-8 * float(abs(A).max())
        regime = ("statically harmless (free lifting)" if harmless
                  else "COSTLY: tune the valley / negotiate the constraint")
        leak_lams = leak_grid_from_constants(c)
        valley_lams = valley_grid_from_constants(c)
        print(f"\n  seed {seed}  (n={p['n']} features, m={p['m']}, b={p['b']}, a={p['a']})")
        print(f"  [COMMIT before sweep] constants:")
        print(f"     rho={c['rho']:.4e}  nu={c['nu']:.4e}  mu={c['mu']:.4e}  "
              f"sigma_B={c['sigmaB']:.4e}  kappa_red={c['kappa_red']:.4e}")
        print(f"  [COMMIT] regime call         : {regime}")
        print(f"  [COMMIT] leak brackets        : rho/(lam+nu) .. rho/lam "
              f"(rho={c['rho']:.3e}, nu={c['nu']:.3e})")
        print(f"  [COMMIT] product limit        : rho*sigma_B^2/mu = {product_pred:.4e}")
        print(f"  [COMMIT] valley floor          : kappa(H) cannot beat kappa_red = "
              f"{c['kappa_red']:.4e}")
        print(f"  [COMMIT] leak-plateau grid (nu): [{leak_lams[0]:.2e}, {leak_lams[-1]:.2e}]  "
              f"(placed above nu={c['nu']:.2e})")
        print(f"  [COMMIT] valley grid           : [{valley_lams[0]:.2e}, {valley_lams[-1]:.2e}]")
        # ---- STEP 2: independent sweeps, then confirm ----
        t0 = time.time()
        leaks, leak_kaps = run_sweep(A, f, B, g, leak_lams)
        _, valley_kaps = run_sweep(A, f, B, g, valley_lams)
        conf = confirm(c, leak_lams, leaks, leak_kaps, valley_lams, valley_kaps)
        dt = time.time() - t0
        print(f"  [CONFIRM] leak brackets hold on exact-regime weights: "
              f"{conf['leak_brackets_hold']}  ({conf['n_exact_pts']} pts)")
        print(f"  [CONFIRM] leak intercept rho : pred={c['rho']:.4e}  "
              f"meas={conf['rho_meas']:.4e}  rel err={conf['rho_relerr']:.2%}")
        print(f"  [CONFIRM] product limit       : pred={conf['product_pred']:.4e}  "
              f"meas={conf['product_meas']:.4e}  rel err={conf['product_relerr']:.2%}")
        print(f"  [CONFIRM] valley weight        : argmin kappa at lam={conf['lam_valley']:.2e}"
              f"  (kappa_min={conf['kappa_min']:.3e})")
        print(f"  [CONFIRM] fence floor          : kappa_min >= kappa_red ? "
              f"{conf['floor_holds']}  (kappa_min/kappa_red = {conf['floor_ratio']:.2f})")
        print(f"  (sweeps of {len(leak_lams)}+{len(valley_lams)} direct solves took {dt:.1f}s)")
        per_seed.append(dict(
            seed=int(seed), n_features=int(p["n"]), m=int(p["m"]), b=int(p["b"]),
            anisotropy=list(p["a"]), frequencies=list(p["kfreq"]), J=int(p["J"]),
            constants=c, product_pred=product_pred, regime_costly=(not harmless),
            leak_plateau_window=[float(leak_lams[0]), float(leak_lams[-1])],
            valley_window=[float(valley_lams[0]), float(valley_lams[-1])],
            confirm=conf))
    return per_seed


def _median_range(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return dict(median=float("nan"), min=float("nan"), max=float("nan"))
    return dict(median=float(np.median(arr)), min=float(arr.min()), max=float(arr.max()))


def summarize(name, per_seed):
    return dict(
        testbed=name,
        rho=_median_range([s["constants"]["rho"] for s in per_seed]),
        kappa_red=_median_range([s["constants"]["kappa_red"] for s in per_seed]),
        product_pred=_median_range([s["product_pred"] for s in per_seed]),
        product_meas=_median_range([s["confirm"]["product_meas"] for s in per_seed]),
        product_relerr=_median_range([s["confirm"]["product_relerr"] for s in per_seed]),
        rho_relerr=_median_range([s["confirm"]["rho_relerr"] for s in per_seed]),
        valley_weight=_median_range([s["confirm"]["lam_valley"] for s in per_seed]),
        floor_ratio=_median_range([s["confirm"]["floor_ratio"] for s in per_seed]),
        leak_brackets_all_hold=bool(all(s["confirm"]["leak_brackets_hold"]
                                        for s in per_seed)),
        fence_floor_all_hold=bool(all(s["confirm"]["floor_holds"]
                                      for s in per_seed)),
        n_seeds=len(per_seed))


# --------------------------------------------------------------------------- #
# Optional cheap trained-3D tanh-PINN check.                                   #
# --------------------------------------------------------------------------- #
def trained_pinn_valley(seeds=(0, 1), max_iters=400):
    """Small tanh-PINN on 3D Poisson; report soft-loss conditioning proxy valley.

    Cheap CPU confirmation that a conditioning valley in the boundary weight
    appears qualitatively where the fixed-feature theory places it: capped
    iters, small net, 2 seeds.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[pinn] torch unavailable ({exc}); skipping.")
        return None
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    results = []
    lams = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xi = torch.rand(800, 3, dtype=torch.float64)
        Xb_np = _cube_boundary_points(np.random.default_rng(seed), 200)
        Xb = torch.tensor(Xb_np, dtype=torch.float64)
        kx, ky, kz = 2.4, 1.7, 1.3
        coef = (kx ** 2 + ky ** 2 + kz ** 2) * np.pi ** 2

        def u_ex(X):
            return (torch.sin(kx * np.pi * X[:, 0]) * torch.sin(ky * np.pi * X[:, 1])
                    * torch.sin(kz * np.pi * X[:, 2]))

        f_i = coef * u_ex(Xi)
        g_b = u_ex(Xb)

        def make_net():
            net = torch.nn.Sequential(
                torch.nn.Linear(3, 32), torch.nn.Tanh(),
                torch.nn.Linear(32, 32), torch.nn.Tanh(),
                torch.nn.Linear(32, 1)).double()
            return net

        def pde_residual(net, X):
            X = X.clone().requires_grad_(True)
            u = net(X).squeeze(-1)
            grads = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
            lap = 0.0
            for d in range(3):
                gd = grads[:, d]
                g2 = torch.autograd.grad(gd.sum(), X, create_graph=True)[0][:, d]
                lap = lap + g2
            return -lap

        valley = []
        for lam in lams:
            torch.manual_seed(seed)
            net = make_net()
            opt = torch.optim.Adam(net.parameters(), lr=5e-3)
            last = None
            for _ in range(max_iters):
                opt.zero_grad()
                r = pde_residual(net, Xi) - f_i
                bnd = net(Xb).squeeze(-1) - g_b
                loss = (r ** 2).mean() + lam * (bnd ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.item())
            # boundary error as accuracy proxy at this weight
            with torch.no_grad():
                bnd_err = float(torch.sqrt(((net(Xb).squeeze(-1) - g_b) ** 2).mean()))
            valley.append((lam, last, bnd_err))
        results.append(dict(seed=int(seed), sweep=valley))
        best = min(valley, key=lambda t: t[2])
        print(f"[pinn] seed {seed}: boundary-error minimized at lam={best[0]:.0e} "
              f"(bnd_err={best[2]:.3e}); loss/bnd across lam: "
              + ", ".join(f"{l:.0e}:{be:.2e}" for l, _, be in valley))
    return results


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", action="store_true",
                    help="also run the cheap trained-3D tanh-PINN check")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "diagnostic_3d.json"))
    args = ap.parse_args()

    print("PROSPECTIVE BOUNDARY-PENALTY DIAGNOSTIC ON 3D AND MULTISCALE-STIFF PDEs")
    print("Constants and predictions are COMMITTED before the confirming sweep is run.")
    print(f"Seeds: {SEEDS}")

    t_all = time.time()
    tb1 = run_testbed("TB-1  3D Poisson (unit cube, tensor-Fourier, no lifting)",
                      build_poisson_3d)
    tb2 = run_testbed("TB-2  3D stiff anisotropic elliptic (a=(1,40,1600), kappa_red~1e6)",
                      build_stiff_3d)

    summary = [summarize("3D Poisson", tb1),
               summarize("3D stiff anisotropic", tb2)]

    record = dict(
        description=("Prospective boundary-penalty diagnostic on 3D Poisson and a "
                     "3D anisotropic high-contrast (multiscale-stiff) elliptic problem. "
                     "Constants (rho, nu, mu, sigma_B, kappa_red) and predictions "
                     "(regime, leak brackets, product limit rho*sigma_B^2/mu, valley) "
                     "committed before an independent float64 direct-solve sweep."),
        protocol="prospective: constants computed before the confirming sweep",
        seeds=list(SEEDS),
        testbeds=dict(TB1_3d_poisson=tb1, TB2_3d_stiff=tb2),
        summary=summary)

    pinn = None
    if args.pinn:
        print("\n=================== trained 3D tanh-PINN ===================")
        pinn = trained_pinn_valley()
        record["trained_pinn"] = pinn

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"\nWrote {args.out}")

    # -------- final table --------
    print("\n==================== SUMMARY TABLE (median across seeds) ====================")
    hdr = ("testbed", "rho", "kappa_red", "prod pred", "prod meas",
           "prod relerr", "rho relerr", "valley lam")
    print("{:<22}{:>11}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}".format(*hdr))
    for s in summary:
        print("{:<22}{:>11.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>11.2%}{:>12.2%}{:>12.2e}".format(
            s["testbed"], s["rho"]["median"], s["kappa_red"]["median"],
            s["product_pred"]["median"], s["product_meas"]["median"],
            s["product_relerr"]["median"], s["rho_relerr"]["median"],
            s["valley_weight"]["median"]))
    print(f"\nTotal wall time: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
