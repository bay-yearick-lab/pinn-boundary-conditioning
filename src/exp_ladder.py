"""The ladder: four formulations of the same problem, one budget axis.

On the misfit 1D testbed (Jacobi-normalized eigenfunction basis, the best case
for a penalty), run first-order training under the four classical rungs:

  1. quadratic penalty (vanilla PINN loss), best fixed lam per budget
                                           -> algebraic envelope ~ 1/T
  2. augmented Lagrangian (method of multipliers), fixed moderate lam
                                           -> geometric, multiplier learns y*
  3. consistent penalty (subtract the multiplier correction using the
     KKT-consistent flux estimate, as in consistent finite-element
     boundary penalties) at fixed lam      -> geometric to the consistent point
  4. reduced trial space (the fence): GD on ker(B) after lifting, the
     neural analogue of FEM essential BCs  -> geometric, lam-free

Exact modal/linear computations, no stochastic noise. Writes
data/ladder.json with error-vs-budget curves for each rung.
"""

import json
from pathlib import Path

import numpy as np
from problems import build_poisson_1d_eig, constrained_solution

OUT = Path(__file__).resolve().parent.parent / "data" / "ladder.json"


def gd_error_curve(H, rhs, w_ref, w0, ts):
    h, V = np.linalg.eigh(H)
    eta = 1.0 / h[-1]
    w_min = V @ ((V.T @ rhs) / h)
    c0 = V.T @ (w0 - w_min)
    off = V.T @ (w_min - w_ref)
    fac = 1.0 - eta * h
    return np.array([np.linalg.norm(c0 * np.power(fac, float(t)) + off) for t in ts])


def al_error_curve(A, f, B, g, w_star, lam, ts, inner=200):
    """Augmented Lagrangian: outer multiplier updates, inner GD steps.

    Budget = total inner GD steps. Uses exact inner GD (modal propagation)
    from the previous iterate, y-update after each inner block.
    """
    n = A.shape[1]
    H = A.T @ A + lam * (B.T @ B)
    h, V = np.linalg.eigh(H)
    eta = 1.0 / h[-1]
    fac = 1.0 - eta * h
    w = np.zeros(n)
    y = np.zeros(B.shape[0])
    errs, total = [], 0
    t_targets = list(ts)
    out = {}
    while t_targets:
        # inner problem: min ||Aw-f||^2/2 + y'(Bw-g) + lam/2 ||Bw-g||^2
        rhs = A.T @ f - B.T @ y + lam * (B.T @ g)
        w_min = V @ ((V.T @ rhs) / h)
        c0 = V.T @ (w - w_min)
        steps = min(inner, max(t_targets) - total)
        # record error at any targets inside this block
        for t in [t for t in t_targets if t <= total + steps]:
            dt = t - total
            out[t] = np.linalg.norm(V @ (c0 * np.power(fac, float(dt))) + (w_min - w_star))
        t_targets = [t for t in t_targets if t > total + steps]
        w = V @ (c0 * np.power(fac, float(steps))) + w_min
        total += steps
        y = y + lam * (B @ w - g)
    return np.array([out[t] for t in ts])


def consistent_penalty_error_curve(A, f, B, g, w_star, y_star, lam, ts):
    """Consistent penalty: add the multiplier correction term to the loss.

    L_N(w) = 1/2||Aw-f||^2 + y*'(Bw-g) + lam/2 ||Bw-g||^2 has stationary point
    exactly at w* for EVERY lam (perturbed-KKT consistency); in a consistent finite-element
    boundary penalty the exact flux y* is replaced by the discrete conormal derivative,
    which this linear testbed can supply exactly. Geometric to w*, no leak.
    """
    H = A.T @ A + lam * (B.T @ B)
    rhs = A.T @ f - B.T @ y_star + lam * (B.T @ g)
    return gd_error_curve(H, rhs, w_star, np.zeros(A.shape[1]), ts)


def main():
    prob = build_poisson_1d_eig(normalize=True)
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    ts = np.unique(np.round(np.logspace(0, 6.5, 40))).astype(int)

    # rung 1: quadratic penalty, envelope over lam
    lam_grid = np.logspace(0, 9, 28)
    env = np.full(len(ts), np.inf)
    per_lam = {}
    for lam in lam_grid:
        H = A.T @ A + lam * (B.T @ B)
        rhs = A.T @ f + lam * (B.T @ g)
        errs = gd_error_curve(H, rhs, w_star, np.zeros(A.shape[1]), ts)
        per_lam[lam] = errs
        env = np.minimum(env, errs)

    lam_fix = 100.0
    penalty_fixed = per_lam[lam_grid[np.argmin(np.abs(lam_grid - lam_fix))]]

    # rung 2: augmented Lagrangian at the same fixed lam
    al = al_error_curve(A, f, B, g, w_star, lam_fix, ts, inner=200)

    # rung 3: consistent penalty at the same fixed lam
    nit = consistent_penalty_error_curve(A, f, B, g, w_star, y_star, lam_fix, ts)

    # rung 4: the fence (reduced space)
    w_p = np.linalg.lstsq(B, g, rcond=None)[0]
    M = Z.T @ (A.T @ A) @ Z
    rhs_v = Z.T @ (A.T @ (f - A @ w_p))
    v_star = np.linalg.lstsq(M, rhs_v, rcond=None)[0]
    hard = gd_error_curve(M, rhs_v, v_star, np.zeros(M.shape[0]), ts)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(
        ts=ts.tolist(),
        penalty_envelope=env.tolist(),
        penalty_fixed=dict(lam=lam_fix, errs=penalty_fixed.tolist()),
        augmented_lagrangian=dict(lam=lam_fix, errs=al.tolist()),
        consistent=dict(lam=lam_fix, errs=nit.tolist()),
        hard=hard.tolist(),
        rho=float(np.linalg.norm(y_star)),
    ), indent=1))
    print(f"wrote {OUT}")
    for name, arr in (("penalty env", env), ("penalty lam=100", penalty_fixed),
                      ("augmented L", al), ("consistent", nit), ("hard", hard)):
        print(f"{name:16s} err@1e3={arr[np.searchsorted(ts, 1000)]:.2e} "
              f"err@1e5={arr[np.searchsorted(ts, 100000)]:.2e} final={arr[-1]:.2e}")


if __name__ == "__main__":
    main()
