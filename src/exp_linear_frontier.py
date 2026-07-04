"""Figure data for the linear-regime exhibits: leak, drag, U, frontier.

Dumps data/linear_frontier.json with:
  statics: lam grid, leak norms, rho (1D eig testbed and 2D tensor testbed)
  drag:    lam grid, kappa(H_lam) for the misfit testbed (normalized)
  ushape:  lam grid, kappa_eff for the realizable testbed, lam_ntk, kappa_red
  frontier: budgets, soft envelope (GD), heavy-ball envelope, hard curve,
            plus per-lam GD curves for a few marked lam values
"""

import json
from pathlib import Path

import numpy as np
from problems import (build_poisson_1d_eig, build_poisson_2d,
                      constrained_solution, penalized_solution)
from verify_dynamics import gd_error_curve, heavyball_error_curve
from verify_ushape import build_realizable, kappa_eff

OUT = Path(__file__).resolve().parent.parent / "data" / "linear_frontier.json"


def statics_block(prob):
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = float(np.linalg.norm(y_star))
    # boundary Schur curvature for the lower bracket rho/(lam+nu)
    Y = B.T @ np.linalg.inv(B @ B.T)
    AZ, AY = A @ Z, A @ Y
    Mz = AZ.T @ AZ
    Nmat = AY.T @ (AY - AZ @ np.linalg.solve(Mz, AZ.T @ AY))
    nu = float(np.linalg.norm(Nmat, 2))
    lams = np.logspace(0, 8, 33)
    leaks = []
    for lam in lams:
        w_lam, _ = penalized_solution(A, f, B, g, lam)
        leaks.append(float(np.linalg.norm(B @ w_lam - g)))
    return dict(lams=lams.tolist(), leaks=leaks, rho=rho, nu=nu)


def main():
    out = {}
    prob1 = build_poisson_1d_eig(normalize=True)
    prob2 = build_poisson_2d()
    out["statics_1d"] = statics_block(prob1)
    out["statics_2d"] = statics_block(prob2)

    # drag: kappa(H_lam) on the misfit testbed
    A, f, B, g = prob1["A"], prob1["f"], prob1["B"], prob1["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    M = Z.T @ (A.T @ A) @ Z
    mu = float(np.linalg.eigvalsh(M)[0])
    lams = np.logspace(-4, 8, 49)
    kaps = []
    for lam in lams:
        ev = np.linalg.eigvalsh(A.T @ A + lam * (B.T @ B))
        kaps.append(float(ev[-1] / ev[0]))
    out["drag"] = dict(lams=lams.tolist(), kappas=kaps, mu=mu,
                       sB2=float(np.linalg.norm(B, 2) ** 2),
                       rho=float(np.linalg.norm(y_star)))

    # ushape (realizable testbed)
    Ar, fr, Br, gr, w_true = build_realizable()
    lams_u = np.logspace(-4, 6, 41)
    kaps_u = [float(kappa_eff(Ar.T @ Ar + lam * (Br.T @ Br))) for lam in lams_u]
    _, sv, Vt = np.linalg.svd(Br, full_matrices=True)
    Zr = Vt[len(sv):].T
    kap_red = float(kappa_eff(Zr.T @ (Ar.T @ Ar) @ Zr))
    lam_ntk = float(np.trace(Ar @ Ar.T) / np.trace(Br @ Br.T))
    out["ushape"] = dict(lams=lams_u.tolist(), kappas=kaps_u,
                         kappa_red=kap_red, lam_ntk=lam_ntk)

    # frontier: envelopes and per-lam curves
    budgets = np.unique(np.round(np.logspace(0, 7, 36))).astype(float)
    lam_grid = np.logspace(1, 13, 49)
    w0 = np.zeros_like(w_star)
    env = np.full(len(budgets), np.inf)
    env_hb = np.full(len(budgets), np.inf)
    curves = {}
    marked = [1e2, 1e4, 1e6, 1e8]
    for lam in lam_grid:
        H = A.T @ A + lam * (B.T @ B)
        rhs = A.T @ f + lam * (B.T @ g)
        errs, _ = gd_error_curve(H, rhs, w_star, w0, budgets)
        env = np.minimum(env, errs)
        env_hb = np.minimum(env_hb, heavyball_error_curve(H, rhs, w_star, w0, budgets))
        for mk in marked:
            if abs(np.log10(lam / mk)) < 0.01:
                curves[f"lam{mk:.0e}"] = errs.tolist()
    w_p = np.linalg.lstsq(B, g, rcond=None)[0]
    rhs_v = Z.T @ (A.T @ (f - A @ w_p))
    v_star = np.linalg.lstsq(M, rhs_v, rcond=None)[0]
    hard, _ = gd_error_curve(M, rhs_v, v_star, np.zeros(M.shape[0]), budgets)
    out["frontier"] = dict(budgets=budgets.tolist(), soft_envelope=env.tolist(),
                           heavyball_envelope=env_hb.tolist(), hard=hard.tolist(),
                           curves=curves)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
