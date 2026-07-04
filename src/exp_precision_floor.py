"""The arithmetic tax: even exact solvers pay kappa(lam) * eps_mach.

Solve the penalized normal equations H_lam w = rhs by Cholesky in float32 and
float64 across lam, on the misfit 1D testbed. Total solution error decomposes
as
    err(lam) ~ leak(lam) + kappa(lam) * eps_mach * amplification
             ~ rho/(lam sigma_B) + c * lam * eps_mach,
a V-shaped curve whose floor scales as sqrt(rho * eps_mach * c') and whose
optimal lam scales as sqrt(rho / eps_mach): in float32 the attainable error
of a soft-constrained solve is capped orders of magnitude above float64,
REGARDLESS of optimizer. Writes data/precision_floor.json.
"""

import json
from pathlib import Path

import numpy as np
from problems import build_poisson_1d_eig, constrained_solution

OUT = Path(__file__).resolve().parent.parent / "data" / "precision_floor.json"


def solve_in(dtype, A, f, B, g, lam):
    A_, f_, B_, g_ = (x.astype(dtype) for x in (A, f, B, g))
    lam = dtype(lam)
    H = A_.T @ A_ + lam * (B_.T @ B_)
    rhs = A_.T @ f_ + lam * (B_.T @ g_)
    try:
        c = np.linalg.cholesky(H)
        w = np.linalg.solve(c.T, np.linalg.solve(c, rhs))
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(H, rhs, rcond=None)[0]
    return w.astype(np.float64)


def main():
    prob = build_poisson_1d_eig(normalize=True)
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    rho = float(np.linalg.norm(y_star))
    lams = np.logspace(0, 12, 49)
    out = {"lams": lams.tolist(), "rho": rho,
           "sigma_B": float(np.linalg.norm(B, 2))}
    for dtype, name in ((np.float32, "float32"), (np.float64, "float64")):
        errs = []
        for lam in lams:
            w = solve_in(dtype, A, f, B, g, lam)
            errs.append(float(np.linalg.norm(w - w_star)))
        out[name] = errs
        best = int(np.argmin(errs))
        print(f"{name}: best err {errs[best]:.3e} at lam={lams[best]:.2e}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
