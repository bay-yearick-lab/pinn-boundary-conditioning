"""Drift of the tangent-kernel constants during feature learning.

Trains the standard soft Poisson PINN at lam=100 and measures rho, nu, and
the kappa_eff-argmin at checkpoints {0, 2k, 5k, 10k, 20k} steps, quantifying
how fast the linearization's constants move. Writes data/ntk_drift.json.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from exp_ntk_constants import (boundary_row, jacobian_rows, kappa_eff_from_gram,
                               make_points, residual_row)
from exp_pinn_sweep import MLP, f_rhs_t, u_exact_t

OUT = Path(__file__).resolve().parent.parent / "data" / "ntk_drift.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def constants_at(model):
    xy_int, xy_bnd = make_points(n_int=384, n_bnd=96)
    m, b = xy_int.shape[0], xy_bnd.shape[0]
    Jr, r0 = jacobian_rows(model, xy_int, lambda M, p: residual_row(M, p, False))
    Jb, b0 = jacobian_rows(model, xy_bnd, boundary_row)
    Jr = (Jr / math.sqrt(m)).cpu().numpy().astype(np.float64)
    Jb = (Jb / math.sqrt(b)).cpu().numpy().astype(np.float64)
    r0, b0 = r0 / math.sqrt(m), b0 / math.sqrt(b)
    Krr, Kbb, Krb = Jr @ Jr.T, Jb @ Jb.T, Jr @ Jb.T
    lams = np.logspace(-2, 6, 25)
    kaps = []
    for lam in lams:
        s = math.sqrt(lam)
        G = np.block([[Krr, s * Krb], [s * Krb.T, lam * Kbb]])
        kaps.append(kappa_eff_from_gram(G))
    lam_argmin = float(lams[int(np.argmin(kaps))])
    Kbb_inv = np.linalg.pinv(Kbb, rcond=1e-10)
    K_red = Krr - Krb @ Kbb_inv @ Krb.T
    AYm = Krb @ Kbb_inv
    evr, Ur = np.linalg.eigh(K_red)
    nzmask = evr > evr[-1] * 1e-10
    Pr = Ur[:, nzmask] @ Ur[:, nzmask].T
    nu = float(np.linalg.norm(AYm.T @ (np.eye(m) - Pr) @ AYm, 2))
    from numpy.linalg import lstsq
    dw_p, *_ = lstsq(Jb, -b0, rcond=None)
    BIG = 1e8
    Astack = np.vstack([Jr, math.sqrt(BIG) * Jb])
    ystack = np.concatenate([-(r0 + Jr @ dw_p), np.zeros(b)])
    w_corr, *_ = lstsq(Astack, ystack, rcond=None)
    r_star = r0 + Jr @ (dw_p + w_corr)
    y_star, *_ = lstsq(Jb.T, -(Jr.T @ r_star), rcond=None)
    return dict(lam_argmin=lam_argmin, nu=nu, rho=float(np.linalg.norm(y_star)))


def main():
    lam = 100.0
    checkpoints = [0, 2000, 5000, 10000, 20000]
    torch.manual_seed(0)
    model = MLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20000, eta_min=1e-5)
    gen = torch.Generator().manual_seed(1)
    xy_int = torch.rand((2048, 2), generator=gen).to(DEV)
    tb = torch.rand((256,), generator=gen).to(DEV)
    edge = (torch.arange(256) % 4).to(DEV)
    zx, ox = torch.zeros_like(tb), torch.ones_like(tb)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, zx, ox)))
    by = torch.where(edge == 0, zx, torch.where(edge == 1, ox,
                     torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])

    out = {}
    step = 0
    for ck in checkpoints:
        while step < ck:
            opt.zero_grad(set_to_none=True)
            x = xy_int.clone().requires_grad_(True)
            u = model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
            uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
            res = uxx + uyy + f_int
            ub = model(xy_bnd).squeeze(-1)
            loss = torch.mean(res ** 2) + lam * torch.mean((ub - g_bnd) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        c = constants_at(model)
        out[str(ck)] = c
        print(ck, json.dumps(c))
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
