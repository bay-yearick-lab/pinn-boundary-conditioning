"""Close the nonlinear quantitative loop: measure the tangent-kernel constants
of the trained 2D MLP and test whether the linearized theory prices the
observed valley, plus the practical fence's clock against kappa_red.

Protocol:
  1. Train the standard 4x64 tanh PINN (Poisson, lam=100) exactly as in the
     sweep; also train the hard (transfinite lifting plus bubble) twin. Save final weights.
  2. At the trained soft weights, form the tangent operators by per-sample
     parameter gradients: rows J_r (interior residual map) and J_b (boundary
     map). Work kernel-side (Gram matrices), never in parameter space:
       nonzero spec of H_lam = spec of [[K_rr, sqrt(lam) K_rb],
                                        [sqrt(lam) K_br, lam K_bb]].
  3. Compute kappa_eff(lam) across lam: the predicted valley = argmin; compare
     with the observed sweep valley (lam ~ 1e2).
  4. Compute the reduced clock kappa_red via K_red = J_r P J_r^T with
     P = I - J_b^T (J_b J_b^T)^{-1} J_b, and the shadow price rho of the
     linearization at the trained point (dual/kernel-side least squares).
  5. At the trained hard twin, compute the tangent kernel's kappa_eff and
     compare against kappa_red: the practical fence's clock, priced.

Writes data/ntk_constants.json.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from exp_pinn_sweep import MLP, transfinite_lifting, f_rhs_t, u_exact_t, run_one

OUT = Path(__file__).resolve().parent.parent / "data" / "ntk_constants.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_points(n_int=512, n_bnd=128, seed=1):
    gen = torch.Generator().manual_seed(seed)
    xy_int = torch.rand((n_int, 2), generator=gen).to(DEV)
    tb = torch.rand((n_bnd,), generator=gen).to(DEV)
    edge = (torch.arange(n_bnd) % 4).to(DEV)
    zx, ox = torch.zeros_like(tb), torch.ones_like(tb)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, zx, ox)))
    by = torch.where(edge == 0, zx, torch.where(edge == 1, ox,
                     torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    return xy_int, xy_bnd


def residual_row(model, xy_one, hard):
    """PDE residual at one interior point (scalar), as in the sweep."""
    x = xy_one.clone().requires_grad_(True)
    if hard:
        xx, yy = x[:, :1], x[:, 1:]
        u = transfinite_lifting(xx, yy) + xx * (1 - xx) * yy * (1 - yy) * model(x)
    else:
        u = model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
    return (uxx + uyy + f_rhs_t(x[:, 0], x[:, 1]))[0]


def boundary_row(model, xy_one):
    return (model(xy_one).squeeze(-1) - u_exact_t(xy_one[:, 0], xy_one[:, 1]))[0]


def jacobian_rows(model, xy, fn):
    """Stack per-point parameter gradients into J (n_points x n_params)."""
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    vals = []
    for i in range(xy.shape[0]):
        v = fn(model, xy[i:i + 1])
        g = torch.autograd.grad(v, params, retain_graph=False, allow_unused=True)
        g = [gi if gi is not None else torch.zeros_like(p)
             for gi, p in zip(g, params)]
        rows.append(torch.cat([gi.reshape(-1) for gi in g]).detach())
        vals.append(float(v.item()))
    return torch.stack(rows), np.array(vals)


def kappa_eff_from_gram(G, tol=1e-9):
    ev = np.linalg.eigvalsh(G)
    nz = ev[ev > ev[-1] * tol]
    return ev[-1] / nz[0]


def main():
    # 1. trained models (identical protocol to the sweep, seed 0)
    print("training soft lam=100 ...")
    rec_soft = run_one("soft", 100.0, 0, 20000, DEV, torch.float32)
    print("  soft rel_l2", rec_soft["rel_l2"])
    # retrain but keep the model: run_one does not return it, so rebuild here
    # by rerunning the training loop inline with the same seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Instead of duplicating the loop, monkey-run: temporarily patch run_one's
    # internals is fragile; simplest robust path: train quickly here with the
    # same hyperparameters (results differ trivially from the sweep record).
    def train(mode, lam, steps=20000):
        import exp_pinn_sweep as E
        torch.manual_seed(0)
        model = MLP().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                           eta_min=1e-5)
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
        hard = mode == "hard"
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            x = xy_int.clone().requires_grad_(True)
            if hard:
                xx, yy = x[:, :1], x[:, 1:]
                u = transfinite_lifting(xx, yy) + xx * (1 - xx) * yy * (1 - yy) * model(x)
            else:
                u = model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
            uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
            res = uxx + uyy + f_int
            loss = torch.mean(res ** 2)
            if not hard:
                ub = model(xy_bnd).squeeze(-1)
                loss = loss + lam * torch.mean((ub - g_bnd) ** 2)
            loss.backward()
            opt.step()
            sched.step()
        return model

    model_soft = train("soft", 100.0)
    model_hard = train("hard", None)

    xy_int, xy_bnd = make_points()
    m, b = xy_int.shape[0], xy_bnd.shape[0]

    print("assembling tangent Jacobians (soft model) ...")
    Jr, r0 = jacobian_rows(model_soft, xy_int, lambda M, p: residual_row(M, p, False))
    Jb, b0 = jacobian_rows(model_soft, xy_bnd, boundary_row)
    Jr = (Jr / math.sqrt(m)).cpu().numpy().astype(np.float64)
    Jb = (Jb / math.sqrt(b)).cpu().numpy().astype(np.float64)
    r0 = r0 / math.sqrt(m)
    b0 = b0 / math.sqrt(b)

    Krr = Jr @ Jr.T
    Kbb = Jb @ Jb.T
    Krb = Jr @ Jb.T

    lams = np.logspace(-2, 6, 33)
    kaps = []
    for lam in lams:
        s = math.sqrt(lam)
        G = np.block([[Krr, s * Krb], [s * Krb.T, lam * Kbb]])
        kaps.append(float(kappa_eff_from_gram(G)))
    lam_pred = float(lams[int(np.argmin(kaps))])

    # reduced clock and shadow price of the linearization at the trained point
    Kbb_inv = np.linalg.pinv(Kbb, rcond=1e-10)
    # reduced Gram kernel-side: J_r P_ker(J_b) J_r^T = Krr - Krb Kbb^+ Krb^T
    K_red = Krr - Krb @ Kbb_inv @ Krb.T
    kap_red = float(kappa_eff_from_gram(K_red))
    # boundary Schur curvature nu = ||N||, kernel-side:
    # N = (J_r Y)^T (I - P_range(K_red)) (J_r Y), with J_r Y = Krb Kbb^+ (m x b)
    AYm = Krb @ Kbb_inv
    evr, Ur = np.linalg.eigh(K_red)
    nzmask = evr > evr[-1] * 1e-10
    Pr = (Ur[:, nzmask] @ Ur[:, nzmask].T)
    Nmat = AYm.T @ (np.eye(m) - Pr) @ AYm
    nu = float(np.linalg.norm(Nmat, 2))
    # multiplier of the linearized constrained problem at the trained point:
    # minimize ||r0 + Jr dw||^2 s.t. b0 + Jb dw = 0; dual y* solves
    # (Jb M^+ Jb^T) y = ... kernel-side via KKT in dual variables:
    # stationarity Jr^T(r0 + Jr dw) + Jb^T y = 0, feasibility Jb dw = -b0.
    # Solve the (m+b) KKT system in dual coordinates dw = Jr^T a + Jb^T c:
    # Solve in parameter space via lstsq (n_params ~ 13k is manageable):
    Jr_t = Jr
    Jb_t = Jb
    from numpy.linalg import lstsq
    # nullspace method: dw = dw_p + Zv with Jb dw = -b0. Use lstsq for dw_p:
    dw_p, *_ = lstsq(Jb_t, -b0, rcond=None)
    # project residual objective onto ker(Jb) via iterative solve: use CG on
    # normal equations with projector applied kernel-side is overkill; instead
    # solve min ||r0 + Jr(dw_p + w)||, Jb w = 0 by lstsq on augmented system
    # with large penalty (1e8) as a numerically adequate proxy:
    BIG = 1e8
    Astack = np.vstack([Jr_t, math.sqrt(BIG) * Jb_t])
    ystack = np.concatenate([-(r0 + Jr_t @ dw_p), np.zeros(b)])
    w_corr, *_ = lstsq(Astack, ystack, rcond=None)
    dw = dw_p + w_corr
    r_star = r0 + Jr_t @ dw
    y_star, *_ = lstsq(Jb_t.T, -(Jr_t.T @ r_star), rcond=None)
    rho = float(np.linalg.norm(y_star))

    print("assembling tangent Jacobians (hard twin) ...")
    Jr_h, _ = jacobian_rows(model_hard, xy_int, lambda M, p: residual_row(M, p, True))
    Jr_h = (Jr_h / math.sqrt(m)).cpu().numpy().astype(np.float64)
    kap_hard = float(kappa_eff_from_gram(Jr_h @ Jr_h.T))

    out = dict(lams=lams.tolist(), kappas=kaps, lam_pred=lam_pred,
               kap_red=kap_red, kap_hard_practical=kap_hard, rho=rho, nu=nu,
               soft_rel_l2=rec_soft["rel_l2"])
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("lams", "kappas")},
                     indent=1))
    print(f"predicted valley lam={lam_pred:g} (observed sweep valley ~1e2)")


if __name__ == "__main__":
    main()
