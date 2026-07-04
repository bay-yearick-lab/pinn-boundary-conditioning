"""Verify the NONLINEAR ledger: three exact laws with no linearization.

At any stationary point w of L_lam(w) = 1/2||r(w)||^2 + (lam/2)||b(w)||^2 with
smooth residual maps r, b (any architecture):

  N1 (exact multiplier): y_lam = lam * b(w) satisfies the exact KKT
      stationarity J_r^T r + J_b^T y_lam = 0 (this IS stationarity, measured
      as the relative gradient norm at convergence).
  N2 (Pareto slope): along a smooth branch of stationary points,
      dP/dQ = -lam with P = ||r||^2, Q = ||b||^2.
  N3 (envelope): dJ/dlam = Q/2 along the branch, J = L_lam(w_lam).

Testbed: a genuine tanh MLP on 2D Poisson with nonzero Dirichlet data and a
RICH boundary group (64 points), sized so residuals outnumber parameters
(both misfits strictly positive at optima). Full-batch float64, Adam then
repeated L-BFGS to tight relative stationarity, warm-started across
lam0*(1 +/- delta) to stay on one branch.
"""

import math

import numpy as np
import scipy.optimize
import torch

PASS = True


def check(name, ok, detail=""):
    global PASS
    status = "PASS" if ok else "FAIL"
    if not ok:
        PASS = False
    print(f"[{status}] {name} {detail}")


torch.manual_seed(0)
DEV = "cpu"          # fp64 L-BFGS is faster on CPU than consumer GPUs
DT = torch.float64

KX, KY = 2.5, 1.5
LAM_EX = (KX ** 2 + KY ** 2) * math.pi ** 2


def u_exact(x, y):
    return torch.sin(KX * math.pi * x) * torch.sin(KY * math.pi * y)


class Net(torch.nn.Module):
    def __init__(self, width=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1))

    def forward(self, xy):
        return self.net(xy)


def make_data(m=200, b=64):
    gen = torch.Generator().manual_seed(3)
    xy_i = torch.rand((m, 2), generator=gen).to(device=DEV, dtype=DT)
    t = torch.rand((b,), generator=gen).to(device=DEV, dtype=DT)
    edge = (torch.arange(b) % 4).to(DEV)
    zx = torch.zeros_like(t)
    ox = torch.ones_like(t)
    bx = torch.where(edge == 0, t, torch.where(edge == 1, t,
                     torch.where(edge == 2, zx, ox)))
    by = torch.where(edge == 0, zx, torch.where(edge == 1, ox,
                     torch.where(edge == 2, t, t)))
    xy_b = torch.stack([bx, by], dim=1)
    f_i = LAM_EX * u_exact(xy_i[:, 0], xy_i[:, 1])
    g_b = u_exact(xy_b[:, 0], xy_b[:, 1])
    return xy_i, xy_b, f_i, g_b


def loss_parts(model, lam, xy_i, xy_b, f_i, g_b):
    x = xy_i.clone().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
    r = uxx + uyy + f_i
    b = model(xy_b).squeeze(-1) - g_b
    P = torch.mean(r ** 2)
    Q = torch.mean(b ** 2)
    return 0.5 * P + 0.5 * lam * Q, P, Q


def get_flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy()


def set_flat(model, x):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.from_numpy(x[i:i + n]).reshape(p.shape).to(p)
        i += n


def train_to_stationarity(model, lam, data, adam_steps=20000, rounds=6):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1),
                                                       eta_min=1e-5)
    for _ in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        L, _, _ = loss_parts(model, lam, *data)
        L.backward()
        opt.step()
        sched.step()

    def fun(x):
        set_flat(model, x)
        L, _, _ = loss_parts(model, lam, *data)
        g = torch.autograd.grad(L, list(model.parameters()))
        gflat = torch.cat([gi.reshape(-1) for gi in g]).numpy()
        return float(L.item()), gflat

    def grad_only(x):
        return fun(x)[1]

    def hess_fd(x, h=1e-6):
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n); e[i] = h
            H[:, i] = (grad_only(x + e) - grad_only(x - e)) / (2 * h)
        return 0.5 * (H + H.T)

    x = get_flat(model)
    res = scipy.optimize.minimize(fun, x, jac=True, method="BFGS",
                                  options=dict(gtol=1e-10, maxiter=20000))
    x = res.x
    for _round in range(rounds):
        res = scipy.optimize.minimize(fun, x, jac=True, hess=hess_fd,
                                      method="trust-exact",
                                      options=dict(gtol=1e-14, maxiter=3000))
        x = res.x
        if np.linalg.norm(res.jac) / (1.0 + abs(res.fun)) < 1e-12:
            break
    set_flat(model, x)
    L, P, Q = loss_parts(model, lam, *data)
    g = torch.autograd.grad(L, list(model.parameters()))
    gnorm = torch.sqrt(sum((gi ** 2).sum() for gi in g)).item()
    gnorm_rel = gnorm / (1.0 + abs(L.item()))
    return float(L.item()), float(P.item()), float(Q.item()), gnorm_rel


def main():
    data = make_data()
    for lam0 in (10.0, 100.0):
        best = None
        for seed in range(4):
            torch.manual_seed(seed)
            cand = Net().to(device=DEV, dtype=DT)
            out = train_to_stationarity(cand, lam0, data)
            if best is None or out[3] < best[1][3]:
                best = (cand, out)
            if out[3] < 1e-8:
                break
        model, (L0, P0, Q0, g0) = best
        check(f"N1 relative stationarity at lam={lam0:g}", g0 < 1e-5,
              f"|grad|/(1+L)={g0:.2e}  P={P0:.3e} Q={Q0:.3e}")
        delta = 0.05
        vals = {}
        for s in (+1, -1):
            lam = lam0 * (1 + s * delta)
            mdl = Net().to(device=DEV, dtype=DT)
            mdl.load_state_dict(model.state_dict())
            vals[s] = train_to_stationarity(mdl, lam, data, adam_steps=0)
        (Lp, Pp, Qp, gp), (Lm, Pm, Qm, gm) = vals[+1], vals[-1]
        check(f"N1 relative stationarity on branch ends lam={lam0:g}",
              max(gp, gm) < 1e-5, f"|grad|/(1+L)={max(gp,gm):.2e}")
        slope = (Pp - Pm) / (Qp - Qm)
        check(f"N2 nonlinear Pareto slope at lam={lam0:g}",
              abs(slope + lam0) / lam0 < 0.02,
              f"dP/dQ={slope:.4f} vs -lam={-lam0:g} "
              f"(rel err {abs(slope+lam0)/lam0:.2e})")
        dJ = (Lp - Lm) / (2 * delta * lam0)
        check(f"N3 nonlinear envelope at lam={lam0:g}",
              abs(dJ - 0.5 * Q0) / (0.5 * Q0) < 0.02,
              f"dJ/dlam={dJ:.3e} vs Q/2={0.5*Q0:.3e} "
              f"(rel err {abs(dJ-0.5*Q0)/(0.5*Q0):.2e})")

    print("\nALL PASS" if PASS else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if PASS else 1)


if __name__ == "__main__":
    main()
