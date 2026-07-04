"""Full tanh-MLP PINN experiment: the U in practice, and the fence at equal budget.

2D Poisson on the unit square, manufactured solution
    u(x, y) = sin(2.5 pi x) sin(1.5 pi y),   f = -Lap u = 8.5 pi^2 u,
nonzero Dirichlet data on the edges.

Runs (each at several seeds, fixed Adam budget):
  soft lam=...   : vanilla PINN loss  L = mse_r + lam * mse_b
  adaptive       : gradient-statistic weighting (learning-rate-annealing style)
  hard           : transfinite-interpolation lifting u = G + x(1-x)y(1-y) N
                   (the fence: boundary exact by construction)

Outputs incremental JSON records to data/pinn_sweep.jsonl.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "data" / "pinn_sweep.jsonl"

KX, KY = 2.5, 1.5
LAM_EX = (KX ** 2 + KY ** 2) * math.pi ** 2   # 8.5 pi^2
K2 = 0.0                                       # Helmholtz -Lap u - K2 u = f


def u_exact_t(x, y):
    return torch.sin(KX * math.pi * x) * torch.sin(KY * math.pi * y)


def f_rhs_t(x, y):
    return (LAM_EX - K2) * u_exact_t(x, y)


class MLP(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


def transfinite_lifting(x, y):
    """Transfinite interpolation of the exact edge data (the built-in lifting)."""
    def g(xx, yy):
        return torch.sin(KX * math.pi * xx) * torch.sin(KY * math.pi * yy)
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    G = ((1 - y) * g(x, zero) + y * g(x, one)
         + (1 - x) * g(zero, y) + x * g(one, y)
         - ((1 - x) * (1 - y) * g(zero, zero) + x * (1 - y) * g(one, zero)
            + (1 - x) * y * g(zero, one) + x * y * g(one, one)))
    return G


def laplacian(model, xy, hard=False):
    xy = xy.requires_grad_(True)
    if hard:
        x, y = xy[:, :1], xy[:, 1:]
        u = transfinite_lifting(x, y) + x * (1 - x) * y * (1 - y) * model(xy)
    else:
        u = model(xy)
    grads = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_xx = torch.autograd.grad(grads[:, 0].sum(), xy, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(grads[:, 1].sum(), xy, create_graph=True)[0][:, 1]
    return u.squeeze(-1), u_xx + u_yy


def evaluate(model, device, dtype, hard=False, n=101):
    with torch.no_grad():
        gx = torch.linspace(0, 1, n, device=device, dtype=dtype)
        X, Y = torch.meshgrid(gx, gx, indexing="ij")
        xy = torch.stack([X.ravel(), Y.ravel()], dim=1)
        if hard:
            x, y = xy[:, :1], xy[:, 1:]
            u = (transfinite_lifting(x, y) + x * (1 - x) * y * (1 - y) * model(xy)).squeeze(-1)
        else:
            u = model(xy).squeeze(-1)
        ue = u_exact_t(xy[:, 0], xy[:, 1])
        rel_l2 = (torch.norm(u - ue) / torch.norm(ue)).item()
        # boundary error on the four edges
        t = torch.linspace(0, 1, n, device=device, dtype=dtype)
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        eb = []
        for (bx, by) in ((t, zeros), (t, ones), (zeros, t), (ones, t)):
            bxy = torch.stack([bx, by], dim=1)
            if hard:
                x, y = bxy[:, :1], bxy[:, 1:]
                ub = (transfinite_lifting(x, y) + x * (1 - x) * y * (1 - y) * model(bxy)).squeeze(-1)
            else:
                ub = model(bxy).squeeze(-1)
            eb.append(ub - u_exact_t(bxy[:, 0], bxy[:, 1]))
        bnd_rms = torch.sqrt(torch.mean(torch.cat(eb) ** 2)).item()
    return rel_l2, bnd_rms


def run_one(mode, lam, seed, steps, device, dtype, width=64, depth=4,
            n_int=2048, n_bnd=256, lr=1e-3, record_every=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(width, depth).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xy_int = torch.rand((n_int, 2), generator=gen).to(device=device, dtype=dtype)
    tb = torch.rand((n_bnd,), generator=gen).to(device=device, dtype=dtype)
    edge = (torch.arange(n_bnd) % 4).to(device)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, torch.zeros_like(tb), torch.ones_like(tb))))
    by = torch.where(edge == 0, torch.zeros_like(tb), torch.where(edge == 1,
                     torch.ones_like(tb), torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])

    hard = mode == "hard"
    lam_t = float(lam) if lam is not None else 1.0
    zmul = torch.zeros_like(g_bnd)         # scaled multiplier z = y/lambda
    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        u_i, lap = laplacian(model, xy_int.clone(), hard=hard)
        res = lap + K2 * u_i + f_int           # Lap u + K2 u + f = 0
        loss_r = torch.mean(res ** 2)
        if hard:
            loss = loss_r
            loss_b = torch.zeros((), device=device, dtype=dtype)
        elif mode == "ledger":
            # augmented Lagrangian with the measured multiplier (first-order multiplier
            # update on the boundary residuals), weight held FIXED
            u_b = model(xy_bnd).squeeze(-1)
            bres = u_b - g_bnd
            loss_b = torch.mean(bres ** 2)
            if step % 500 == 0 and step > 0:
                zmul = (zmul + bres).detach()
            loss = loss_r + lam_t * torch.mean((bres + zmul) ** 2)
        else:
            u_b = model(xy_bnd).squeeze(-1)
            loss_b = torch.mean((u_b - g_bnd) ** 2)
            if mode == "adaptive" and step % 100 == 0 and step > 0:
                # gradient-statistic rule: balance max residual-gradient scale
                # against mean boundary-gradient scale (annealing-style)
                gr = torch.autograd.grad(loss_r, [p for p in model.parameters()],
                                         retain_graph=True, allow_unused=True)
                gb = torch.autograd.grad(loss_b, [p for p in model.parameters()],
                                         retain_graph=True, allow_unused=True)
                mg = max(g.abs().max().item() for g in gr if g is not None)
                mb = np.mean([g.abs().mean().item() for g in gb if g is not None])
                if mb > 0:
                    lam_t = 0.9 * lam_t + 0.1 * (mg / mb)
            loss = loss_r + lam_t * loss_b
        loss.backward()
        opt.step()
        sched.step()
        if step % record_every == 0:
            rel_l2, bnd_rms = evaluate(model, device, dtype, hard=hard)
            traj.append(dict(step=step, rel_l2=rel_l2, bnd_rms=bnd_rms,
                             loss_r=loss_r.item(), loss_b=float(loss_b.item()),
                             lam=lam_t))
    rel_l2, bnd_rms = evaluate(model, device, dtype, hard=hard)
    return dict(mode=mode, lam=lam, lam_final=lam_t, seed=seed, steps=steps,
                dtype=str(dtype).split(".")[-1], rel_l2=rel_l2, bnd_rms=bnd_rms,
                wall_s=round(time.time() - t0, 1), traj=traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pde", choices=["poisson", "helmholtz"], default="poisson")
    ap.add_argument("--ledger", action="store_true",
                    help="run only the multiplier-corrected (ledger) mode")
    args = ap.parse_args()
    global K2, OUT
    if args.pde == "helmholtz":
        K2 = 40.0
        OUT = OUT.with_name("pinn_sweep_helmholtz.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    OUT.parent.mkdir(exist_ok=True)
    lams = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    if args.quick:
        lams, args.seeds, args.steps = [1e0, 1e3], 1, 2000
    jobs = []
    for seed in range(args.seeds):
        for lam in lams:
            jobs.append(("soft", lam, seed, torch.float32))
        jobs.append(("adaptive", None, seed, torch.float32))
        jobs.append(("hard", None, seed, torch.float32))
    if args.ledger:
        jobs = [("ledger", lam, seed, torch.float32)
                for seed in range(args.seeds)
                for lam in (1e0, 1e1, 1e2, 1e3, 1e4)]
    # float64 spot checks of the U shape
    for lam in ([1e0, 1e3, 1e5] if (not args.quick and args.pde == "poisson") else []):
        jobs.append(("soft", lam, 0, torch.float64))
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["lam"], r["seed"], r["dtype"], r["steps"]))
    with OUT.open("a") as fh:
        for mode, lam, seed, dtype in jobs:
            key = (mode, lam, seed, str(dtype).split(".")[-1], args.steps)
            if key in done:
                print(f"skip {key}")
                continue
            rec = run_one(mode, lam, seed, args.steps, device, dtype)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{mode:9s} lam={str(lam):8s} seed={seed} dtype={rec['dtype']} "
                  f"rel_l2={rec['rel_l2']:.3e} bnd={rec['bnd_rms']:.3e} "
                  f"lam_final={rec['lam_final']:.3g} ({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
