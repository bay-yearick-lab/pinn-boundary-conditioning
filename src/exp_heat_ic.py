"""The ledger on a time-dependent constraint: heat equation, soft INITIAL
condition with swept weight (boundary weight held fixed), versus the exact
initial-condition lifting.

u_t = u_xx on (0,1) x (0,1], u(x,0) = sin(pi x) + 0.5 sin(3 pi x),
u(0,t) = u(1,t) = 0; exact u = sin(pi x)e^{-pi^2 t} + 0.5 sin(3 pi x)e^{-9 pi^2 t}.

soft lam_ic : loss = mse_r + 100 * mse_bc + lam_ic * mse_ic
hard        : u = IC(x) + t * x(1-x) * N(x,t)  (IC and BC exact)

Writes data/heat_ic_sweep.jsonl.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "data" / "heat_ic_sweep.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def u_exact(x, t):
    return (torch.sin(math.pi * x) * torch.exp(-math.pi ** 2 * t)
            + 0.5 * torch.sin(3 * math.pi * x) * torch.exp(-9 * math.pi ** 2 * t))


def ic_fun(x):
    return torch.sin(math.pi * x) + 0.5 * torch.sin(3 * math.pi * x)


class MLP(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xt):
        return self.net(xt)


def predict(model, xt, hard):
    if hard:
        x, t = xt[:, :1], xt[:, 1:]
        return ic_fun(x) + t * x * (1 - x) * model(xt)
    return model(xt)


def heat_residual(model, xt, hard):
    z = xt.clone().requires_grad_(True)
    u = predict(model, z, hard)
    du = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_t = du[:, 1]
    u_xx = torch.autograd.grad(du[:, 0].sum(), z, create_graph=True)[0][:, 0]
    return u_t - u_xx


def run_one(mode, lam_ic, seed, steps=20000, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    gen = torch.Generator().manual_seed(seed + 1)
    xt_int = torch.rand((2048, 2), generator=gen).to(DEV)
    x_ic = torch.rand((256, 1), generator=gen).to(DEV)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    g_ic = ic_fun(xt_ic[:, 0])
    tb = torch.rand((128, 1), generator=gen).to(DEV)
    xt_bc = torch.cat([torch.cat([torch.zeros_like(tb), tb], dim=1),
                       torch.cat([torch.ones_like(tb), tb], dim=1)])
    hard = mode == "hard"
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        res = heat_residual(model, xt_int, hard)
        loss = torch.mean(res ** 2)
        if not hard:
            u_ic = model(xt_ic).squeeze(-1)
            u_bc = model(xt_bc).squeeze(-1)
            loss = (loss + 100.0 * torch.mean(u_bc ** 2)
                    + lam_ic * torch.mean((u_ic - g_ic) ** 2))
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        gx = torch.linspace(0, 1, 101, device=DEV)
        gt = torch.linspace(0, 1, 101, device=DEV)
        X, T = torch.meshgrid(gx, gt, indexing="ij")
        xt = torch.stack([X.ravel(), T.ravel()], dim=1)
        u = predict(model, xt, hard).squeeze(-1)
        ue = u_exact(xt[:, 0], xt[:, 1])
        rel = (torch.norm(u - ue) / torch.norm(ue)).item()
        x0 = torch.linspace(0, 1, 201, device=DEV).unsqueeze(-1)
        xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=1)
        u0 = predict(model, xt0, hard).squeeze(-1)
        ic_rms = torch.sqrt(torch.mean((u0 - ic_fun(xt0[:, 0])) ** 2)).item()
    return dict(mode=mode, lam_ic=lam_ic, seed=seed, steps=steps, rel_l2=rel,
                ic_rms=ic_rms, wall_s=round(time.time() - t0, 1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["lam_ic"], r["seed"]))
    jobs = []
    for seed in (0, 1, 2, 3, 4):
        for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
            jobs.append(("soft", lam, seed))
        jobs.append(("hard", None, seed))
    with OUT.open("a") as fh:
        for mode, lam, seed in jobs:
            if (mode, lam, seed) in done:
                continue
            rec = run_one(mode, lam, seed)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{mode:5s} lam_ic={str(lam):7s} seed={seed} "
                  f"rel={rec['rel_l2']:.3e} ic={rec['ic_rms']:.3e} ({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
