"""The ledger on a NONLINEAR PDE: forced viscous Burgers with a soft Dirichlet
boundary, weight swept, versus the transfinite lifting twin.

u_t + u u_x - nu u_xx = f on (0,1) x (0,1], manufactured
u*(x,t) = e^{-t} sin(pi x) + 0.3 e^{-2t} sin(2 pi x), nu = 0.05,
f = u*_t + u* u*_x - nu u*_xx (computed by autograd on the exact expression).
Dirichlet data on x in {0,1} and the initial slice t=0 form ONE soft
constraint group with common weight lambda (swept); the hard twin lifts both:
u = u*(x,0) + t x (1-x) N(x,t) reproduces IC and (homogeneous) BC exactly
since u*(0,t)=u*(1,t)=0.

Writes data/burgers_sweep.jsonl.
"""

import json
import math
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "data" / "burgers_sweep.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NU = 0.05


def u_star(x, t):
    return (torch.exp(-t) * torch.sin(math.pi * x)
            + 0.3 * torch.exp(-2 * t) * torch.sin(2 * math.pi * x))


def forcing(x, t):
    """f = u*_t + u* u*_x - nu u*_xx via autograd on the closed form."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_star(x, t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return (u_t + u * u_x - NU * u_xx).detach()


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
        return u_star(x, torch.zeros_like(t)) + t * x * (1 - x) * model(xt)
    return model(xt)


def residual(model, xt, f_val, hard):
    z = xt.clone().requires_grad_(True)
    u = predict(model, z, hard).squeeze(-1)
    du = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_x, u_t = du[:, 0], du[:, 1]
    u_xx = torch.autograd.grad(u_x.sum(), z, create_graph=True)[0][:, 0]
    return u_t + u * u_x - NU * u_xx - f_val


def run_one(mode, lam, seed, steps=20000, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    gen = torch.Generator().manual_seed(seed + 1)
    xt_int = torch.rand((2048, 2), generator=gen).to(DEV)
    f_int = forcing(xt_int[:, 0], xt_int[:, 1])
    # constraint group: boundary lines + initial slice
    tb = torch.rand((128, 1), generator=gen).to(DEV)
    xb = torch.rand((128, 1), generator=gen).to(DEV)
    xt_con = torch.cat([torch.cat([torch.zeros_like(tb), tb], dim=1),
                        torch.cat([torch.ones_like(tb), tb], dim=1),
                        torch.cat([xb, torch.zeros_like(xb)], dim=1)])
    g_con = u_star(xt_con[:, 0], xt_con[:, 1])
    hard = mode == "hard"
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        res = residual(model, xt_int, f_int, hard)
        loss = torch.mean(res ** 2)
        if not hard:
            u_c = model(xt_con).squeeze(-1)
            loss = loss + lam * torch.mean((u_c - g_con) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        gx = torch.linspace(0, 1, 101, device=DEV)
        gt = torch.linspace(0, 1, 101, device=DEV)
        X, T = torch.meshgrid(gx, gt, indexing="ij")
        xt = torch.stack([X.ravel(), T.ravel()], dim=1)
        u = predict(model, xt, hard).squeeze(-1)
        ue = u_star(xt[:, 0], xt[:, 1])
        rel = (torch.norm(u - ue) / torch.norm(ue)).item()
        uc = predict(model, xt_con, hard).squeeze(-1)
        con_rms = torch.sqrt(torch.mean((uc - g_con) ** 2)).item()
    return dict(mode=mode, lam=lam, seed=seed, steps=steps, rel_l2=rel,
                con_rms=con_rms, wall_s=round(time.time() - t0, 1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["lam"], r["seed"]))
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
            print(f"{mode:5s} lam={str(lam):7s} seed={seed} "
                  f"rel={rec['rel_l2']:.3e} con={rec['con_rms']:.3e} ({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
