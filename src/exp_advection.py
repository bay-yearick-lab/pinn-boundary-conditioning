"""The ledger on a STIFF operator: steady advection-diffusion with a boundary
layer, weight swept, versus the lifting twin and the three-line ledger.

-eps u'' + u' = f on (0,1), eps = 0.01, Dirichlet u(0)=0, u(1)=1, manufactured
    u*(x) = sin(pi x)/2 + (e^{(x-1)/eps} - e^{-1/eps}) / (1 - e^{-1/eps}),
whose second term is an outflow boundary layer of width eps at x=1; f is
computed by autograd on the closed form. Advection-dominated (mesh Peclet
well above 1 for the collocation spacing), the classic stiff elliptic test.

Arms: soft lambda swept, hard (affine lifting times bubble), ledger (soft plus
the three-line multiplier update at fixed lambda). Five seeds each.
Writes data/advection_sweep.jsonl.
"""

import json
import math
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "data" / "advection_sweep.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 0.01


def u_star(x):
    layer = (torch.exp((x - 1) / EPS) - math.exp(-1 / EPS)) / (1 - math.exp(-1 / EPS))
    return 0.5 * torch.sin(math.pi * x) + layer


def forcing(x):
    """f = -eps u*'' + u*' via autograd on the closed form."""
    x = x.clone().requires_grad_(True)
    u = u_star(x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return (-EPS * u_xx + u_x).detach()


class MLP(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predict(model, x, hard):
    if hard:
        # affine lifting of u(0)=0, u(1)=1 plus a vanishing bubble
        return x + x * (1 - x) * model(x)
    return model(x)


def residual(model, x, f_val, hard):
    z = x.clone().requires_grad_(True)
    u = predict(model, z, hard).squeeze(-1)
    u_x = torch.autograd.grad(u.sum(), z, create_graph=True)[0].squeeze(-1)
    u_xx = torch.autograd.grad(u_x.sum(), z, create_graph=True)[0].squeeze(-1)
    return -EPS * u_xx + u_x - f_val


def run_one(mode, lam, seed, steps=20000, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    gen = torch.Generator().manual_seed(seed + 1)
    x_int = torch.rand((2048, 1), generator=gen).to(DEV)
    f_int = forcing(x_int).squeeze(-1)
    x_bnd = torch.tensor([[0.0], [1.0]], device=DEV)
    g_bnd = u_star(x_bnd).squeeze(-1)
    hard = mode == "hard"
    zmul = torch.zeros_like(g_bnd)
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        res = residual(model, x_int, f_int, hard)
        loss = torch.mean(res ** 2)
        if not hard:
            b = model(x_bnd).squeeze(-1) - g_bnd
            if mode == "ledger":
                if step % 200 == 0 and step > 0:
                    with torch.no_grad():
                        zmul += b
                loss = loss + lam * torch.mean((b + zmul) ** 2)
            else:
                loss = loss + lam * torch.mean(b ** 2)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        gx = torch.linspace(0, 1, 2001, device=DEV).unsqueeze(1)
        u = predict(model, gx, hard).squeeze(-1)
        ue = u_star(gx).squeeze(-1)
        rel = (torch.norm(u - ue) / torch.norm(ue)).item()
        ub = predict(model, x_bnd, hard).squeeze(-1)
        bnd_rms = torch.sqrt(torch.mean((ub - g_bnd) ** 2)).item()
    return dict(mode=mode, lam=lam, seed=seed, steps=steps, rel_l2=rel,
                bnd_rms=bnd_rms, wall_s=round(time.time() - t0, 1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["lam"], r["seed"]))
    jobs = []
    for seed in range(5):
        for lam in (1e0, 1e1, 1e2, 1e3, 1e4):
            jobs.append(("soft", lam, seed))
        jobs.append(("hard", None, seed))
        for lam in (1e2, 1e3):
            jobs.append(("ledger", lam, seed))
    with OUT.open("a") as fh:
        for mode, lam, seed in jobs:
            if (mode, lam, seed) in done:
                continue
            rec = run_one(mode, lam, seed)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{mode:7s} lam={str(lam):7s} seed={seed} "
                  f"rel={rec['rel_l2']:.3e} bnd={rec['bnd_rms']:.3e} ({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
