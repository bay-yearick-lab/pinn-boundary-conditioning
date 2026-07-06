"""The ledger in operator learning: a physics-informed DeepONet under the boundary penalty.

Family of problems: -u'' = f on (0,1), u(0)=g0, u(1)=g1, with
f(x) = sum_k c_k sin(k pi x), c_k ~ N(0, k^{-1.5}), g0,g1 ~ N(0, 0.35^2).
Exact solution u = sum_k c_k sin(k pi x)/(k pi)^2 + g0 (1-x) + g1 x.

soft lam : DeepONet trained with residual + lam * boundary loss (data-free:
           physics only, no solution labels).
hard     : same architecture with the affine lifting installed family-wide,
           u = g0 (1-x) + g1 x + x(1-x) * (branch . trunk).

Writes data/operator_sweep.jsonl.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "data" / "operator_sweep.jsonl"
K = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sample_instances(n, gen):
    ks = torch.arange(1, K + 1, dtype=torch.float32)
    c = torch.randn(n, K, generator=gen) * ks.pow(-1.5)[None, :]
    g = torch.randn(n, 2, generator=gen) * 0.35
    return c, g


def f_of(c, x):
    """f(x) = sum_k c_k sin(k pi x); x: (m,), c: (n,K) -> (n,m)."""
    ks = torch.arange(1, K + 1, device=c.device, dtype=c.dtype)
    return torch.einsum("nk,km->nm", c, torch.sin(math.pi * ks[:, None] * x[None, :]))


def u_exact(c, g, x):
    ks = torch.arange(1, K + 1, device=c.device, dtype=c.dtype)
    modes = torch.sin(math.pi * ks[:, None] * x[None, :]) / (math.pi * ks[:, None]) ** 2
    return (torch.einsum("nk,km->nm", c, modes)
            + g[:, :1] * (1 - x)[None, :] + g[:, 1:] * x[None, :])


class DeepONet(torch.nn.Module):
    def __init__(self, p=64, width=64):
        super().__init__()
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(K + 2, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, p))
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(1, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, p))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def core(self, v, x):
        """v: (n, K+2) instance codes; x: (m,1) -> (n, m)."""
        B = self.branch(v)                       # (n,p)
        T = self.trunk(x)                        # (m,p)
        return torch.einsum("np,mp->nm", B, T) + self.bias


def predict(model, v, g, x, hard):
    if hard:
        lift = g[:, :1] * (1 - x.squeeze(-1))[None, :] + g[:, 1:] * x.squeeze(-1)[None, :]
        bubble = (x.squeeze(-1) * (1 - x.squeeze(-1)))[None, :]
        return lift + bubble * model.core(v, x)
    return model.core(v, x)


def run_one_batched(mode, lam, seed, steps=15000, n_batch=32, m_col=64, lr=1e-3):
    """Correct per-instance Laplacian: give each instance its own x tensor by
    tiling x to shape (n*m, 1) and pairing with repeated instance codes."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 10)
    model = DeepONet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    hard = mode == "hard"
    x_bnd = torch.tensor([[0.0], [1.0]], device=DEV)
    t0 = time.time()
    for step in range(steps):
        c, g = sample_instances(n_batch, gen)
        c, g = c.to(DEV), g.to(DEV)
        v = torch.cat([c, g], dim=1)
        x = torch.rand((m_col, 1), generator=gen).to(DEV)
        # per-instance second derivatives via tiling: each instance gets its
        # own copy of the collocation coordinates
        xt = x.squeeze(-1).repeat(n_batch).unsqueeze(-1).requires_grad_(True)  # (n*m,1)
        vt = v.repeat_interleave(m_col, dim=0)                   # (n*m, K+2)
        gt = g.repeat_interleave(m_col, dim=0)
        Bv = model.branch(vt)                                    # (n*m, p)
        Tx = model.trunk(xt)                                     # (n*m, p)
        ut = (Bv * Tx).sum(-1, keepdim=True) + model.bias
        if hard:
            xs = xt
            lift = gt[:, :1] * (1 - xs) + gt[:, 1:] * xs
            ut = lift + xs * (1 - xs) * ut
        dut = torch.autograd.grad(ut.sum(), xt, create_graph=True)[0]
        d2ut = torch.autograd.grad(dut.sum(), xt, create_graph=True)[0]  # (n*m,1)
        f_val = f_of(c, x.squeeze(-1))                           # (n, m)
        res = d2ut.reshape(n_batch, m_col) + f_val               # u'' + f = 0
        loss_r = torch.mean(res ** 2)
        if hard:
            loss = loss_r
            loss_b = torch.zeros((), device=DEV)
        else:
            ub = predict(model, v, g, x_bnd, hard=False)         # (n,2)
            loss_b = torch.mean((ub - g) ** 2)
            loss = loss_r + lam * loss_b
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    # evaluation on held-out instances
    gen_ev = torch.Generator().manual_seed(999)
    c_ev, g_ev = sample_instances(256, gen_ev)
    c_ev, g_ev = c_ev.to(DEV), g_ev.to(DEV)
    v_ev = torch.cat([c_ev, g_ev], dim=1)
    xg = torch.linspace(0, 1, 201, device=DEV).unsqueeze(-1)
    with torch.no_grad():
        u_pred = predict(model, v_ev, g_ev, xg, hard)
        u_true = u_exact(c_ev, g_ev, xg.squeeze(-1))
        rel = (torch.norm(u_pred - u_true, dim=1)
               / torch.norm(u_true, dim=1)).median().item()
        ub = predict(model, v_ev, g_ev, x_bnd, hard)
        bnd = torch.sqrt(torch.mean((ub - g_ev) ** 2)).item()
    return dict(mode=mode, lam=lam, seed=seed, steps=steps, rel_l2_med=rel,
                bnd_rms=bnd, wall_s=round(time.time() - t0, 1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["lam"], r["seed"]))
    jobs = []
    for seed in range(10):
        for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
            jobs.append(("soft", lam, seed))
        jobs.append(("hard", None, seed))
    STEPS = 15000
    with OUT.open("a") as fh:
        for mode, lam, seed in jobs:
            if (mode, lam, seed) in done:
                continue
            rec = run_one_batched(mode, lam, seed, steps=STEPS)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{mode:5s} lam={str(lam):7s} seed={seed} "
                  f"rel={rec['rel_l2_med']:.3e} bnd={rec['bnd_rms']:.3e} "
                  f"({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
