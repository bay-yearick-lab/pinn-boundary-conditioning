"""Print the headline results, recomputed from the shipped run records in data/.

    python scripts/headline.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def medians(name, key="rel_l2", **filt):
    groups = defaultdict(list)
    for line in (DATA / name).read_text().splitlines():
        r = json.loads(line)
        if all(r.get(k) == v for k, v in filt.items()):
            groups[(r["mode"], r.get("lam"))].append(r[key])
    return {k: float(np.median(v)) for k, v in groups.items()}


def soft_best_and_hard(m):
    best_lam, best = min(((l, v) for (mode, l), v in m.items() if mode == "soft"),
                         key=lambda t: t[1])
    return best_lam, best, m[("hard", None)]


def main():
    print("HEADLINE RESULTS (median relative L2 error over seeds)")
    print("-" * 78)
    m = medians("pinn_sweep.jsonl", steps=20000, dtype="float32")
    lam, best, hard = soft_best_and_hard(m)
    print(f"2D Poisson tanh PINN   soft lam=1e-2 {m[('soft', 0.01)]:.2f} -> valley "
          f"{best:.1e} (lam={lam:g}); hard twin {hard:.1e} ({best / hard:.1f}x)")
    al = min(v for (mode, _), v in m.items() if mode == "ledger")
    print(f"                       augmented Lagrangian best {al:.1e} "
          f"({best / al:.1f}x below the best fixed penalty)")
    m = medians("burgers_sweep.jsonl", steps=20000)
    lam, best, hard = soft_best_and_hard(m)
    print(f"Viscous Burgers PINN   valley {best:.1e} (lam={lam:g}); hard twin {hard:.1e} "
          f"({best / hard:.1f}x)")
    m = medians("operator_sweep.jsonl", key="rel_l2_med", steps=15000)
    lam, best, hard = soft_best_and_hard(m)
    print(f"PI-DeepONet (10 seeds) valley {best:.1e} (lam={lam:g}); lifted {hard:.1e} "
          f"({best / hard:.1f}x)")
    d = json.loads((DATA / "trained_3d.json").read_text())
    for key, label in (("P1_poisson", "3D Poisson, GPU     "), ("P2_stiff", "3D anisotropic, GPU ")):
        c = d[key]["confirmation"]
        print(f"{label}   best soft {c['best_soft_median_vs_fd']:.1e}; hard twin "
              f"{c['hard_median_vs_fd']:.1e} ({c['hard_advantage_x']:.0f}x)")
    d = json.loads((DATA / "diagnostic_3d.json").read_text())
    for s in d["summary"]:
        print(f"{s['testbed']:<22} product limit rho*sB^2/mu predicted a priori: median "
              f"error {100 * s['product_relerr']['median']:.2f}% "
              f"(max {100 * s['product_relerr']['max']:.2f}%, {s['n_seeds']} seeds)")
    print("-" * 78)


if __name__ == "__main__":
    main()
