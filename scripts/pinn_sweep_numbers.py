"""Print the trained-PINN numbers quoted in the paper from the completed sweeps."""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def med(xs):
    return float(np.median(xs))


def main():
    recs = [json.loads(l) for l in (DATA / "pinn_sweep.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r["steps"] >= 20000]
    soft = {}
    for r in recs:
        if r["mode"] == "soft" and r["dtype"] == "float32":
            soft.setdefault(r["lam"], []).append(r)
    print("lam    med relL2   [min,max]      med bnd")
    for lam in sorted(soft):
        g = soft[lam]
        print(f"{lam:8g} {med([x['rel_l2'] for x in g]):.3e} "
              f"[{min(x['rel_l2'] for x in g):.2e},{max(x['rel_l2'] for x in g):.2e}] "
              f"{med([x['bnd_rms'] for x in g]):.3e}")
    lam_star = min(soft, key=lambda l: med([x["rel_l2"] for x in soft[l]]))
    e_star = med([x["rel_l2"] for x in soft[lam_star]])
    print(f"valley: lam={lam_star} med relL2={e_star:.3e}")
    hard = [r for r in recs if r["mode"] == "hard"]
    if hard:
        eh = med([r["rel_l2"] for r in hard])
        bh = med([r["bnd_rms"] for r in hard])
        print(f"hard: med relL2={eh:.3e} (x{e_star/eh:.1f} better than valley), bnd={bh:.2e}")
    f64 = {r["lam"]: r for r in recs if r["dtype"] == "float64" and r["mode"] == "soft"}
    for lam, r in sorted(f64.items()):
        f32 = med([x["rel_l2"] for x in soft.get(lam, [])]) if lam in soft else float("nan")
        print(f"float64 spot lam={lam:8g}: relL2={r['rel_l2']:.3e} (float32 med {f32:.3e}), bnd={r['bnd_rms']:.2e}")
    ad = DATA / "pinn_adaptive.jsonl"
    if ad.exists():
        arecs = [json.loads(l) for l in ad.read_text().splitlines()]
        for rule in ("anneal", "ntk"):
            g = [r for r in arecs if r["mode"] == rule]
            if g:
                print(f"{rule}: med relL2={med([r['rel_l2'] for r in g]):.3e} "
                      f"med bnd={med([r['bnd_rms'] for r in g]):.3e} "
                      f"lam_final={[f'{r['lam_final']:.2g}' for r in g]}")


if __name__ == "__main__":
    main()
