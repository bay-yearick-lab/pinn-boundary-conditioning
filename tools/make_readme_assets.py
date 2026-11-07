"""Build the README hero banner and result cards from the repository's own data.

    python tools/make_readme_assets.py

Inputs (all relative to the repository root):
  paper/figures/*.pdf          paper figures (rendered into the hero banner)
  data/pinn_sweep.jsonl        2D Poisson tanh-PINN weight sweep + hard twin
  data/burgers_sweep.jsonl     forced viscous Burgers sweep + hard twin
  data/operator_sweep.jsonl    physics-informed DeepONet sweep + lifted twin
  data/trained_3d.json         GPU-trained 3D PINN sweeps + hard twins
  data/linear_frontier.json    exact leak law in the fixed-feature regime
  data/diagnostic_3d.json      a-priori vs measured leak-conditioning product

Outputs: assets/hero.png (1600 x 560) and assets/card_*.png.
Requires numpy, matplotlib and pymupdf (for rasterizing the paper PDFs).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "paper" / "figures"
OUT = ROOT / "assets"
STYLE = ROOT.parent / "_style" / "lab.mplstyle"

INK, TEAL, AMBER, SLATE, CORAL, GRAY = ("#1B2A41", "#0F9D8A", "#F2A541",
                                       "#3D5A80", "#E07A5F", "#8D99AE")


def use_style():
    """Shared lab style if present; otherwise an equivalent inline fallback."""
    if STYLE.exists():
        plt.style.use(str(STYLE))
    else:
        plt.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
            "axes.edgecolor": INK, "axes.labelcolor": INK,
            "axes.titleweight": "bold", "axes.titlesize": 15,
            "axes.labelsize": 13, "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": True,
            "grid.color": "#E6E9EF", "grid.linewidth": 0.8,
            "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
            "savefig.dpi": 200, "savefig.bbox": "tight",
        })


def jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def medians(records, key="rel_l2", steps=None, dtype=None):
    """Median / min / max of `key` per (mode, lam), filtered on the full record key."""
    groups = defaultdict(list)
    for r in records:
        if steps is not None and r.get("steps") != steps:
            continue
        if dtype is not None and r.get("dtype", dtype) != dtype:
            continue
        groups[(r["mode"], r.get("lam"))].append(r[key])
    return {k: (float(np.median(v)), min(v), max(v), len(v)) for k, v in groups.items()}


def render_pdf(name, dpi=260, crop=None, vcrop=None):
    """Rasterize the first page of a paper figure PDF to an RGB array."""
    import pymupdf
    page = pymupdf.open(FIGS / name)[0]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    if crop is not None:                      # crop = (x0, x1) as width fractions
        x0, x1 = (int(c * img.shape[1]) for c in crop)
        img = img[:, x0:x1]
    if vcrop is not None:                     # vcrop = (y0, y1) as height fractions
        y0, y1 = (int(c * img.shape[0]) for c in vcrop)
        img = img[y0:y1]
    return img


# --------------------------------------------------------------------------- #
def hero():
    plt.rcParams["savefig.bbox"] = "standard"      # exact 1600 x 560 canvas
    fig = plt.figure(figsize=(8.0, 2.8), dpi=200)
    fig.patch.set_facecolor("white")

    band = fig.add_axes([0, 0, 0.415, 1])
    band.set_axis_off()
    band.add_patch(plt.Rectangle((0, 0), 1, 1, color=INK, transform=band.transAxes))
    band.add_patch(plt.Rectangle((0.075, 0.665), 0.11, 0.012, color=TEAL,
                                 transform=band.transAxes))
    band.text(0.075, 0.74, "pinn-boundary-conditioning", color="white",
              fontsize=12.5, fontweight="bold", transform=band.transAxes)
    band.text(0.075, 0.60,
              "Soft boundary conditions in physics-informed\n"
              "neural networks as a quadratic penalty method:\n"
              "exact leakage law, conditioning valley, and\n"
              "constraint enforcement by lifting, multipliers\n"
              "and consistent penalties",
              color="#DCE3EC", fontsize=8.0, linespacing=1.45, va="top",
              transform=band.transAxes)
    band.text(0.075, 0.105, "Yong Yi Bay  ·  Kathleen A. Yearick", color="white",
              fontsize=7.6, transform=band.transAxes)
    band.text(0.075, 0.045, "PINN  ·  DeepONet  ·  PyTorch autograd  ·  GPU",
              color=TEAL, fontsize=6.6, transform=band.transAxes)

    panels = [
        (render_pdf("fig_pinn.pdf", crop=(0.0, 0.505), vcrop=(0.13, 1.0)),
         "tanh PINN: error vs penalty weight"),
        (render_pdf("fig_ladder.pdf"),
         "penalty stalls, escapes converge"),
    ]
    x0, w, gap = 0.435, 0.272, 0.012
    for i, (img, cap) in enumerate(panels):
        ax = fig.add_axes([x0 + i * (w + gap), 0.04, w, 0.80])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        fig.text(x0 + i * (w + gap) + 0.004, 0.86, cap, color=INK, fontsize=6.9,
                 fontweight="bold", ha="left")
    fig.savefig(OUT / "hero.png", dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    plt.rcParams["savefig.bbox"] = "tight"


# --------------------------------------------------------------------------- #
def card_valley():
    """2D Poisson tanh PINN: error vs boundary weight, soft vs hard vs multiplier."""
    m = medians(jsonl("pinn_sweep.jsonl"), steps=20000, dtype="float32")
    lams = sorted(l for (mode, l) in m if mode == "soft")
    soft = np.array([m[("soft", l)] for l in lams])
    al_lams = sorted(l for (mode, l) in m if mode == "ledger")
    al = np.array([m[("ledger", l)][0] for l in al_lams])
    hard_med = m[("hard", None)][0]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.fill_between(lams, soft[:, 1], soft[:, 2], color=AMBER, alpha=0.22, lw=0)
    ax.loglog(lams, soft[:, 0], "o-", color=AMBER, lw=2.4, ms=6,
              label="soft penalty, fixed weight (median, 5 seeds)")
    ax.loglog(al_lams, al, "s--", color=SLATE, lw=1.8, ms=5,
              label="augmented Lagrangian, same weight (median, 5 seeds)")
    ax.axhline(hard_med, color=TEAL, lw=2.8,
               label="hard constraint by lifting (median, 10 seeds)")
    i0, ib = 0, int(np.argmin(soft[:, 0]))
    ax.annotate(f"{soft[i0, 0]:.2f}", (lams[i0], soft[i0, 0]), xytext=(8, 2),
                textcoords="offset points", color=INK, fontsize=11)
    ax.annotate(f"valley {soft[ib, 0]:.1e}", (lams[ib], soft[ib, 0]),
                xytext=(-12, 22), textcoords="offset points", color=INK, fontsize=11)
    ax.annotate(f"hard {hard_med:.1e}  ({soft[ib, 0] / hard_med:.0f}x below the valley)",
                (lams[0], hard_med), xytext=(0, -17), textcoords="offset points",
                color=TEAL, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"boundary penalty weight $\lambda$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_ylim(5e-5, 2.0)
    ax.set_title("Penalty weight sets an error valley; lifting sits 9x below it",
                 loc="left", fontsize=13.5)
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(OUT / "card_valley.png")
    plt.close(fig)
    return soft[i0, 0], soft[ib, 0], hard_med


def card_hard_vs_soft():
    """Best tuned penalty vs structural constraint across five testbeds."""
    rows = []
    m = medians(jsonl("pinn_sweep.jsonl"), steps=20000, dtype="float32")
    rows.append(("2D Poisson, tanh PINN",
                 min(v[0] for (mo, _), v in m.items() if mo == "soft"), m[("hard", None)][0]))
    m = medians(jsonl("burgers_sweep.jsonl"), steps=20000)
    rows.append(("Viscous Burgers, tanh PINN",
                 min(v[0] for (mo, _), v in m.items() if mo == "soft"), m[("hard", None)][0]))
    m = medians(jsonl("operator_sweep.jsonl"), key="rel_l2_med", steps=15000)
    rows.append(("Poisson family, PI-DeepONet",
                 min(v[0] for (mo, _), v in m.items() if mo == "soft"), m[("hard", None)][0]))
    t3 = json.loads((DATA / "trained_3d.json").read_text())
    for key, label in (("P1_poisson", "3D Poisson, 6x128 PINN (GPU)"),
                       ("P2_stiff", "3D anisotropic 1:40:1600 (GPU)")):
        c = t3[key]["confirmation"]
        rows.append((label, c["best_soft_median_vs_fd"], c["hard_median_vs_fd"]))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(rows))[::-1]
    h = 0.36
    soft = np.array([r[1] for r in rows])
    hard = np.array([r[2] for r in rows])
    ax.barh(y + h / 2, soft, h, color=AMBER, label="best soft penalty over the weight sweep")
    ax.barh(y - h / 2, hard, h, color=TEAL, label="hard constraint (lifting), same budget")
    ax.set_xscale("log")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=11)
    ax.grid(axis="y", visible=False)
    for yi, s, hd in zip(y, soft, hard):
        ax.text(s * 1.25, yi + h / 2, f"{s:.1e}", va="center", fontsize=9.5, color=INK)
        ax.text(hd * 1.25, yi - h / 2, f"{hd:.1e}   {s / hd:.0f}x", va="center",
                fontsize=9.5, color=TEAL, fontweight="bold")
    ax.set_xlim(1e-5, 30)
    ax.set_xlabel(r"relative $L^2$ error (median over seeds)")
    ax.set_title("Structural constraint enforcement beats the best tuned penalty",
                 loc="left", fontsize=13.5)
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(OUT / "card_hard_vs_soft.png")
    plt.close(fig)
    return rows


def card_prediction():
    """Exact leak bracket (2D) and a-priori vs measured product limit (3D, 20 seeds)."""
    s = json.loads((DATA / "linear_frontier.json").read_text())["statics_2d"]
    lam, leak = np.array(s["lams"]), np.array(s["leaks"])
    rho, nu = s["rho"], s["nu"]
    d = json.loads((DATA / "diagnostic_3d.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    ax.fill_between(lam, rho / (lam + nu), rho / lam, color=TEAL, alpha=0.18, lw=0,
                    label=r"bracket $\rho/(\lambda+\nu)\leq$ leak $\leq\rho/\lambda$")
    ax.loglog(lam, rho / lam, color=TEAL, lw=1.4)
    ax.loglog(lam, rho / (lam + nu), color=TEAL, lw=1.4, ls="--")
    ax.loglog(lam, leak, "o", color=INK, ms=4, label=r"measured $\|Bw_\lambda-g\|$")
    ax.set_xlabel(r"penalty weight $\lambda$")
    ax.set_ylabel("boundary leak")
    ax.set_title("Exact leak law, every weight", loc="left", fontsize=13)
    ax.legend(loc="lower left", fontsize=10)

    ax = axes[1]
    summ = {r["testbed"]: r for r in d["summary"]}
    for tb, color, label in (("TB1_3d_poisson", TEAL, "3D Poisson"),
                             ("TB2_3d_stiff", SLATE, "3D anisotropic")):
        pred = np.array([r["confirm"]["product_pred"] for r in d["testbeds"][tb]])
        meas = np.array([r["confirm"]["product_meas"] for r in d["testbeds"][tb]])
        name = "3D Poisson" if tb == "TB1_3d_poisson" else "3D stiff anisotropic"
        err = 100 * summ[name]["product_relerr"]["median"]
        ax.loglog(pred, meas, "o", color=color, ms=7, alpha=0.9,
                  label=f"{label}, median error {err:.2f}%")
    lo, hi = 6e3, 1.5e5
    ax.loglog([lo, hi], [lo, hi], color=GRAY, lw=1.2, ls=":", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"predicted $\rho\,\sigma_B^2/\mu$ (before the sweep)")
    ax.set_ylabel(r"measured leak $\cdot\,\kappa(H_\lambda)$, large $\lambda$")
    ax.set_title("Product limit predicted a priori", loc="left", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    fig.savefig(OUT / "card_prediction.png")
    plt.close(fig)
    return {k: v["product_relerr"]["median"] for k, v in summ.items()}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    use_style()
    hero()
    print("valley card (left end, valley, hard):", card_valley())
    for r in card_hard_vs_soft():
        print(f"  {r[0]:34s} soft {r[1]:.3e}  hard {r[2]:.3e}  ratio {r[1] / r[2]:.1f}")
    print("product limit median rel. error:", card_prediction())
    print(f"wrote assets to {OUT.relative_to(ROOT)}/")
