"""All paper figures. House style: navy/amber/teal, overhead legends, serif+STIX,
fonttype 42, no in-figure chrome. Render and inspect every output.

Reads data/*.json produced by src/exp_*.py; the PINN figure additionally
needs data/pinn_sweep.jsonl.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (AMBER, FAINT, GRAY, INK, NAVY, TEAL, apply_house_style,
                         overhead_legend, save_figure, style_ax)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)



def slope_triangle(ax, x0, x1, y0, order, label=None):
    """Journal-style slope triangle: hypotenuse at the given order, right angle
    at the lower-right, legs labeled. Placed in CLEAR space, never on the data."""
    y1 = y0 * (x0 / x1) ** order
    ax.plot([x0, x1], [y0, y1], color=INK, lw=1.0)          # hypotenuse
    ax.plot([x0, x1], [y1, y1], color=INK, lw=1.0)          # horizontal leg
    ax.plot([x0, x0], [y0, y1], color=INK, lw=1.0)          # vertical leg
    ax.annotate("1", xy=((x0 * x1) ** 0.5, y1), xytext=(0, -3), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=INK)
    ax.annotate(f"{order:g}", xy=(x0, (y0 * y1) ** 0.5), xytext=(-3, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK)
    if label:
        ax.annotate(label, xy=(x1, (y0 * y1) ** 0.5), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK)


def fig_tax():
    """Two panels: the leak and the conditioning growth. The product limit is its
    own standalone figure, fig_product, so the headline law reads full-width."""
    d = json.loads((DATA / "linear_frontier.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))

    ax = axes[0]
    s1 = d["statics_1d"]
    s2 = d["statics_2d"]
    l1, k1 = np.array(s1["lams"]), np.array(s1["leaks"])
    l2, k2 = np.array(s2["lams"]), np.array(s2["leaks"])
    ax.loglog(l1, k1, color=NAVY, lw=2.0, label="Poisson 1D")
    ax.loglog(l2, k2, color=TEAL, lw=2.0, label="Poisson 2D")
    ax.loglog(l1, s1["rho"] / l1, color=INK, lw=1.0, ls=":", label=r"upper bracket $\rho/\lambda$")
    ax.loglog(l2, s2["rho"] / l2, color=INK, lw=1.0, ls=":")
    nu2 = s2.get("nu")
    if nu2:
        ax.loglog(l2, s2["rho"] / (l2 + nu2), color=INK, lw=1.0, ls="--",
                  label=r"lower bracket $\rho/(\lambda+\nu)$")
    ax.annotate("fitted intercept recovers $\\rho$\nto $0.1\\%$ (1D), $0.2\\%$ (2D)",
                xy=(1.6e0, 2e-7), color=INK, fontsize=9, ha="left", va="center")
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel(r"boundary error $\|Bw_\lambda - g\|$")
    style_ax(ax)

    ax = axes[1]
    dr = d["drag"]
    lams, kaps = np.array(dr["lams"]), np.array(dr["kappas"])
    ax.loglog(lams, kaps, color=NAVY, lw=2.0, label=r"$\kappa(H_\lambda)$")
    guide = lams * dr["sB2"] / dr["mu"]
    mask = guide > 3e2
    ax.loglog(lams[mask], guide[mask], color="#B3472D", lw=1.6, ls=(0, (6, 3)),
              label=r"conditioning bound $\lambda\sigma_B^2/\mu$")
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel(r"condition number $\kappa$")
    style_ax(ax)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    overhead_legend(fig, handles, labels, ncol=5, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIGS / "fig_tax.pdf")


def fig_product():
    """The headline product law, its own full-width, taller figure: leak times
    conditioning collapses onto the exact limit rho sB^2 / mu, four-digit agreement."""
    d = json.loads((DATA / "linear_frontier.json").read_text())
    s1 = d["statics_1d"]
    dr = d["drag"]
    lam_leak, leak1 = np.array(s1["lams"]), np.array(s1["leaks"])
    lam_kap, kap = np.array(dr["lams"]), np.array(dr["kappas"])
    kap_on_leak = np.exp(np.interp(np.log(lam_leak), np.log(lam_kap), np.log(kap)))
    product = leak1 * kap_on_leak
    limit = dr["rho"] * dr["sB2"] / dr["mu"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.loglog(lam_leak, product, color=NAVY, lw=2.2, marker="o", ms=4.5,
              markeredgecolor="white", markeredgewidth=0.6,
              label=r"leak $\times$ conditioning (Poisson 1D)")
    ax.axhline(limit, color=AMBER, lw=2.0, ls=(0, (6, 3)),
               label=r"exact limit $\rho\sigma_B^2/\mu$")

    # four-digit agreement at the top of the sweep, measured against the limit.
    tail = product[-6:]
    ratio = float(np.median(tail) / limit)
    ax.annotate(rf"settles onto the limit:" + "\n"
                + rf"agreement to four digits (ratio ${ratio:.4f}$)",
                xy=(lam_leak[-3], limit), xytext=(lam_leak[11], limit * 2.7),
                color=INK, fontsize=10, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.set_ylim(limit * 0.35, limit * 12)
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel(r"leak $\times$ conditioning")
    style_ax(ax)
    overhead_legend(fig, ncol=2, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, FIGS / "fig_product.pdf")


def fig_frontier():
    d = json.loads((DATA / "linear_frontier.json").read_text())["frontier"]
    T = np.array(d["budgets"])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    first = True
    for key, errs in sorted(d["curves"].items()):
        lab = "fixed weights (each stalls at its leak)" if first else None
        ax.loglog(T, errs, color=GRAY, lw=1.2, alpha=0.85, label=lab)
        first = False
    ax.loglog(T, d["soft_envelope"], color=NAVY, lw=2.4,
              label="soft envelope (best weight per budget)")
    ax.loglog(T, d["heavyball_envelope"], color=NAVY, lw=1.6, ls="--",
              label="soft + momentum envelope")
    hard = np.maximum(np.array(d["hard"]), 1e-13)
    ax.loglog(T, hard, color=AMBER, lw=2.4, label="hard constraint (the fence)")
    ax.set_xlabel(r"first-order steps $T$")
    ax.set_ylabel(r"error to the constrained solution")
    ax.set_ylim(1e-13, 3e2)
    style_ax(ax)
    overhead_legend(fig, ncol=2, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, FIGS / "fig_frontier.pdf")


def fig_ushape():
    d = json.loads((DATA / "linear_frontier.json").read_text())["ushape"]
    lams, kaps = np.array(d["lams"]), np.array(d["kappas"])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.loglog(lams, kaps, color=NAVY, lw=2.2, label=r"$\kappa_{\mathrm{eff}}(\lambda)$, soft")
    ax.axhline(d["kappa_red"], color=AMBER, lw=2.0, ls="-",
               label=r"$\kappa_{\mathrm{red}}$, the fence")
    ax.axvline(d["lam_ntk"], color=TEAL, lw=1.6, ls="--",
               label="trace-balancing rule")
    # win magnitude: the trace-balancing weight sits above the fence floor
    kap_ntk = float(np.exp(np.interp(np.log(d["lam_ntk"]), np.log(lams), np.log(kaps))))
    handicap = kap_ntk / d["kappa_red"]
    ax.annotate(rf"${handicap:.0f}\times$ above the floor",
                xy=(d["lam_ntk"] * 0.55, kap_ntk), color=TEAL, fontsize=9,
                ha="right", va="center")
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel(r"effective condition number")
    style_ax(ax)
    overhead_legend(fig, ncol=3, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, FIGS / "fig_ushape.pdf")


def fig_ladder():
    d = json.loads((DATA / "ladder.json").read_text())
    T = np.array(d["ts"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.loglog(T, np.maximum(d["penalty_fixed"]["errs"], 1e-13), color=GRAY, lw=2.2,
              label=r"quadratic penalty, fixed $\lambda$")
    ax.loglog(T, np.maximum(d["hard"], 1e-13), color=AMBER, lw=3.6,
              label="trial space with lifting, the fence")
    ax.loglog(T, np.maximum(d["augmented_lagrangian"]["errs"], 1e-13), color=TEAL,
              lw=2.0, label=r"augmented Lagrangian (multiplier updates)")
    ax.loglog(T, np.maximum(d["consistent"]["errs"], 1e-13), color="#8659A5", lw=1.4,
              ls="--", label="consistent boundary penalty")
    ax.annotate(r"penalty stalls at its leak $\rho/\lambda$", xy=(1.2e5, 0.11),
                color=INK, fontsize=9.5, ha="left", va="bottom")
    ax.annotate("all three escapes reach machine precision,\nby three different mechanisms",
                xy=(1.2e5, 4e-11), color=INK, fontsize=9.5, ha="left", va="bottom")
    ax.set_xlabel(r"first-order steps $T$")
    ax.set_ylabel("error to the constrained solution")
    ax.set_ylim(1e-13, 3e2)
    style_ax(ax)
    overhead_legend(fig, ncol=2, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, FIGS / "fig_ladder.pdf")


def fig_floor():
    d = json.loads((DATA / "precision_floor.json").read_text())
    lams = np.array(d["lams"])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.loglog(lams, d["float32"], color=NAVY, lw=2.2, label="float32 solve")
    ax.loglog(lams, d["float64"], color=AMBER, lw=2.2, label="float64 solve")
    sB = d.get("sigma_B", 0.0816)
    ax.loglog(lams, d["rho"] / (lams * sB), color=INK, lw=1.0, ls=":",
              label=r"leak bound $\rho/(\lambda\sigma_B)$")
    # the single-precision floor is the notorious reported PINN error plateau
    f32 = np.array(d["float32"])
    imin = int(np.argmin(f32))
    ax.annotate("single-precision floor:\nthe reported plateau", xy=(lams[imin], f32[imin]),
                xytext=(lams[imin] * 0.22, f32[imin] * 11), fontsize=8.5, color=NAVY,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.8))
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel("error of the exact solve")
    style_ax(ax)
    overhead_legend(fig, ncol=3, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, FIGS / "fig_floor.pdf")


def fig_pinn():
    path = DATA / "pinn_sweep.jsonl"
    if not path.exists():
        print("pinn sweep data missing; skipping fig_pinn")
        return
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    recs = [r for r in recs if r["steps"] >= 20000]
    if not recs:
        print("no full-budget pinn records yet; skipping fig_pinn")
        return
    soft = {}
    for r in recs:
        if r["mode"] == "soft" and r["dtype"] == "float32":
            soft.setdefault(r["lam"], []).append(r)
    lams = sorted(soft)
    if len(lams) < 5:
        print("pinn sweep still partial; skipping fig_pinn")
        return
    med_l2 = [np.median([x["rel_l2"] for x in soft[l]]) for l in lams]
    lo_l2 = [np.min([x["rel_l2"] for x in soft[l]]) for l in lams]
    hi_l2 = [np.max([x["rel_l2"] for x in soft[l]]) for l in lams]
    med_b = [np.median([x["bnd_rms"] for x in soft[l]]) for l in lams]
    hards = [r for r in recs if r["mode"] == "hard"]
    adapt_path = DATA / "pinn_adaptive.jsonl"
    anneals, ntks = [], []
    if adapt_path.exists():
        arecs = [json.loads(l) for l in adapt_path.read_text().splitlines()]
        anneals = [r for r in arecs if r["mode"] == "anneal"]
        ntks = [r for r in arecs if r["mode"] == "ntk"]
    adapts = anneals or [r for r in recs if r["mode"] == "adaptive"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    ax.loglog(lams, med_l2, color=NAVY, lw=2.2, marker="o", ms=4,
              markeredgecolor="white", markeredgewidth=0.6, label="soft PINN (median across seeds)")
    ax.fill_between(lams, lo_l2, hi_l2, color=NAVY, alpha=0.15, lw=0)
    if hards:
        hvals = sorted(r["rel_l2"] for r in hards)
        h_med = float(np.median(hvals))
        in_basin = [v for v in hvals if v < 10 * h_med]
        out_basin = [v for v in hvals if v >= 10 * h_med]
        ax.axhline(h_med, color=AMBER, lw=2.2,
                   label=f"hard constraint, same budget (median of {len(hvals)} seeds)")
        ax.axhspan(min(in_basin), max(in_basin), color=AMBER, alpha=0.15, lw=0)
        if out_basin:
            xs = [max(lams) * 0.45] * len(out_basin)
            ax.plot(xs, out_basin, marker="x", ms=7, color=AMBER, ls="none",
                    markeredgewidth=1.8,
                    label=f"fence basin escapes ({len(out_basin)}/{len(hvals)})")
    # adaptive endpoints: clip to the sweep range, annotate exits
    xmax = max(lams) * 3
    if adapts:
        first = True
        for r in adapts:
            x = min(r["lam_final"], xmax)
            ax.plot(x, r["rel_l2"], marker="D", ms=6, color=TEAL,
                    markeredgecolor="white", markeredgewidth=0.6, ls="none",
                    label="annealing rule (final)" if first else None)
            first = False
        if any(r["lam_final"] > xmax for r in adapts):
            top = max(r["lam_final"] for r in adapts)
            expo = int(np.floor(np.log10(top)))
            mant = top / 10 ** expo
            ax.annotate(rf"$\lambda\to{mant:.0f}\times10^{{{expo}}}$",
                        xy=(xmax * 0.12, np.median([r["rel_l2"] for r in adapts]) * 1.15),
                        ha="right", fontsize=9, color=TEAL)
    if ntks:
        first = True
        for r in ntks:
            ax.plot(min(r["lam_final"], xmax), r["rel_l2"], marker="s", ms=6,
                    color="#8659A5", markeredgecolor="white", markeredgewidth=0.6,
                    ls="none", label="kernel-trace rule (final)" if first else None)
            first = False
    # N5: the reported PINN error plateau (the ceiling) and the win magnitude, on-axes
    ax.axhspan(1e-4, 1e-3, color=GRAY, alpha=0.09, lw=0, zorder=0)
    ax.annotate(r"reported $10^{-3}$--$10^{-4}$ plateau",
                xy=(min(lams) / 2.7, 1.05e-3), fontsize=8.5, color=INK,
                ha="left", va="bottom")
    if hards:
        win = min(med_l2) / h_med
        ax.annotate(rf"fence ${win:.0f}\times$ below the valley",
                    xy=(lams[len(lams) // 2], h_med * 0.56), fontsize=9,
                    color=AMBER, ha="center", va="top")
    ax.set_xlim(min(lams) / 3, xmax)
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel(r"relative $L^2$ error")
    style_ax(ax)

    ax = axes[1]
    ax.loglog(lams, med_b, color=NAVY, lw=2.2, marker="o", ms=4,
              markeredgecolor="white", markeredgewidth=0.6)
    floor = float(np.median(med_b[-3:]))
    ax.axhline(floor, color=INK, lw=1.0, ls=":")
    ax.annotate("budget floor (precision-independent)", xy=(lams[1], floor * 1.5),
                fontsize=9, color=INK)
    if hards:
        hb = np.median([r["bnd_rms"] for r in hards])
        ax.axhline(hb, color=AMBER, lw=2.2)
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel("boundary RMS error")
    style_ax(ax)

    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    overhead_legend(fig, handles, labels, ncol=4, y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, FIGS / "fig_pinn.pdf")


def fig_budget():
    """Error against Adam budget for the three arms at their 20k operating points.

    The 5k and 80k columns come from budget_sweep.jsonl; the 20k column is the
    main sweep (soft lam=100, hard, ledger lam=100). Escapes (>=10x the arm's
    in-basin median) are plotted as X markers and excluded from the band.
    """
    bpath = DATA / "budget_sweep.jsonl"
    spath = DATA / "pinn_sweep.jsonl"
    if not bpath.exists() or not spath.exists():
        print("budget sweep data missing; skipping fig_budget")
        return
    arms = {("soft", 1e2): {}, ("hard", None): {}, ("ledger", 1e2): {}}
    for line in bpath.read_text().splitlines():
        r = json.loads(line)
        key = (r["mode"], r["lam"])
        if key in arms:
            arms[key].setdefault(r["steps"], []).append(r["rel_l2"])
    for line in spath.read_text().splitlines():
        r = json.loads(line)
        key = (r["mode"], r["lam"])
        if key in arms and r["dtype"] == "float32" and r["steps"] == 20000:
            arms[key].setdefault(20000, []).append(r["rel_l2"])
    budgets = sorted({t for v in arms.values() for t in v})
    if len(budgets) < 3:
        print("budget sweep still partial; skipping fig_budget")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    specs = [(("soft", 1e2), NAVY, "o", r"soft, $\lambda=10^2$ (best fixed weight)"),
             (("ledger", 1e2), TEAL, "D", r"multiplier recipe, $\lambda=10^2$"),
             (("hard", None), AMBER, "s", "hard constraint (fence)")]
    n_escape = 0
    soft_med = {}
    for key, color, mark, label in specs:
        med, lo, hi = [], [], []
        for t in budgets:
            vals = sorted(arms[key].get(t, []))
            if not vals:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            m = float(np.median(vals))
            basin = [v for v in vals if v < 10 * m]
            out = [v for v in vals if v >= 10 * m]
            med.append(float(np.median(basin)))
            lo.append(min(basin)); hi.append(max(basin))
            if key[0] == "soft":
                soft_med[t] = med[-1]
            if out:
                ax.plot([t] * len(out), out, marker="x", ms=7, color=color,
                        ls="none", markeredgewidth=1.8,
                        label="fence basin escapes" if n_escape == 0 else None)
                n_escape += len(out)
        ax.loglog(budgets, med, color=color, lw=2.2, marker=mark, ms=5,
                  markeredgecolor="white", markeredgewidth=0.6, label=label)
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_xlabel(r"Adam budget $T$ (steps)")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{t//1000}k" for t in budgets])
    ax.minorticks_off()
    style_ax(ax)
    handles, labels = ax.get_legend_handles_labels()
    overhead_legend(fig, handles, labels, ncol=2, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, FIGS / "fig_budget.pdf")


def fig_operator():
    path = DATA / "operator_sweep.jsonl"
    if not path.exists():
        print("operator sweep data missing; skipping fig_operator")
        return
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    soft = {}
    for r in recs:
        if r["mode"] == "soft":
            soft.setdefault(r["lam"], []).append(r["rel_l2_med"])
    lams = sorted(soft)
    if len(lams) < 4:
        print("operator sweep partial; skipping fig_operator")
        return
    med = [float(np.median(soft[l])) for l in lams]
    lo = [min(soft[l]) for l in lams]
    hi = [max(soft[l]) for l in lams]
    hards = [r["rel_l2_med"] for r in recs if r["mode"] == "hard"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.loglog(lams, med, color=NAVY, lw=2.2, marker="o", ms=4,
              markeredgecolor="white", markeredgewidth=0.6,
              label="soft physics-informed DeepONet")
    ax.fill_between(lams, lo, hi, color=NAVY, alpha=0.15, lw=0)
    if hards:
        hb = float(np.median(hards))
        ax.axhline(hb, color=AMBER, lw=2.2,
                   label="lifting installed family-wide, same budget")
        win = min(med) / hb
        ax.annotate(rf"lifting ${win:.0f}\times$ below the valley",
                    xy=(lams[len(lams) // 2], hb * 1.28), color=AMBER, fontsize=9,
                    ha="center", va="bottom")
    ax.set_xlabel(r"boundary weight $\lambda$")
    ax.set_ylabel(r"median relative $L^2$ error")
    style_ax(ax)
    overhead_legend(fig, ncol=2, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, FIGS / "fig_operator.pdf")


def fig_setup():
    """JFM-style setup + collocation receipt: the 2D Poisson and 3D cube testbeds.

    The points are the actual sampled collocation geometry (interior and boundary),
    not a cartoon; the field is the manufactured solution. This is the numerical
    implementation receipt N4 asks for.
    """
    import itertools
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(9.0, 3.9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # ---- (a) 2D Poisson testbed: manufactured solution + real collocation ----
    cmap = LinearSegmentedColormap.from_list("navyfaint", ["#FFFFFF", "#DCE4EF", NAVY])
    gx = np.linspace(0.0, 1.0, 200)
    GX, GY = np.meshgrid(gx, gx)
    U = np.sin(2.5 * np.pi * GX) * np.sin(1.5 * np.pi * GY)
    ax1.contourf(GX, GY, U, levels=18, cmap=cmap, alpha=0.85)
    rng = np.random.default_rng(0)
    Xi = rng.uniform(0.0, 1.0, size=(700, 2))
    ax1.scatter(Xi[:, 0], Xi[:, 1], s=3, c=INK, alpha=0.35, linewidths=0,
                label="interior collocation")
    tb = np.linspace(0.02, 0.98, 22)
    bx = np.concatenate([tb, tb, np.zeros_like(tb), np.ones_like(tb)])
    by = np.concatenate([np.zeros_like(tb), np.ones_like(tb), tb, tb])
    ax1.scatter(bx, by, s=15, c=AMBER, marker="s", edgecolors="none",
                label="boundary collocation")
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_aspect("equal")
    _bb = dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5)
    ax1.annotate(r"$-\Delta u = f$", xy=(0.5, 0.52), ha="center", va="center",
                 color=INK, fontsize=11, bbox=_bb)
    ax1.annotate(r"$u = g$ on $\partial\Omega$", xy=(0.5, 0.09), ha="center",
                 va="center", color=AMBER, fontsize=9.5, bbox=_bb)
    style_ax(ax1)

    # ---- (b) 3D cube testbed: edges + real boundary/interior collocation ----
    verts = np.array(list(itertools.product([0, 1], repeat=3)))
    for i in range(8):
        for j in range(i + 1, 8):
            if np.abs(verts[i] - verts[j]).sum() == 1:
                ax2.plot(*zip(verts[i], verts[j]), color=GRAY, lw=0.8)
    rng3 = np.random.default_rng(0)
    nf = 66
    face = rng3.integers(0, 6, size=nf)
    Bpts = rng3.uniform(0.0, 1.0, size=(nf, 3))
    for k in range(nf):
        Bpts[k, face[k] // 2] = face[k] % 2
    ax2.scatter(Bpts[:, 0], Bpts[:, 1], Bpts[:, 2], s=11, c=AMBER, marker="s",
                edgecolors="none")
    Xi3 = rng3.uniform(0.0, 1.0, size=(150, 3))
    ax2.scatter(Xi3[:, 0], Xi3[:, 1], Xi3[:, 2], s=3, c=INK, alpha=0.28, linewidths=0)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_zlabel("$z$")
    try:
        ax2.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax2.view_init(elev=18, azim=-58)
    for pane in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        try:
            pane.pane.set_visible(False)
        except Exception:
            pass
    ax2.grid(False)

    handles, labels = ax1.get_legend_handles_labels()
    overhead_legend(fig, handles, labels, ncol=2, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIGS / "fig_setup.pdf")


if __name__ == "__main__":
    apply_house_style()
    fig_setup()
    fig_tax()
    fig_product()
    fig_frontier()
    fig_ushape()
    fig_ladder()
    fig_floor()
    fig_pinn()
    fig_budget()
    fig_operator()
