"""
Matplotlib house style for the paper figures.

Goal: figures whose text matches a Times / mathptmx
LaTeX paper, with no Type-3 fonts (so arXiv is happy). Import and call
apply_house_style() at the top of any figure script, then use the palette
constants and the overhead_legend / save_figure helpers.

    from paper_style import apply_house_style, NAVY, AMBER, TEAL, overhead_legend, save_figure
    apply_house_style()

Adjust the serif fallback list if the paper uses a different body font.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House palette: navy-anchored, restrained, grayscale-safe. Avoid default
# matplotlib primary colors (they read as amateur).
NAVY = "#1F4E79"   # base / primary series
AMBER = "#E07B39"  # RL / highlight / second series
TEAL = "#2E7D6B"   # third series
INK = "#1A1A1A"    # near-black for text, labels, callouts, zero lines
GRAY = "#8A94A0"   # axes, ticks, gridlines ONLY (never annotation text)
FAINT = "#E6E9ED"  # fills, shaded regions

# Annotation/callout TEXT goes in a visible ink (INK) or a series color (NAVY /
# AMBER / TEAL), never GRAY. Pale-grey text the reader has to read is a defect.


def apply_house_style():
    """Times-metric serif text + STIX math + embedded TrueType (no Type-3)."""
    plt.rcParams.update({
        "savefig.dpi": 220,
        # Match a Times / mathptmx paper. Reorder this list if the paper's body
        # font differs (e.g. put "Latin Modern Roman" first for a CM paper).
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # fonttype 42 embeds TrueType outlines; the matplotlib default (3) emits
        # Type-3 fonts, which arXiv flags. Always 42 for submission-bound figures.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.labelsize": 10,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Open frame: drop the top and right spines for a clean consulting look.
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_ax(ax):
    """Apply per-axis finishing (call after plotting on each axis)."""
    ax.tick_params(length=3, width=0.8)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)


def overhead_legend(fig, handles=None, labels=None, ncol=None, y=1.0):
    """One shared legend, centered ABOVE the panels, outside the plotting area.

    The user wants legends overhead and a single shared legend across panels,
    never a box dropped on top of the data.
    """
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    if ncol is None:
        ncol = len(handles)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.4)


def exhibit_title(fig, title, subtitle=None, x=0.055):
    """DEPRECATED. Do not draw an in-figure action title or grey subtitle.

    This author wants caption-led, chrome-free figures: NO floating headline, NO
    grey subtitle, NO in-figure source line. The LaTeX \\caption is the only place
    the exhibit is named. This stub is kept so older scripts do not crash; it draws
    nothing. (Reverses earlier "action title" guidance, after repeated corrections.)
    """
    import warnings
    warnings.warn(
        "exhibit_title is deprecated: figures are caption-led, no in-figure title. "
        "Name the exhibit in the LaTeX caption instead.",
        stacklevel=2,
    )
    return None


def save_figure(fig, path):
    """Tight save. Pair with a visual check: render and LOOK before trusting it."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {path}")
