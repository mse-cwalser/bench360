"""Regenerate the end-to-end Pareto figure for the paper (figures/pareto_e2e_stacked.pdf).

Fixes relative to visualization_pareto.ipynb:
  * Kleister parsing energy is amortized over the 500 documents actually profiled,
    not over the 337 documents used for extraction (the notebook divided the
    500-document parsing total by 337, inflating Kleister OCR energy by 1.48x).
  * Arctic-TILT + DeepSeek-OCR 2 on VRDU is dropped: no inference run exists for
    that configuration, so its "end-to-end" energy equalled its parsing energy.
  * VRDU "Tesseract" points are the real `vrdu_tesseract` runs. The notebook's
    category function had a catch-all that labelled every task other than docling
    and deepseek as "Tesseract", so `vrdu_default` -- the OCR shipped with the
    VRDU benchmark -- was plotted as Tesseract while being charged Tesseract's
    energy. The dataset-provided OCR is excluded here: its production energy is
    not attributable to our pipeline.
  * Side-by-side panels sized for a two-column figure*, with model names matching
    the paper. All configurations are drawn at full strength -- the dominated
    points carry evidence of their own -- and the frontier is marked by enlarging
    those points, ringing them in black, and joining them with a step line.

Run from the repository root:  python visualizations/make_pareto_acl.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

SRC = Path(__file__).resolve().parent.parent
OUT = SRC.parent / "ACL-IE-Energy" / "figures" / "pareto_e2e_stacked.pdf"

# Parsing energy in Wh for the full profiling run, from
# benchmark/ocr_profiling/report/*.csv, amortized over the 500 profiled documents.
OCR_WH_PER_DOC = {
    "VRDU": {"Tesseract": 4.867 / 500, "Docling": 12.947 / 500, "DeepSeek": 76.409 / 500},
    "Kleister-NDA": {"Tesseract": 12.909 / 500, "Docling": 4.385 / 500, "DeepSeek": 220.488 / 500},
}

# (label, group, end-to-end Wh per document, exact match %)
# group drives colour/marker; None energy means the configuration was not measured.
VRDU = [
    ("Arctic-TILT", "arctic-tess", 0.0197, 64.6),
    ("Arctic-TILT", "arctic-doc", 0.0331, 57.9),
    ("Llama-3.2-1B", "text-tess", 0.0135, 16.5),
    ("Llama-3.2-1B", "text-doc", 0.0301, 18.6),
    ("Llama-3.2-1B", "text-ds", 0.1575, 11.2),
    ("Llama-3.2-3B", "text-tess", 0.0183, 49.1),
    ("Llama-3.2-3B", "text-doc", 0.0334, 45.8),
    ("Llama-3.2-3B", "text-ds", 0.1659, 55.9),
    ("Ministral-3-3B", "text-tess", 0.0173, 54.2),
    ("Ministral-3-3B", "text-doc", 0.0320, 55.0),
    ("Ministral-3-3B", "text-ds", 0.1626, 67.6),
    ("Mistral-7B", "text-tess", 0.0317, 46.9),
    ("Mistral-7B", "text-doc", 0.0448, 48.1),
    ("Mistral-7B", "text-ds", 0.1811, 55.1),
    ("NuExtract-2.0-4B", "nuextract", 0.0327, 78.3),
    ("Qwen3-0.6B", "text-tess", 0.0132, 39.6),
    ("Qwen3-0.6B", "text-doc", 0.0286, 36.5),
    ("Qwen3-0.6B", "text-ds", 0.1573, 50.7),
    ("Qwen3-1.7B", "text-tess", 0.0155, 53.3),
    ("Qwen3-1.7B", "text-doc", 0.0305, 50.7),
    ("Qwen3-1.7B", "text-ds", 0.1600, 64.4),
    ("Qwen3-4B", "text-tess", 0.0210, 58.4),
    ("Qwen3-4B", "text-doc", 0.0357, 56.1),
    ("Qwen3-4B", "text-ds", 0.1680, 73.4),
    ("Qwen3-8B", "text-tess", 0.0277, 58.2),
    ("Qwen3-8B", "text-doc", 0.0408, 55.5),
    ("Qwen3-8B", "text-ds", 0.1775, 67.8),
    ("Qwen3-VL-2B", "vision", 0.0144, 65.1),
    ("Qwen3-VL-4B", "vision", 0.0222, 65.6),
    ("Qwen3-VL-8B", "vision", 0.0635, 71.3),
]

KLEISTER = [
    ("Arctic-TILT", "arctic-ds", 0.4421, 81.5),
    ("Arctic-TILT", "arctic-doc", 0.0486, 91.9),
    ("Arctic-TILT", "arctic-tess", 0.0654, 92.4),
    ("Llama-3.2-1B", "text-ds", 0.4502, 47.6),
    ("Llama-3.2-1B", "text-doc", 0.0156, 42.0),
    ("Llama-3.2-1B", "text-tess", 0.0327, 42.9),
    ("Llama-3.2-3B", "text-ds", 0.4651, 56.5),
    ("Llama-3.2-3B", "text-doc", 0.0261, 59.8),
    ("Llama-3.2-3B", "text-tess", 0.0434, 59.4),
    ("Ministral-3-3B", "text-ds", 0.4638, 71.2),
    ("Ministral-3-3B", "text-doc", 0.0233, 73.2),
    ("Ministral-3-3B", "text-tess", 0.0405, 75.9),
    ("Mistral-7B", "text-ds", 0.5024, 72.3),
    ("Mistral-7B", "text-doc", 0.0483, 70.9),
    ("Mistral-7B", "text-tess", 0.0666, 71.0),
    ("NuExtract-2.0-4B", "nuextract", 0.0666, 52.1),
    ("Qwen3-0.6B", "text-ds", 0.4500, 57.6),
    ("Qwen3-0.6B", "text-doc", 0.0141, 56.0),
    ("Qwen3-0.6B", "text-tess", 0.0315, 61.3),
    ("Qwen3-1.7B", "text-ds", 0.4571, 72.3),
    ("Qwen3-1.7B", "text-doc", 0.0194, 71.6),
    ("Qwen3-1.7B", "text-tess", 0.0365, 73.1),
    ("Qwen3-4B", "text-ds", 0.4740, 74.7),
    ("Qwen3-4B", "text-doc", 0.0291, 75.4),
    ("Qwen3-4B", "text-tess", 0.0464, 76.9),
    ("Qwen3-8B", "text-ds", 0.4959, 71.9),
    ("Qwen3-8B", "text-doc", 0.0436, 71.2),
    ("Qwen3-8B", "text-tess", 0.0603, 74.4),
    ("Qwen3-VL-2B", "vision", 0.0280, 56.5),
    ("Qwen3-VL-4B", "vision", 0.0569, 68.5),
    ("Qwen3-VL-8B", "vision", 0.1571, 66.6),
]

TEXT = "#3f8f6b"
VIS = "#e07a3f"
SPEC = "#7b52c4"

STYLE = {
    "text-tess": (TEXT, "o", "Text-only + Tesseract"),
    "text-doc": (TEXT, "D", "Text-only + Docling"),
    "text-ds": (TEXT, "s", "Text-only + DeepSeek-OCR 2"),
    "vision": (VIS, "^", "Vision-language (no parser)"),
    "arctic-tess": (SPEC, "o", "Arctic-TILT + Tesseract"),
    "arctic-doc": (SPEC, "D", "Arctic-TILT + Docling"),
    "arctic-ds": (SPEC, "s", "Arctic-TILT + DeepSeek-OCR 2"),
    "nuextract": (SPEC, "^", "NuExtract-2.0-4B"),
}


def pareto(points, exclude=()):
    """Points on the upper-left frontier: cheapest first, accuracy strictly increasing.

    `exclude` names groups that are ineligible for the frontier. Arctic-TILT is
    fine-tuned on Kleister-NDA by its authors, so on that dataset it is evaluated
    in-domain and must not be compared against the zero-shot models.
    """
    best, front = -1.0, []
    for p in sorted(points, key=lambda r: r[2]):
        if p[1] in exclude:
            continue
        if p[3] > best:
            front.append(p)
            best = p[3]
    return front


def panel(ax, points, title, label_offsets, xticks, exclude=()):
    front = pareto(points, exclude)
    front_keys = {(p[2], p[3]) for p in front}

    # Every configuration is drawn at full strength -- the dominated points are
    # evidence too (e.g. the stranded DeepSeek-OCR cluster). The frontier is marked
    # by adding a ring and the step line, not by suppressing everything else.
    seen = set()
    for name, group, energy, acc in points:
        color, marker, legend = STYLE[group]
        on_front = (energy, acc) in front_keys
        ax.scatter(
            energy, acc,
            color=color, marker=marker,
            s=68 if on_front else 38,
            alpha=1.0,
            # Thin white outline keeps overlapping points in the dense clusters apart.
            edgecolors="black" if on_front else "white",
            linewidths=1.0 if on_front else 0.5,
            zorder=5 if on_front else 3,
            label=legend if legend not in seen else None,
        )
        seen.add(legend)

    ax.step([p[2] for p in front], [p[3] for p in front],
            where="post", color="#3b3b3b", linestyle="--", linewidth=1.2, zorder=4)

    # Disambiguate: the same model can sit on the frontier with two different parsers.
    repeated = {n for n in (p[0] for p in front)
                if sum(1 for q in front if q[0] == n) > 1}
    suffix = {"text-tess": "\n+ Tesseract", "text-doc": "\n+ Docling", "text-ds": "\n+ DeepSeek",
              "arctic-tess": "\n+ Tesseract", "arctic-doc": "\n+ Docling", "arctic-ds": "\n+ DeepSeek"}

    for name, group, energy, acc in front:
        text = name + (suffix.get(group, "") if name in repeated else "")
        dx, dy, ha = label_offsets.get((name, group), (6, 5, "left"))
        ax.annotate(text, xy=(energy, acc), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=7.0, zorder=6,
                    linespacing=1.15)

    ax.set_xscale("log")
    # Explicit decimal ticks: matplotlib's default log minor labels collide at this size.
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # Extra head-room on both sides so annotations near the extremes are not clipped.
    lo = min(p[2] for p in points)
    hi = max(p[2] for p in points)
    ax.set_xlim(lo * 0.70, hi * 1.45)
    ax.set_xlabel("End-to-end energy per document (Wh, log scale)", fontsize=9)
    ax.set_ylabel("Avg. field exact match (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(True, which="major", linestyle=":", alpha=0.45)
    ax.tick_params(labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

panel(axes[0], VRDU, "VRDU (scanned, layout-rich)", {
    ("Qwen3-0.6B", "text-tess"): (2, -17, "center"),
    ("Qwen3-VL-2B", "vision"): (-2, 9, "center"),
    ("Qwen3-VL-4B", "vision"): (8, -6, "left"),
    ("NuExtract-2.0-4B", "nuextract"): (0, 10, "center"),
}, xticks=[0.01, 0.02, 0.05, 0.1, 0.2])
panel(axes[1], KLEISTER, "Kleister-NDA (born-digital, text-heavy)", {
    ("Qwen3-0.6B", "text-doc"): (0, -25, "center"),
    ("Qwen3-1.7B", "text-doc"): (-11, 6, "right"),
    ("Ministral-3-3B", "text-doc"): (-2, -27, "center"),
    ("Qwen3-4B", "text-doc"): (-4, 8, "center"),
    ("Ministral-3-3B", "text-tess"): (2, -25, "center"),
    ("Qwen3-4B", "text-tess"): (10, 2, "left"),
}, xticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
   exclude=("arctic-tess", "arctic-doc", "arctic-ds"))

# Arctic-TILT is shown on the Kleister panel but excluded from the frontier:
# it is fine-tuned on Kleister-NDA by its authors, so it is not a zero-shot peer.
axes[1].annotate("Arctic-TILT: in-domain,\nexcluded from frontier",
                 xy=(0.0486, 91.9), xytext=(-104, -2), textcoords="offset points",
                 fontsize=7.0, style="italic", color=SPEC, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=SPEC, lw=0.6,
                                 shrinkA=2, shrinkB=4))

handles, labels = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
for h, l in zip(h2, l2):
    if l not in labels:
        handles.append(h)
        labels.append(l)
order = [
    "Text-only + Tesseract", "Text-only + Docling", "Text-only + DeepSeek-OCR 2",
    "Vision-language (no parser)", "Arctic-TILT + Tesseract", "Arctic-TILT + Docling",
    "Arctic-TILT + DeepSeek-OCR 2", "NuExtract-2.0-4B",
]
pairs = {l: h for h, l in zip(handles, labels)}
fig.legend([pairs[l] for l in order if l in pairs],
           [l for l in order if l in pairs],
           loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4,
           fontsize=8, frameon=True)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

for name, pts, excl in (("VRDU", VRDU, ()),
                        ("Kleister-NDA", KLEISTER,
                         ("arctic-tess", "arctic-doc", "arctic-ds"))):
    print(f"\n{name} frontier (zero-shot models only):")
    for label, group, e, a in pareto(pts, excl):
        print(f"  {label:20s} {STYLE[group][2]:30s} {e:.4f} Wh  {a:.1f} EM")
