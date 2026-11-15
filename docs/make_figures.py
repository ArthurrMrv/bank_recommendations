"""README figure: headline results of the cloud run (values from notebooks/report.ipynb and reports/report.tex).

    uv run python docs/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREY, INK, MUTED = "#2a78d6", "#c9ccd1", "#1f2430", "#5a6270"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

PANELS = [
    ("Who is contacted\ncustomers per segment",
     ["Baseline", "Transactor", "Revolver\n(not contacted)"], [1207, 368, 425], [BLUE, BLUE, GREY], "{:,.0f}"),
    ("Should we contact\nuplift of an offer, in points",
     ["Everyone", "Top half by\npredicted uplift"], [8.3, 16.4], [GREY, BLUE], "{:.1f}"),
    ("Which single offer\nvalue per contact, € per month",
     ["Random offer", "Highest value", "Bandit", "Best achievable"], [3.61, 6.92, 7.56, 7.81],
     [GREY, GREY, BLUE, GREY], "€{:.2f}"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [3, 2, 4]})
for ax, (title, labels, values, colors, fmt) in zip(axes, PANELS):
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontweight="bold")
    ax.set_title(title, pad=14)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelcolor=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
fig.tight_layout(w_pad=3)
fig.savefig(Path(__file__).with_name("img") / "results.png", dpi=200, facecolor="white")
