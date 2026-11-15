"""Static HTML report: figures from data/*.parquet -> reports/report.html (PNGs embedded).

    uv run python -m reco.report
"""
import base64
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reco import bandit

DATA = Path(os.environ.get("RECO_DATA", "data"))
OUT = Path(os.environ.get("RECO_REPORT", "reports"))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def heat(ax, tab, title):
    im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    ax.set_title(title, fontsize=10)
    for (i, j), v in np.ndenumerate(tab.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > tab.values.max() * 0.6 else "black")
    return im


def fig_segments(s):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    s.cluster.value_counts().plot.bar(ax=ax[0], color="#2a6fbb", rot=0, title="Users per cluster")
    feats = ["utilization", "travel_share", "online_share", "insufficient_rate", "cash_advance_share"]
    z = s.groupby("cluster")[feats].mean()
    (z / z.max()).T.plot.bar(ax=ax[1], rot=30, title="Mean feature (scaled to max)")
    return fig


def fig_uplift(s):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for c, g in s.groupby("cluster"):
        ax[0].scatter(g.tau, g.cate, s=6, alpha=0.5, label=c)
    lim = [s.tau.min(), s.tau.max()]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="simulated true effect", ylabel="X-Learner CATE",
              title=f"CATE vs truth (corr {np.corrcoef(s.cate, s.tau)[0, 1]:.2f})")
    ax[0].legend(markerscale=2)
    ax[1].hist(s.cate, bins=40, color="#2a6fbb")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set(title="CATE distribution", xlabel="CATE")
    s.groupby("cluster").cate.mean().plot.bar(ax=ax[2], rot=0, color="#2a6fbb", title="Mean CATE by cluster")
    return fig


def fig_reco(r, s):
    r = r.merge(s[["user_id", "cluster"]], on="user_id")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    r.sector.value_counts().plot.barh(ax=ax[0], color="#2a6fbb", title="Users per recommended sector").invert_yaxis()
    r.groupby("sector").value_month.sum().sort_values().plot.barh(
        ax=ax[1], color="#2f9e6f", title="Total value/month by sector (€)")
    heat(ax[2], pd.crosstab(r.sector, r.cluster, normalize="columns"), "Sector mix within cluster")
    fig.tight_layout()
    return fig


def fig_value(r, s):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(r.upgrade_gain_month, bins=40, color="#2a6fbb")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set(title=f"Upgrade gain/month ({(r.upgrade_gain_month > 0).mean():.0%} > 0)", xlabel="€")
    o = r[r.sector != "none"]
    ax[1].hist(o.p_accept, bins=30, color="#2f9e6f")
    ax[1].set(title="Acceptance probability (offered users)", xlabel="p_accept")
    o.merge(s[["user_id", "cluster"]], on="user_id").groupby("cluster").value_month.mean().plot.bar(
        ax=ax[2], rot=0, color="#2a6fbb", title="Mean offered value/month by cluster (€)")
    return fig


def fig_bandit(s, vals):
    cands = bandit.candidates(vals)
    ns = (500, 1000, 2000, 5000)
    res = pd.DataFrame({n: bandit.simulate(s, cands, n=n) for n in ns}).T
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    res.plot(ax=ax[0], marker="o", title="Mean reward/step vs horizon")
    ax[0].set(xlabel="steps", ylabel="€ / step")
    (res.oracle - res.bandit).plot(ax=ax[1], marker="o", color="#c0392b", title="Regret/step (oracle - bandit)")
    return fig, res


def fig_personas(p, s):
    p = p.merge(s[["user_id", "cluster"]], on="user_id")
    cols = ["digital", "channel", "motivation", "tone"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for a, c in zip(ax.ravel(), cols):
        heat(a, pd.crosstab(p[c], p.cluster, normalize="columns"), f"{c} share within cluster")
    fig.tight_layout()
    return fig


def main():
    s, r = (pd.read_parquet(DATA / f"{n}.parquet") for n in ("user_scores", "user_reco"))
    vals, p = (pd.read_parquet(DATA / f"{n}.parquet") for n in ("user_benefit_value", "personas"))
    bfig, res = fig_bandit(s, vals)
    offered = r[r.sector != "none"]
    kpi = {
        "Users": f"{len(s):,}",
        "Offered a benefit": f"{len(offered) / len(r):.0%}",
        "Total expected value / month": f"€{r.value_month.sum():,.0f}",
        "Mean p(accept), offered": f"{offered.p_accept.mean():.2f}",
        "Upgrade gain > 0": f"{(r.upgrade_gain_month > 0).mean():.0%}",
        "Bandit regret/step @5000": f"{res.oracle.iloc[-1] - res.bandit.iloc[-1]:.2f}",
    }
    sections = [
        ("Segmentation", "Behavioural clusters and what separates them.", fig_segments(s)),
        ("Uplift", "X-Learner estimated treatment effect against the simulated ground truth.", fig_uplift(s)),
        ("Recommendations", "Which benefit sector each user is offered, and the value it carries.", fig_reco(r, s)),
        ("Value and acceptance", "Upgrade gain, acceptance probability and offered value.", fig_value(r, s)),
        ("Bandit", "LinTS vs random, greedy-by-value and oracle.", bfig),
        ("Personas", "Synthetic traits conditioned on behaviour; these drive message tone and channel.", fig_personas(p, s)),
    ]
    tiles = "".join(f"<div class=k><b>{v}</b><span>{k}</span></div>" for k, v in kpi.items())
    body = "".join(f"<h2>{t}</h2><p>{d}</p><img src='data:image/png;base64,{png(f)}'>" for t, d, f in sections)
    OUT.mkdir(exist_ok=True)
    (OUT / "report.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Card upgrade engine report</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
img{{max-width:100%}}.g{{display:flex;flex-wrap:wrap;gap:12px}}
.k{{border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:150px}}
.k b{{display:block;font-size:22px}}.k span{{color:#666;font-size:12px}}</style>
<h1>Card upgrade engine</h1><div class=g>{tiles}</div>{body}""")
    print(f"wrote {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
