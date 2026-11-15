"""Fact sheet per customer from the pipeline outputs: plain sentences with € amounts, plus their persona."""
import math
from functools import cache
from pathlib import Path

import pandas as pd

from reco import value

LANGUAGE_NAMES = {"en": "English", "fr": "French"}


@cache
def load(data_dir: Path):
    read = lambda name: pd.read_parquet(Path(data_dir) / name)
    return {
        "reco": read("user_reco.parquet").set_index("user_id"),
        "personas": read("personas.parquet").set_index("user_id"),
        "vals": read("user_benefit_value.parquet").set_index(["user_id", "benefit_id"]),
        "user_mcc": read("warehouse/user_mcc.parquet"),
        "benefits": value.load_benefits().set_index("benefit_id"),
    }


def shown(v):
    return round(v, 1) if v < 10 else round(v)


def eur(v):
    return f"€{v:.1f}" if v < 10 else f"€{v:,.0f}"


def describe(benefit_id, b):
    name = benefit_id.split("_", 1)[1]  # drop tier prefix
    sector = "all purchases" if b.sector == "all" else b.sector.replace("_", " ")
    if b.kind == "cashback":
        cap = f", up to {eur(b.cap_month)}/month" if math.isfinite(b.cap_month) else ""
        return f"{b.rate * 100:g}% cash-back on {sector}{cap}"
    if b.kind == "partner_points":
        return f"{b.rate * 100:g}% back in points with {name.split('points_', 1)[1].replace('_', ' ').title()}"
    if b.kind == "subscription":
        return f"{name.replace('_', ' ')} included (worth {eur(b.price_month)}/month)"
    return f"{name.replace('_', ' ')} included"


def fact_sheet(user_id, data_dir="data"):
    """Prompt inputs for one customer, or None when there is nothing to offer (revolvers, sector 'none')."""
    d = load(Path(data_dir))
    r = d["reco"].loc[user_id]
    if r.sector == "none" or pd.isna(r.benefit_id):
        return None
    b = d["benefits"].loc[r.benefit_id]
    p = d["personas"].loc[user_id]
    mcc = d["user_mcc"][d["user_mcc"].user_id == user_id]
    spend = mcc.spend_month.sum() if r.sector == "all" else mcc.spend_month[mcc.sector == r.sector].sum()
    hard = d["vals"].loc[(user_id, r.benefit_id), "hard_value_month"]
    sector = "total card" if r.sector == "all" else r.sector.replace("_", " ")
    facts = [
        f"Recommended benefit: {describe(r.benefit_id, b)} ({r.tier} card)",
        f"Their {sector} spend: {eur(spend)}/month",
        (f"Value to them: {eur(r.value_month)}/month ({eur(shown(r.value_month) * 12)}/year)" if hard > 0 else
         f"Estimated value of the cover/service, which they don't pay for today: {eur(r.value_month)}/month"),
    ]
    if r.tier == "premium":
        facts.append(f"Premium card fee: {eur(value.TIERS['premium'])}/year")
        if r.upgrade_gain_month >= 1:
            facts.append(f"Net gain of the premium card overall (all its benefits together) over the standard card, after the fee: {eur(r.upgrade_gain_month)}/month")
    language = p.preferred_language
    return {
        "user_id": int(user_id),
        "facts": "\n".join(f"- {f}" for f in facts),
        "persona": "\n".join(f"- {k}: {p[k]}" for k in ("tone", "motivation", "digital", "channel")),
        "channel": p.channel,
        "language": language,
        "language_name": LANGUAGE_NAMES[language],
    }
