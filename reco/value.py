"""Step 5b: monthly value of every catalog benefit for every user, from their own spend per MCC."""
from pathlib import Path

import pandas as pd

CATALOG = Path(__file__).with_name("benefits.csv")
TIERS = {"standard": 0.0, "premium": 240.0}  # annual fee
SOFT_KINDS = ("subscription", "insurance")


def load_benefits(path=CATALOG):
    b = pd.read_csv(path, dtype={"mccs": str})
    return b.fillna({"mccs": "", "rate": 0.0, "cap_month": float("inf"), "value_per_use": 0.0, "price_month": 0.0})


def benefit_value(user_mcc, benefits):
    """Long frame user x benefit: hard_value_month (money we can see) and soft_signal (exposure, no money).

    Hard value: cash-back, partner points, and subscriptions/insurance the user already pays for (the card replaces
    them). Soft signal: monthly transaction count in the benefit's sector when nothing is paid yet.
    tier_value_month: the benefit's share of its card's total. Cash-back doesn't stack: spend earns its sector
    rate up to the cap, and only the rest earns the card's catalog-wide rate.
    """
    spend = user_mcc.pivot_table(index="user_id", columns="mcc", values="spend_month", aggfunc="sum", fill_value=0)
    by_sector = user_mcc.pivot_table(index="user_id", columns="sector", values=["spend_month", "n_month"],
                                     aggfunc="sum", fill_value=0)
    zero = pd.Series(0.0, index=spend.index)
    sector = lambda col, s: by_sector[col][s] if s in by_sector[col] else zero
    total = spend.sum(axis=1)
    covered = {t: zero for t in benefits.tier.unique()}  # spend already earning a sector cash-back, per tier
    for b in benefits[(benefits.kind == "cashback") & (benefits.sector != "all")].itertuples():
        covered[b.tier] = covered[b.tier] + sector("spend_month", b.sector).clip(upper=b.cap_month / b.rate)
    rows = []
    for b in benefits.itertuples():
        at_mccs = spend.reindex(columns=[int(m) for m in b.mccs.split()], fill_value=0).sum(axis=1)
        tier_value = None
        if b.kind == "cashback":
            scope = total if b.sector == "all" else sector("spend_month", b.sector)
            hard = (b.rate * scope).clip(upper=b.cap_month)
            if b.sector == "all":
                tier_value = (b.rate * (total - covered[b.tier])).clip(upper=b.cap_month)
        elif b.kind == "partner_points":
            hard = b.rate * at_mccs
        elif b.kind == "subscription":
            hard = at_mccs.clip(upper=b.price_month)
        elif b.kind == "insurance":
            hard = at_mccs.clip(upper=b.cap_month)
        else:
            raise ValueError(f"unknown benefit kind {b.kind!r}")
        soft = sector("n_month", b.sector).where(hard == 0, 0.0) if b.kind in SOFT_KINDS else zero
        rows.append(pd.DataFrame({
            "user_id": spend.index, "benefit_id": b.benefit_id, "tier": b.tier, "kind": b.kind, "sector": b.sector,
            "hard_value_month": hard.to_numpy(float),
            "tier_value_month": (hard if tier_value is None else tier_value).to_numpy(float),
            "soft_signal": soft.to_numpy(float),
            "nominal_value_month": b.value_per_use,
        }))
    return pd.concat(rows, ignore_index=True)


def best_per_sector(vals):
    """One row per (user, sector): the benefit with the highest hard value, soft signal as tie-break."""
    return (vals.sort_values(["hard_value_month", "soft_signal"])
            .groupby(["user_id", "sector"]).tail(1).reset_index(drop=True))


def upgrade_gain(vals):
    """Monthly value of the premium card over standard (non-stacking cash-back), net of the premium fee."""
    tier = vals.pivot_table(index="user_id", columns="tier", values="tier_value_month", aggfunc="sum", fill_value=0)
    gain = tier.get("premium", 0) - tier.get("standard", 0) - TIERS["premium"] / 12
    return gain.rename("upgrade_gain_month").reset_index()
