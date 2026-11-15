"""End-to-end on a tiny IBM-schema sample with planted segments (no network)."""
import numpy as np
import pandas as pd
import pytest

from reco import bandit, features, ingest, segment, uplift

# segment -> (MCC choices, mean amount, FICO, debt, credit limit, insufficient-balance rate)
PLANTED = {
    "transactor": ([3001, 3504, 7011, 8011, 5912, 5411], 250, 790, 10_000, 25_000, 0.0),
    "baseline": ([5411, 5300, 5812], 40, 710, 20_000, 8_000, 0.0),
    "revolver": ([6011, 4829, 5411, 5300], 80, 570, 90_000, 1_500, 0.2),
}
N = 20  # users per segment


def make_raw(raw, seed=0):
    rng = np.random.default_rng(seed)
    tx, users, cards = [], [], []
    for s, (mccs, amt, fico, debt, limit, ins) in enumerate(PLANTED.values()):
        for u in range(s * N, (s + 1) * N):
            users.append({"Person": f"p{u}", "Yearly Income - Person": "$60000",
                          "Total Debt": f"${debt * rng.uniform(0.8, 1.2):.0f}",
                          "FICO Score": fico + int(rng.integers(-20, 20))})
            cards.append({"User": u, "CARD INDEX": 0, "Card Type": "Credit",
                          "Credit Limit": f"${limit * rng.uniform(0.8, 1.2):.0f}"})
            for _ in range(200):
                tx.append({"User": u, "Card": 0, "Year": int(rng.integers(2018, 2020)),
                           "Month": int(rng.integers(1, 13)), "Day": int(rng.integers(1, 29)),
                           "Time": f"{rng.integers(24):02d}:{rng.integers(60):02d}",
                           "Amount": f"${rng.exponential(amt):.2f}", "Use Chip": "Chip Transaction",
                           "Merchant Name": int(rng.integers(1e9)), "Merchant City": "X", "Merchant State": "CA",
                           "Zip": 90000.0, "MCC": int(rng.choice(mccs)),
                           "Errors?": "Insufficient Balance," if rng.random() < ins else "",
                           "Is Fraud?": "No"})
    tx += [dict(tx[0], MCC=9999), dict(tx[0], **{"Is Fraud?": "Yes"})]  # unknown MCC + fraud row: both dropped
    pd.DataFrame(tx).to_csv(raw / "card_transaction.v1.csv", index=False)
    pd.DataFrame(users).to_csv(raw / "sd254_users.csv", index=False)
    pd.DataFrame(cards).to_csv(raw / "sd254_cards.csv", index=False)
    mccs = sorted({m for v in PLANTED.values() for m in v[0]})
    pd.DataFrame({"mcc": [f"{m:04d}" for m in mccs], "edited_description": "x"}).to_csv(raw / "mcc_codes.csv", index=False)


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    raw, wh = tmp_path_factory.mktemp("raw"), tmp_path_factory.mktemp("wh")
    make_raw(raw)
    ingest.ingest(raw, wh)
    df, *_ = segment.segment(features.build(wh))
    df["planted"] = np.repeat(list(PLANTED), N)
    return df


def test_features_and_segments(scored):
    assert len(scored) == 3 * N and not scored[features.FEATURES].isna().any().any()
    t = scored[scored.planted == "transactor"]
    assert (t.travel_share + t.healthcare_share).min() > 0.5
    assert (scored[scored.planted == "revolver"].cash_advance_share > 0.3).all()
    assert (scored.planted == scored.cluster).mean() > 0.9


@pytest.fixture(scope="module")
def campaign(scored):
    big = pd.concat([scored] * 100, ignore_index=True)  # repeat users so Bernoulli outcomes average out
    big = uplift.simulate_campaign(big)
    big["cate"] = uplift.x_learner(uplift.design(big), big.t, big.y)
    return big


def test_x_learner_recovers_tau(campaign):
    assert np.corrcoef(campaign.cate, campaign.tau)[0, 1] > 0.5


def test_bandit_beats_random_and_guards_revolvers(campaign):
    res = bandit.simulate(campaign, n=3000)
    assert res["bandit"] > res["random"], res
    rev = campaign[campaign.cluster == "revolver"].iloc[0]
    assert bandit.offer(bandit.LinTS(), rev)[0] == "none"
