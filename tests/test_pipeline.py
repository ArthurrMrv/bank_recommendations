"""End-to-end on a tiny IBM-schema sample with planted segments (no network)."""
import numpy as np
import pandas as pd
import pytest

from reco import bandit, features, ingest, persona, segment, uplift, value

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
            users.append({"Person": f"p{u}", "Current Age": int(rng.integers(20, 80)), "Yearly Income - Person": "$60000",
                          "Total Debt": f"${debt * rng.uniform(0.8, 1.2):.0f}",
                          "FICO Score": fico + int(rng.integers(-20, 20))})
            cards.append({"User": u, "CARD INDEX": 0, "Card Type": "Credit",
                          "Credit Limit": f"${limit * rng.uniform(0.8, 1.2):.0f}"})
            for _ in range(200):
                tx.append({"User": u, "Card": 0, "Year": int(rng.integers(2018, 2020)),
                           "Month": int(rng.integers(1, 13)), "Day": int(rng.integers(1, 29)),
                           "Time": f"{rng.integers(24):02d}:{rng.integers(60):02d}",
                           "Amount": f"${rng.exponential(amt):.2f}",
                           "Use Chip": "Online Transaction" if s == 0 and rng.random() < 0.5 else "Chip Transaction",
                           "Merchant Name": int(rng.integers(1e9)), "Merchant City": "X", "Merchant State": "CA",
                           "Zip": 90000.0, "MCC": int(rng.choice(mccs)),
                           "Errors?": "Insufficient Balance," if rng.random() < ins else "",
                           "Is Fraud?": "No"})
    tx += [dict(tx[0], MCC=9999), dict(tx[0], **{"Is Fraud?": "Yes"})]  # unknown MCC + fraud row: both dropped
    pd.DataFrame(tx).to_csv(raw / "credit_card_transactions-ibm_v2.csv", index=False)
    pd.DataFrame(users).to_csv(raw / "sd254_users.csv", index=False)
    pd.DataFrame(cards).to_csv(raw / "sd254_cards.csv", index=False)
    mccs = sorted({m for v in PLANTED.values() for m in v[0]})
    pd.DataFrame({"mcc": [f"{m:04d}" for m in mccs], "edited_description": "x"}).to_csv(raw / "mcc_codes.csv", index=False)


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    raw, wh = tmp_path_factory.mktemp("raw"), tmp_path_factory.mktemp("wh")
    make_raw(raw)
    ingest.ingest(raw, wh)
    return wh


@pytest.fixture(scope="module")
def scored(wh):
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


def test_personas_conditioned_and_seeded(scored):
    assert scored[scored.planted == "transactor"].online_share.between(0.3, 0.7).all()
    n = 2000
    df = pd.DataFrame({"user_id": range(2 * n), "cluster": "baseline", "freq_month": 50.0,
                       "travel_share": 0.1, "healthcare_share": 0.05,
                       "online_share": [1.0] * n + [0.0] * n, "age": [25] * n + [75] * n})
    p = persona.sample(df)
    young, old = p.iloc[:n], p.iloc[n:]
    assert (young.digital == "high").mean() > 0.5 > (old.digital == "high").mean() + 0.4
    assert (old.channel.isin(["sms", "phone"])).mean() > (young.channel.isin(["sms", "phone"])).mean()
    pd.testing.assert_frame_equal(p, persona.sample(df))
    assert all(pref in spoken for pref, spoken in zip(p.preferred_language, p.spoken_languages))
    assert 0.6 < (p.preferred_language == "en").mean() < 0.8 and (p.spoken_languages.map(len) == 2).any()


def test_benefit_value_kinds():
    user_mcc = pd.DataFrame({
        "user_id": [0, 0, 0, 1, 1],
        "mcc": [5812, 3005, 6300, 5812, 3001],
        "sector": ["dining", "air", "insurance", "dining", "air"],
        "spend_month": [2000.0, 300.0, 40.0, 100.0, 500.0],
        "n_month": [10.0, 2.0, 1.0, 5.0, 3.0],
    })
    cat = value.load_benefits()
    cat = cat[cat.benefit_id.isin(["prem_cashback_dining", "prem_points_british_airways",
                                   "prem_streaming", "prem_travel_insurance"])]
    v = value.benefit_value(user_mcc, cat).set_index(["user_id", "benefit_id"])
    hard, soft = v.hard_value_month, v.soft_signal
    assert hard[0, "prem_cashback_dining"] == 30 and hard[1, "prem_cashback_dining"] == 3  # 3%, capped at 30
    assert hard[0, "prem_points_british_airways"] == 9 and hard[1, "prem_points_british_airways"] == 0  # 3005 only
    assert (hard.xs("prem_streaming", level=1) == 0).all()  # nobody pays at 4899/5815
    assert hard[0, "prem_travel_insurance"] == 25 and soft[0, "prem_travel_insurance"] == 0  # pays 40, cover capped
    assert hard[1, "prem_travel_insurance"] == 0 and soft[1, "prem_travel_insurance"] == 3  # 3 flights, no policy

    # cash-back doesn't stack: dining 3% covers 1000 of the 2000 (cap 30), 1% on everything gets the other 1340
    cb = value.load_benefits().query("benefit_id in ['prem_cashback_all', 'prem_cashback_dining', 'std_cashback_all']")
    gain = value.upgrade_gain(value.benefit_value(user_mcc, cb)).set_index("user_id").upgrade_gain_month
    assert gain[0] == pytest.approx(30 + 13.4 - 10 - 20)


@pytest.fixture(scope="module")
def cands(wh):
    return bandit.candidates(value.benefit_value(pd.read_parquet(wh / "user_mcc.parquet"), value.load_benefits()))


def test_bandit_beats_random_and_guards_revolvers(campaign, cands):
    res = bandit.simulate(campaign, cands, n=3000)
    assert res["random"] < res["bandit"] and res["bandit"] > 0.8 * res["oracle"], res
    rev = campaign[campaign.cluster == "revolver"].iloc[0]
    assert bandit.offer(bandit.LinTS(), rev, cands[rev.user_id])[0]["sector"] == "none"
    reco = bandit.recommend(campaign, cands, n=3000)
    assert len(reco) == 3 * N and (reco[reco.user_id.isin(campaign[campaign.cluster == "revolver"].user_id)]
                                   .sector == "none").all()
