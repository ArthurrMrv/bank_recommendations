"""python -m reco ingest|train   (data dir: $RECO_DATA, default ./data)"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

from reco import bandit, features, ingest, persona, segment, uplift, value

DATA = Path(os.environ.get("RECO_DATA", "data"))
WH = DATA / "warehouse"


def train():
    df, embed, kmeans, scores = segment.segment(features.build(WH))
    print("silhouette by k:", scores)
    personas = persona.sample(df)
    personas.to_parquet(DATA / "personas.parquet", index=False)
    for trait in persona.TRAITS:
        print(pd.crosstab(personas[trait], df.cluster.to_numpy(), colnames=["cluster"], normalize="columns").round(2))
    df = uplift.simulate_campaign(df)
    df["cate"] = uplift.x_learner(uplift.design(df), df.t, df.y)
    joblib.dump((embed, kmeans), DATA / "segment.joblib")
    df.to_parquet(DATA / "user_scores.parquet", index=False)
    print(df.groupby("cluster")[["cate", "tau"]].mean())

    vals = value.benefit_value(pd.read_parquet(WH / "user_mcc.parquet"), value.load_benefits())
    vals.to_parquet(DATA / "user_benefit_value.parquet", index=False)
    cands = bandit.candidates(vals)
    print("bandit (mean monthly reward):", bandit.simulate(df, cands))
    reco = bandit.recommend(df, cands).merge(value.upgrade_gain(vals), on="user_id", how="left")
    reco.to_parquet(DATA / "user_reco.parquet", index=False)
    print(reco.sector.value_counts())


STEPS = {"ingest": lambda: ingest.ingest(DATA / "raw", WH), "train": train}

if __name__ == "__main__":
    STEPS[sys.argv[1]]()
