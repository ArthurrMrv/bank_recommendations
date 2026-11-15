"""python -m reco ingest|train   (data dir: $RECO_DATA, default ./data)"""
import os
import sys
from pathlib import Path

import joblib

from reco import features, ingest, segment, uplift

DATA = Path(os.environ.get("RECO_DATA", "data"))
WH = DATA / "warehouse"


def train():
    df, embed, kmeans, scores = segment.segment(features.build(WH))
    print("silhouette by k:", scores)
    df = uplift.simulate_campaign(df)
    df["cate"] = uplift.x_learner(uplift.design(df), df.t, df.y)
    joblib.dump((embed, kmeans), DATA / "segment.joblib")
    df.to_parquet(DATA / "user_scores.parquet", index=False)
    print(df.groupby("cluster")[["cate", "tau"]].mean())


STEPS = {"ingest": lambda: ingest.ingest(DATA / "raw", WH), "train": train}

if __name__ == "__main__":
    STEPS[sys.argv[1]]()
