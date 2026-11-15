"""python -m messaging <user_id>... | --sample N   (needs OPENROUTER_API_KEY; data dir: $RECO_DATA, default ./data)"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from messaging import agent, context

DATA = Path(os.environ.get("RECO_DATA", "data"))


def grid(scores):
    return pd.DataFrame({f"attempt {i + 1}": {k: v["score"] for k, v in g.items()} for i, g in enumerate(scores)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user_ids", nargs="*", type=int)
    ap.add_argument("--sample", type=int, default=0, help="also pick N random users that have an offer")
    args = ap.parse_args()
    ids = list(args.user_ids)
    if args.sample:
        reco = pd.read_parquet(DATA / "user_reco.parquet")
        ids += reco[reco.sector != "none"].user_id.sample(args.sample, random_state=0).tolist()
    writer, judge = agent.openrouter_models()
    rows = []
    for uid in ids:
        facts = context.fact_sheet(uid, DATA)
        if facts is None:
            print(f"\n=== user {uid}: no offer, skipped")
            continue
        out = agent.write_message(facts, writer, judge)
        print(f"\n=== user {uid} ({facts['channel']}, {facts['language']})\n{grid(out['scores']).to_string()}\n")
        print(out["draft"] if out["approved"] else f"FLAGGED after {out['attempts']} attempts, no message")
        rows.append({"user_id": uid, "channel": facts["channel"], "language": facts["language"],
                     "message": out["draft"], "approved": out["approved"], "attempts": out["attempts"],
                     "scores_json": json.dumps(out["scores"], ensure_ascii=False)})
    if rows:
        path = DATA / "messages.parquet"
        new = pd.DataFrame(rows)
        (pd.concat([pd.read_parquet(path), new]) if path.exists() else new).to_parquet(path, index=False)
        print(f"\nwrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
