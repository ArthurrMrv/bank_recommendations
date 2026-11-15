"""Step 6: linear Thompson Sampling over benefit sectors; simulate() replays a simulated acceptance model offline."""
import numpy as np
import pandas as pd
from scipy.special import expit

from reco.features import SECTORS
from reco.value import best_per_sector

ARMS = SECTORS + ("all", "none")
CLUSTERS = ("transactor", "baseline", "revolver")
DIM = len(CLUSTERS) + 5
TOP_K = 3
NONE = {"sector": "none", "benefit_id": None, "tier": None,
        "hard_value_month": 0.0, "soft_signal": 0.0, "nominal_value_month": 0.0}

# ponytail: simulated acceptance stands in for campaign logs; swap env_reward for logged outcomes once they exist.
RISK_APPETITE = {"transactor": 1.0, "baseline": 0.4, "revolver": 0.0}
# acceptance depends only on the offer's value and the user's risk appetite: no invented per-sector preferences.
ALPHA, BETA, GAMMA = -2.0, 1.5, 1.0


def candidates(vals):
    """user_id -> offers: top-K sectors by hard value, the strongest soft-signal sector, and 'none'."""
    best = best_per_sector(vals)
    top = best[best.hard_value_month > 0].sort_values("hard_value_month", ascending=False).groupby("user_id").head(TOP_K)
    soft = best[best.soft_signal > 0].sort_values("soft_signal", ascending=False).groupby("user_id").head(1)
    picked = pd.concat([top, soft]).drop_duplicates(["user_id", "sector"])
    return {u: g.to_dict("records") + [NONE] for u, g in picked.groupby("user_id")}


def norm(c):
    return np.log1p(c["hard_value_month"]) / 4, np.log1p(c["soft_signal"]) / 2


def context(row, c):
    hard, soft = norm(c)
    return np.array([row["cluster"] == k for k in CLUSTERS]
                    + [hard, soft, float(row["cate"]), np.log1p(float(row["travel_30d"])) / 10, 1.0], dtype=float)


def value(c):
    """Monthly value if accepted: hard money plus a nominal amount for soft benefits (cover, new subscription)."""
    return c["hard_value_month"] + c["nominal_value_month"] * (c["soft_signal"] > 0)


def p_accept(row, c):
    if c["sector"] == "none":
        return 0.0
    hard, soft = norm(c)
    k = row["cluster"]
    return float(expit(ALPHA + BETA * hard + GAMMA * soft * RISK_APPETITE[k]))


def env_reward(row, c, rng):
    return value(c) * float(rng.random() < p_accept(row, c))


class LinTS:
    def __init__(self, v=0.3, seed=None):
        self.A = np.stack([np.eye(DIM)] * len(ARMS))
        self.b = np.zeros((len(ARMS), DIM))
        self.v, self.rng = v, np.random.default_rng(seed)

    def score(self, arm, x, greedy=False):
        i = ARMS.index(arm)
        mu = np.linalg.solve(self.A[i], self.b[i])
        theta = mu if greedy else self.rng.multivariate_normal(mu, self.v**2 * np.linalg.inv(self.A[i]))
        return float(x @ theta)

    def update(self, arm, x, reward):
        i = ARMS.index(arm)
        self.A[i] += np.outer(x, x)
        self.b[i] += reward * x


def offer(bandit, row, cands, greedy=False):
    if row["cluster"] == "revolver":
        return NONE, None
    xs = [context(row, c) for c in cands]
    # the bandit learns acceptance only; the offer's value is known, so score = p_hat * value
    i = int(np.argmax([bandit.score(c["sector"], x, greedy) * value(c) for c, x in zip(cands, xs)]))
    return cands[i], xs[i]


def simulate(df, cands, n=5000, seed=0, bandit=None):
    """Mean monthly reward: bandit vs random and greedy-by-value (both guarded) vs oracle (true expected best)."""
    rng, bandit = np.random.default_rng(seed), bandit or LinTS(seed=seed)
    rows = df.to_dict("records")
    got = {k: [] for k in ("bandit", "random", "greedy", "oracle")}
    for row in (rows[i] for i in rng.integers(len(rows), size=n)):
        cs = cands.get(row["user_id"], [NONE])
        c, x = offer(bandit, row, cs)
        accepted = rng.random() < p_accept(row, c)
        r = value(c) * float(accepted)
        if x is not None:
            bandit.update(c["sector"], x, float(accepted))
        got["bandit"].append(r)
        guarded = [NONE] if row["cluster"] == "revolver" else cs
        got["random"].append(env_reward(row, guarded[rng.integers(len(guarded))], rng))
        got["greedy"].append(env_reward(row, max(guarded, key=value), rng))
        got["oracle"].append(max(p_accept(row, c) * value(c) for c in guarded))
    return {k: float(np.mean(v)) for k, v in got.items()}


def recommend(df, cands, n=20000, seed=0):
    """Train on the simulator, then one offer per user from the posterior mean."""
    bandit = LinTS(seed=seed)
    simulate(df, cands, n=n, seed=seed, bandit=bandit)
    out = []
    for row in df.drop_duplicates("user_id").to_dict("records"):
        c, x = offer(bandit, row, cands.get(row["user_id"], [NONE]), greedy=True)
        v = value(c)
        p = float(np.clip(bandit.score(c["sector"], x, greedy=True), 0, 1)) if x is not None else 0.0
        out.append({"user_id": row["user_id"], "sector": c["sector"], "benefit_id": c["benefit_id"],
                    "tier": c["tier"], "value_month": v, "p_accept": p if v else 0.0})
    return pd.DataFrame(out)
