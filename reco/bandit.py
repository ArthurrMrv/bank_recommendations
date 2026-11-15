"""Step 6: linear Thompson Sampling over offers; simulate() replays the simulated campaign offline."""
import numpy as np

ARMS = ("premium", "cashback", "none")
CLUSTERS = ("transactor", "baseline", "revolver")
DIM = len(CLUSTERS) + 3


def context(row):
    return np.array(
        [row["cluster"] == c for c in CLUSTERS]
        + [np.log1p(float(row["travel_30d"])) / 10, float(row["cate"]), 1.0],
        dtype=float,
    )


class LinTS:
    def __init__(self, v=0.3, seed=None):
        self.A = np.stack([np.eye(DIM)] * len(ARMS))
        self.b = np.zeros((len(ARMS), DIM))
        self.v, self.rng = v, np.random.default_rng(seed)

    def choose(self, x):
        draws = [self.rng.multivariate_normal(np.linalg.solve(A, b), self.v**2 * np.linalg.inv(A))
                 for A, b in zip(self.A, self.b)]
        return ARMS[int(np.argmax([x @ theta for theta in draws]))]

    def update(self, arm, x, reward):
        i = ARMS.index(arm)
        self.A[i] += np.outer(x, x)
        self.b[i] += reward * x


def offer(bandit, row):
    x = context(row)
    return ("none" if row["cluster"] == "revolver" else bandit.choose(x)), x


def env_reward(row, arm, rng):
    """Simulated value after a decision: organic + offer-driven premium upgrades, plus cash-back uptake."""
    p_upgrade = row["base"] + (row["tau"] if arm == "premium" else 0)
    p_cashback = (0.3 if row["cluster"] == "baseline" else 0.05) if arm == "cashback" else 0
    return float(rng.random() < p_upgrade) + 0.3 * float(rng.random() < p_cashback)


def simulate(df, n=5000, seed=0):
    """Mean reward of the bandit vs a random (guarded) policy on the simulated environment."""
    rng, bandit = np.random.default_rng(seed), LinTS(seed=seed)
    rows = df.to_dict("records")
    got = {"bandit": [], "random": []}
    for row in (rows[i] for i in rng.integers(len(rows), size=n)):
        arm, x = offer(bandit, row)
        r = env_reward(row, arm, rng)
        bandit.update(arm, x, r)
        got["bandit"].append(r)
        rand = "none" if row["cluster"] == "revolver" else ARMS[rng.integers(3)]
        got["random"].append(env_reward(row, rand, rng))
    return {k: float(np.mean(v)) for k, v in got.items()}
