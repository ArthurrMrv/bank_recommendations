"""Step 5: X-Learner CATE on a simulated randomized upgrade campaign."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor

from reco.features import FEATURES


def simulate_campaign(df, seed=0):
    """Randomized offer with a known true effect `tau` and organic upgrade rate `base`.

    ponytail: simulated because IBM data has no campaigns; swap for real campaign logs once they exist.
    """
    rng = np.random.default_rng(seed)
    std = lambda s: (s - s.mean()) / (s.std() or 1)
    affinity = std(df.travel_share + df.healthcare_share)
    base = expit(-2.5 + 1.5 * std(np.log1p(df.spend_month)))  # sure things: big spenders upgrade anyway
    tau = 0.3 * expit(2 * affinity) * (1 - 2 * base) - 0.15 * (df.cluster == "revolver")  # revolvers: sleeping dogs
    t = rng.random(len(df)) < 0.5
    y = rng.random(len(df)) < np.clip(base + t * tau, 0, 1)
    return df.assign(base=base, tau=tau, t=t.astype(int), y=y.astype(int))


def design(df):
    return pd.get_dummies(df[FEATURES + ["cluster"]], dtype=float)


def x_learner(X, t, y, e=0.5):
    X, t, y = np.asarray(X, float), np.asarray(t, bool), np.asarray(y, float)
    model = lambda: HistGradientBoostingRegressor(max_depth=3, random_state=0)
    mu0, mu1 = model().fit(X[~t], y[~t]), model().fit(X[t], y[t])
    tau1 = model().fit(X[t], y[t] - mu0.predict(X[t]))
    tau0 = model().fit(X[~t], mu1.predict(X[~t]) - y[~t])
    return e * tau0.predict(X) + (1 - e) * tau1.predict(X)
