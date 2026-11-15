"""Synthetic contact personas: seeded categorical draws whose odds shift with observed behaviour.

ponytail: invented distributions conditioned on real signals; replace TRAITS with weights fitted on real
channel-engagement / response data once messages are sent.
"""
import numpy as np
import pandas as pd
from scipy.special import softmax

NUMERIC = ("online_share", "age", "freq_month", "travel_share", "healthcare_share")
LANGUAGES = {"en": 0.7, "fr": 0.3}  # probability of being the preferred language
P_SECOND_LANGUAGE = 0.35
CLUSTERS = ("transactor", "baseline", "revolver")

# trait -> category -> {input: logit weight}. Inputs: z-scored NUMERIC, "cluster=<name>", earlier traits as
# "<trait>=<category>", and "1" for a bias. Traits are drawn in this order, so later ones can use earlier ones.
TRAITS = {
    "digital": {
        "low": {"age": 1.0, "online_share": -1.5, "freq_month": -0.5},
        "medium": {},
        "high": {"age": -1.0, "online_share": 1.5, "freq_month": 0.5},
    },
    "channel": {
        "push": {"digital=high": 1.5, "digital=low": -1.5},
        "in_app": {"digital=high": 1.0, "digital=low": -1.0},
        "email": {"1": 0.5},
        "sms": {"digital=low": 1.0},
        "phone": {"digital=low": 1.5, "digital=high": -1.5, "age": 0.7},
    },
    "motivation": {
        "savings": {"cluster=revolver": 1.5, "cluster=baseline": 0.7},
        "status": {"cluster=transactor": 1.2, "travel_share": 0.8},
        "security": {"healthcare_share": 0.8, "age": 0.6},
        "convenience": {"freq_month": 0.6, "online_share": 0.6},
    },
    "tone": {
        "concise_data": {"motivation=savings": 1.5},
        "friendly": {"motivation=convenience": 1.5, "1": 0.3},
        "exclusive": {"motivation=status": 1.5},
        "reassuring": {"motivation=security": 1.5},
    },
}


def sample(df, seed=0):
    """One persona per row of df (needs user_id, cluster and NUMERIC columns) -> user_id + one column per trait."""
    rng, n = np.random.default_rng(seed), len(df)
    x = {"1": np.ones(n)}
    for c in NUMERIC:
        s = df[c].astype(float)
        x[c] = ((s - s.mean()) / (s.std(ddof=0) or 1)).to_numpy()
    for k in CLUSTERS:
        x[f"cluster={k}"] = (df.cluster == k).to_numpy(float)
    out = pd.DataFrame({"user_id": df.user_id.to_numpy()})
    for trait, cats in TRAITS.items():
        logits = np.column_stack([sum((w * x[name] for name, w in ws.items()), np.zeros(n)) for ws in cats.values()])
        cum = softmax(logits, axis=1).cumsum(axis=1)
        pick = np.minimum((rng.random((n, 1)) > cum).sum(axis=1), len(cats) - 1)
        out[trait] = np.array(list(cats))[pick]
        for c in cats:
            x[f"{trait}={c}"] = (out[trait] == c).to_numpy(float)
    langs = list(LANGUAGES)
    out["preferred_language"] = rng.choice(langs, size=n, p=list(LANGUAGES.values()))
    second = rng.random(n) < P_SECOND_LANGUAGE
    out["spoken_languages"] = [[p] + ([l for l in langs if l != p] if s else [])
                               for p, s in zip(out.preferred_language, second)]
    return out
