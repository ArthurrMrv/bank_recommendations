"""Steps 3-4: scaled features -> PCA(4) -> K-Means, k by silhouette, clusters named by centroid rules."""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reco.features import FEATURES


def segment(df, ks=range(3, 7), seed=0):
    embed = make_pipeline(StandardScaler(), PCA(n_components=4, random_state=seed))
    z = embed.fit_transform(df[FEATURES])
    fits = {k: KMeans(k, n_init=10, random_state=seed).fit(z) for k in ks}
    scores = {k: silhouette_score(z, m.labels_) for k, m in fits.items()}
    best = fits[max(scores, key=scores.get)]
    names = name_clusters(df, best.labels_)
    return df.assign(cluster=[names[c] for c in best.labels_]), embed, best, scores


def name_clusters(df, labels):
    f = df[FEATURES]
    z = (f - f.mean()) / f.std(ddof=0).replace(0, 1)
    risk = (z.utilization + z.cash_advance_share + z.dti + z.insufficient_rate - z.fico).groupby(labels).mean()
    premium = ((f.travel_share + f.healthcare_share) * f.spend_month).groupby(labels).mean()
    revolver = risk.idxmax()
    transactor = premium.drop(revolver).idxmax()
    return {c: "revolver" if c == revolver else "transactor" if c == transactor else "baseline" for c in risk.index}
