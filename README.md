# Card upgrade recommendation engine

IBM synthetic card transactions -> MCC buckets + RFM (DuckDB) -> PCA + K-Means segments ->
X-Learner CATE (simulated campaign) -> linear Thompson Sampling offers, evaluated offline.

| Step | Module |
|------|--------|
| 1. Ingest to Parquet warehouse | `reco/ingest.py` |
| 2. MCC join, buckets, monthly RFM | `reco/features.py` |
| 3-4. PCA + K-Means (k by silhouette) | `reco/segment.py` |
| 5. Simulated campaign + X-Learner | `reco/uplift.py` |
| 6. Contextual bandit (offline simulation) | `reco/bandit.py` |

## Data

Download the Kaggle dataset `ealtman2019/credit-card-transactions` into `data/raw/`
(`card_transaction.v1.csv`, `sd254_users.csv`, `sd254_cards.csv`):

```bash
uvx kaggle datasets download ealtman2019/credit-card-transactions -p data/raw --unzip
```

The ISO 18245 MCC list (`greggles/mcc-codes`) is fetched by `ingest` automatically.

## Run (CPU RunPod, 16 GB+ RAM)

```bash
uv sync
uv run python -m reco ingest
uv run python -m reco train      # writes data/user_scores.parquet
uv run pytest
```

Data dir is `RECO_DATA` (default `./data`).
Results: `notebooks/report.ipynb`.
