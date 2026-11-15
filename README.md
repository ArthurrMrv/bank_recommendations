# Card upgrade recommendation engine

IBM synthetic card transactions -> MCC buckets + RFM (DuckDB) -> PCA + K-Means segments ->
X-Learner CATE (simulated campaign) -> linear Thompson Sampling offers, evaluated offline.

| Step | Module |
|------|--------|
| 1. Ingest to Parquet warehouse | `reco/ingest.py` |
| 2. MCC join, buckets, monthly RFM | `reco/features.py` |
| 3-4. PCA + K-Means (k by silhouette) | `reco/segment.py` |
| 5. Simulated campaign + X-Learner | `reco/uplift.py` |
| 4b. Synthetic contact personas (channel, tone, motivation, digital, languages) | `reco/persona.py` |
| 5b. Benefit value per user from `reco/benefits.csv` | `reco/value.py` |
| 6. Contextual bandit over benefit sectors (offline simulation) | `reco/bandit.py` |
| 7. Tailored message by "Claire" (LangGraph: write, score, retry x2, else flag) | `messaging/` (prompts: `prompt.md`, `judge.md`) |

## Data

Download the Kaggle dataset `ealtman2019/credit-card-transactions` into `data/raw/`
(`credit_card_transactions-ibm_v2.csv`, `sd254_users.csv`, `sd254_cards.csv`):

```bash
uvx kaggle datasets download ealtman2019/credit-card-transactions -p data/raw --unzip
```

The ISO 18245 MCC list (`greggles/mcc-codes`) is fetched by `ingest` automatically.

## Run (CPU RunPod, 16 GB+ RAM)

```bash
uv sync
uv run python -m reco ingest
uv run python -m reco train      # writes data/user_scores.parquet, personas.parquet, user_benefit_value.parquet, user_reco.parquet
uv run pytest
```

Messages (local is fine, needs the `train` outputs and `OPENROUTER_API_KEY`, see `.env.example`):

```bash
source .env
uv run python -m messaging 12 57      # or: --sample 5; appends to data/messages.parquet
```

Data dir is `RECO_DATA` (default `./data`).
Results: `notebooks/report.ipynb`.
