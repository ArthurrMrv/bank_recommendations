"""Step 2: deterministic MCC buckets + monthly RFM -> one feature row per user (offline store)."""
from pathlib import Path

import duckdb

FEATURES = [
    "recency_days", "freq_month", "spend_month",
    "travel_share", "healthcare_share", "cash_advance_share", "travel_30d",
    "insufficient_rate", "fico", "dti", "utilization",
]

BUCKET = """CASE
    WHEN mcc BETWEEN 3000 AND 3999 OR mcc IN (4511, 4722, 7011) THEN 'travel'
    WHEN mcc IN (8011, 8021, 8062, 5912) THEN 'healthcare'
    WHEN mcc IN (6010, 6011, 4829) THEN 'cash_advance'
    ELSE 'other' END"""


def build(wh: Path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{wh}/transactions/**/*.parquet')")
    con.execute(f"CREATE VIEW dim_mcc AS SELECT * FROM '{wh}/dim_mcc.parquet'")
    con.execute(f"CREATE VIEW cards AS SELECT * FROM '{wh}/cards.parquet'")
    con.execute(f"CREATE VIEW users AS SELECT * FROM '{wh}/users.parquet'")
    dropped = con.execute("SELECT count(*) FROM tx ANTI JOIN dim_mcc USING (mcc)").fetchall()[0][0]
    print(f"features: {dropped} transactions dropped by dim_mcc join")
    df = con.sql(f"""
    WITH t AS (
        SELECT user_id, ts, greatest(amount, 0) AS spend, insufficient_balance,
               coalesce(card_type = 'Credit', false) AS is_credit, {BUCKET} AS bucket
        FROM tx JOIN dim_mcc USING (mcc) LEFT JOIN cards USING (user_id, card_id)
    ),
    monthly AS (
        SELECT user_id, date_trunc('month', ts) AS month, count(*) AS n, sum(spend) AS spend,
               coalesce(sum(spend) FILTER (is_credit), 0) AS credit_spend
        FROM t GROUP BY ALL
    ),
    rfm AS (
        SELECT user_id, avg(n) AS freq_month, avg(spend) AS spend_month,
               avg(credit_spend) AS credit_spend_month
        FROM monthly GROUP BY user_id
    ),
    totals AS (
        SELECT user_id,
               date_diff('day', max(ts), (SELECT max(ts) FROM t)) AS recency_days,
               sum(spend) FILTER (bucket = 'travel') / sum(spend) AS travel_share,
               sum(spend) FILTER (bucket = 'healthcare') / sum(spend) AS healthcare_share,
               sum(spend) FILTER (bucket = 'cash_advance') / sum(spend) AS cash_advance_share,
               sum(spend) FILTER (bucket = 'travel' AND ts > (SELECT max(ts) FROM t) - INTERVAL 30 DAY) AS travel_30d,
               avg(insufficient_balance::INT) AS insufficient_rate
        FROM t GROUP BY user_id
    ),
    limits AS (
        SELECT user_id, sum(credit_limit) AS credit_limit FROM cards WHERE card_type = 'Credit' GROUP BY user_id
    )
    SELECT user_id, recency_days, freq_month, spend_month, travel_share, healthcare_share,
           cash_advance_share, travel_30d, insufficient_rate, fico,
           debt / nullif(income, 0) AS dti, credit_spend_month / nullif(credit_limit, 0) AS utilization
    FROM totals JOIN rfm USING (user_id) JOIN users USING (user_id) LEFT JOIN limits USING (user_id)
    ORDER BY user_id
    """).df()
    df[FEATURES] = df[FEATURES].astype(float).fillna(0)
    df.to_parquet(wh / "user_features.parquet", index=False)
    return df
