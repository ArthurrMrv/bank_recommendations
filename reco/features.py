"""Step 2: deterministic MCC buckets + monthly RFM -> one feature row per user (offline store)."""
from pathlib import Path

import duckdb

FEATURES = [
    "recency_days", "freq_month", "spend_month",
    "travel_share", "healthcare_share", "cash_advance_share", "travel_30d",
    "insufficient_rate", "fico", "dti", "utilization",
]
PERSONA_INPUTS = ["online_share", "age"]  # written to user_features, kept out of segmentation

BUCKET = """CASE
    WHEN mcc BETWEEN 3000 AND 3999 OR mcc IN (4511, 4722, 7011) THEN 'travel'
    WHEN mcc IN (8011, 8021, 8062, 5912) THEN 'healthcare'
    WHEN mcc IN (6010, 6011, 4829) THEN 'cash_advance'
    ELSE 'other' END"""

SECTORS = (
    "air", "hotel", "car_rental", "transit", "groceries", "dining", "fuel", "retail_online",
    "entertainment", "telecom_utilities", "insurance", "healthcare", "cash_advance", "other",
)
SECTOR = """CASE
    WHEN mcc BETWEEN 3000 AND 3350 OR mcc = 4511 THEN 'air'
    WHEN mcc BETWEEN 3351 AND 3500 OR mcc IN (7512, 7513, 7519) THEN 'car_rental'
    WHEN mcc BETWEEN 3501 AND 3999 OR mcc = 7011 THEN 'hotel'
    WHEN mcc IN (4111, 4112, 4121, 4131, 4722, 4784, 4789) THEN 'transit'
    WHEN mcc IN (5300, 5411, 5422, 5441, 5451, 5462, 5499) THEN 'groceries'
    WHEN mcc BETWEEN 5811 AND 5814 THEN 'dining'
    WHEN mcc IN (5541, 5542, 5983) THEN 'fuel'
    WHEN mcc IN (6300, 5960, 6381, 6399) THEN 'insurance'
    WHEN mcc BETWEEN 5815 AND 5818 OR mcc IN (7832, 7922, 7929, 7991, 7995, 7996, 7997, 7999) THEN 'entertainment'
    WHEN mcc IN (4812, 4814, 4816, 4899, 4900) THEN 'telecom_utilities'
    WHEN mcc IN (8011, 8021, 8062, 5912) OR mcc BETWEEN 8031 AND 8099 THEN 'healthcare'
    WHEN mcc IN (6010, 6011, 4829) THEN 'cash_advance'
    WHEN mcc BETWEEN 5200 AND 5999 THEN 'retail_online'
    ELSE 'other' END"""
WATCH_MCCS = (6300, 5960, 4899, 4814, 5815, 5816, 5817, 5818, 7997)  # insurance + subscription payments


def build(wh: Path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{wh}/transactions/**/*.parquet')")
    con.execute(f"CREATE VIEW dim_mcc AS SELECT * FROM '{wh}/dim_mcc.parquet'")
    con.execute(f"CREATE VIEW cards AS SELECT * FROM '{wh}/cards.parquet'")
    con.execute(f"CREATE VIEW users AS SELECT * FROM '{wh}/users.parquet'")
    dropped = con.execute("SELECT count(*) FROM tx ANTI JOIN dim_mcc USING (mcc)").fetchall()[0][0]
    print(f"features: {dropped} transactions dropped by dim_mcc join")
    con.execute(f"""CREATE VIEW t AS
        SELECT user_id, ts, mcc, greatest(amount, 0) AS spend, insufficient_balance,
               coalesce(card_type = 'Credit', false) AS is_credit, coalesce(online, false) AS online,
               {BUCKET} AS bucket, {SECTOR} AS sector
        FROM tx JOIN dim_mcc USING (mcc) LEFT JOIN cards USING (user_id, card_id)""")
    seen = con.execute(f"SELECT mcc, count(*) FROM t WHERE mcc IN {WATCH_MCCS} GROUP BY mcc ORDER BY mcc").fetchall()
    print(f"features: insurance/subscription MCCs present (mcc, n): {seen or 'none'}")
    con.execute(f"""COPY (
        WITH months AS (SELECT user_id, count(DISTINCT date_trunc('month', ts)) AS months FROM t GROUP BY user_id)
        SELECT user_id, mcc, any_value(sector) AS sector,
               sum(spend) / months AS spend_month, count(*) / months AS n_month,
               coalesce(sum(spend) FILTER (ts > (SELECT max(ts) FROM t) - INTERVAL 3 MONTH), 0) / 3
                   AS spend_recent_month
        FROM t JOIN months USING (user_id) GROUP BY user_id, mcc, months ORDER BY user_id, mcc
    ) TO '{wh}/user_mcc.parquet'""")
    df = con.sql("""
    WITH monthly AS (
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
               avg(insufficient_balance::INT) AS insufficient_rate, avg(online::INT) AS online_share
        FROM t GROUP BY user_id
    ),
    limits AS (
        SELECT user_id, sum(credit_limit) AS credit_limit FROM cards WHERE card_type = 'Credit' GROUP BY user_id
    )
    SELECT user_id, recency_days, freq_month, spend_month, travel_share, healthcare_share,
           cash_advance_share, travel_30d, insufficient_rate, fico,
           debt / nullif(income, 0) AS dti, credit_spend_month / nullif(credit_limit, 0) AS utilization,
           online_share, age
    FROM totals JOIN rfm USING (user_id) JOIN users USING (user_id) LEFT JOIN limits USING (user_id)
    ORDER BY user_id
    """).df()
    df[FEATURES + PERSONA_INPUTS] = df[FEATURES + PERSONA_INPUTS].astype(float).fillna(0)
    df.to_parquet(wh / "user_features.parquet", index=False)
    return df
