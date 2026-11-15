"""Step 1: IBM TabFormer CSVs + ISO 18245 MCC list -> Parquet warehouse."""
import urllib.request
from pathlib import Path

import duckdb

MCC_URL = "https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv"


def money(col):
    return f"""CAST(replace("{col}", '$', '') AS DOUBLE)"""


def ingest(raw: Path, wh: Path):
    wh.mkdir(parents=True, exist_ok=True)
    if not (raw / "mcc_codes.csv").exists():
        urllib.request.urlretrieve(MCC_URL, raw / "mcc_codes.csv")
    con = duckdb.connect()
    # ponytail: partitioned by year and sorted by user_id, not partitioned by user_id (2,000+ tiny dirs)
    con.execute(f"""COPY (
        SELECT "User" AS user_id, "Card" AS card_id, "Year" AS year,
               make_date("Year", "Month", "Day") + CAST("Time" AS TIME) AS ts,
               {money("Amount")} AS amount,
               CAST("MCC" AS INT) AS mcc,
               coalesce("Errors?", '') LIKE '%Insufficient Balance%' AS insufficient_balance
        FROM read_csv('{raw}/card_transaction.v1.csv',
                      types={{'Amount': 'VARCHAR', 'Errors?': 'VARCHAR', 'Is Fraud?': 'VARCHAR'}})
        WHERE "Is Fraud?" = 'No'
        ORDER BY user_id, ts
    ) TO '{wh}/transactions' (FORMAT parquet, PARTITION_BY (year), OVERWRITE)""")
    con.execute(f"""COPY (
        SELECT row_number() OVER () - 1 AS user_id, "FICO Score" AS fico,
               {money("Yearly Income - Person")} AS income, {money("Total Debt")} AS debt
        FROM read_csv('{raw}/sd254_users.csv')
    ) TO '{wh}/users.parquet'""")
    con.execute(f"""COPY (
        SELECT "User" AS user_id, "CARD INDEX" AS card_id, "Card Type" AS card_type,
               {money("Credit Limit")} AS credit_limit
        FROM read_csv('{raw}/sd254_cards.csv')
    ) TO '{wh}/cards.parquet'""")
    con.execute(f"""COPY (
        SELECT CAST(mcc AS INT) AS mcc, edited_description AS description
        FROM read_csv('{raw}/mcc_codes.csv', types={{'mcc': 'VARCHAR'}})
    ) TO '{wh}/dim_mcc.parquet'""")
