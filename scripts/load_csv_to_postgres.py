import sys
import pandas as pd
from sqlalchemy import create_engine
from app.config import settings

def main(csv_path: str):
    df = pd.read_csv(csv_path)
    engine = create_engine(settings.db_url)
    df.to_sql("orders_raw", engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into orders_raw")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/load_csv_to_postgres.py <path_to_csv>")
        raise SystemExit(1)
    main(sys.argv[1])
