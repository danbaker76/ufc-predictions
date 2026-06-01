from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///ufc.db')

def save_to_db(df, table_name='fight_features'):
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Saved {len(df)} rows to {table_name}")

def load_table(table_name='fight_features'):
    return pd.read_sql(table_name, engine)