import pandas as pd

dataset = "051_Pokemon"
df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset}/all.parquet")


query = df['name'].eq('Pikachu').any()


print(query)