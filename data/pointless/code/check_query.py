import pandas as pd


df = pd.read_csv("data/datasets/018_Staff.csv")

query = pd.to_datetime(df['Date Hired']).dt.year.value_counts().idxmax()

print(query)