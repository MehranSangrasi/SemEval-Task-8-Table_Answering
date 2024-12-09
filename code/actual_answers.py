import pandas as pd


df1 = pd.read_csv("data/queries_with_answers_final_4.csv")
df2 = pd.read_csv("data/queries_new_answers.csv")

df2["actual_answer"] = df1["actual_answer"]

df2.to_csv("data/queries_new_answers.csv", index=False)

