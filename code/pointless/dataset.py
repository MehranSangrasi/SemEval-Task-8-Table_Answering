import pandas as pd
from datasets import load_dataset

# Load the dataset
all_qa = load_dataset("cardiffnlp/databench", name="semeval", split="dev")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(all_qa)

# Save the DataFrame to a CSV file
df.to_csv("Table_Answering/data/all_qa_dev.csv", index=False)

# print("data/QA_dev.csv")
