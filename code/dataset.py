import pandas as pd
from datasets import load_dataset

# Load the dataset
all_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(all_qa)

# Save the DataFrame to a CSV file
df.to_csv("all_qa.csv", index=False)

print("QA.csv")
