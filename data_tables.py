import pandas as pd
from datasets import load_dataset


all_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")

# Assuming all_qa is loaded as a DataFrame or Dataset object
# And each dataset ID is stored in the 'dataset' column
dataset_ids = all_qa['dataset']

# Initialize an empty DataFrame to store column names and types
columns_info = pd.DataFrame(columns=['Dataset_ID', 'Column_Name', 'Data_Type'])

# Loop through each dataset ID and extract column names and types
datasets = (list(set(dataset_ids)))


for ds_id in datasets:
#     # Load the full dataset (or sample dataset based on your needs)
#     # import pdb; pdb.set_trace()
    # print(f"hf://datasets/cardiffnlp/databench/data/{ds_id}/all.parquet")
    df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{ds_id}/all.parquet")
    
    # Append the column names and types to the columns_info DataFrame
    temp_df = pd.DataFrame({
        'Dataset_ID': ds_id,
        'Column_Name': df.columns,
        'Data_Type': [df[col].dtype for col in df.columns]
    })
    columns_info = pd.concat([columns_info, temp_df], ignore_index=True)

# Drop duplicate rows to keep only unique column names and types
# unique_columns_info = columns_info.drop_duplicates()

# # Save the unique column names and types to a CSV file
columns_info.to_csv("unique_columns_info.csv", index=False)

print("Unique column names and types saved to unique_columns_info.csv")
