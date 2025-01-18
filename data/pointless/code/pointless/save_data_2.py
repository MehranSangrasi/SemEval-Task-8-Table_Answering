from datasets import load_dataset
import os
import pandas as pd

# # Directory to store downloaded datasets
local_directory = "data/test_set"
os.makedirs(local_directory, exist_ok=True)  # Ensure the directory exists

# List of dataset IDs
table_info = "competition\\competition"

# dataset_ids = []


for item in os.listdir(table_info):
    
    dataset_id = item
    
    parquet_path = os.path.join(table_info, dataset_id, "sample.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        csv_path = os.path.join("data", "test_set", f"{dataset_id}_sample.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Saved dataset {dataset_id} to {csv_path}")
        
    except Exception as e:
        print(f"Failed to process dataset {dataset_id}: {e}")

# for index, row in table_info.iterrows():
#     dataset_id = row['Dataset_ID']
#     dataset_ids.append(dataset_id)

# # Download and save datasets locally as CSV
# for dataset_id in dataset_ids:
#     parquet_path = f"hf://datasets/cardiffnlp/databench/data/{dataset_id}/all.parquet"
#     print(f"Processing dataset: {dataset_id}")
    
#     try:
#         # Load the parquet file using pandas
#         df = pd.read_parquet(parquet_path, engine='pyarrow')  # Ensure pyarrow is installed
        
#         # Save the dataset as a CSV file
#         csv_path = os.path.join(local_directory, f"{dataset_id}.csv")
#         df.to_csv(csv_path, index=False)
#         print(f"Saved dataset {dataset_id} to {csv_path}")
#     except Exception as e:
#         print(f"Failed to process dataset {dataset_id}: {e}")
