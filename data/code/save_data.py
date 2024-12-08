import os
import requests
import pandas as pd

# Directory to store downloaded parquet files
local_directory = "data/datasets"

# Create the directory if it doesn't exist
# os.makedirs(local_directory, exist_ok=True)

# Function to download a parquet file
def download_parquet(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {url}, Status code: {response.status_code}")

# Example parquet links from dataset

table_info = pd.read_csv("data/table_info.csv")

parquet_links = [
]

for index, row in table_info.iterrows():
    dataset_id = row['Dataset_ID']
    
    parquet_link = "hf://datasets/cardiffnlp/databench/data/"+dataset_id+"/all.parquet"
    
    parquet_links.append(parquet_link)
    
    
    



# Download each parquet file
for link in parquet_links:
    filename = link.split("/")[-1]
    save_path = os.path.join(local_directory, filename)
    download_parquet(link, save_path)
