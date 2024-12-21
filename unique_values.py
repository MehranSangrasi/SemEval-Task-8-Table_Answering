from datasets import load_dataset
import pandas as pd

# Load the dataset
semeval = load_dataset("cardiffnlp/databench", name="semeval", split="train")

# Convert to Pandas DataFrame
semeval_df = semeval.to_pandas()

# Dictionary to store results
results = []

# Iterate through all 65 datasets (assuming there are 65 datasets stored in columns)
for dataset_name in semeval_df.columns:
    dataset = semeval_df[dataset_name]
    
    # Check if the dataset is valid (e.g., non-null and not empty)
    if isinstance(dataset, pd.Series):
        # Count unique values in the column
        unique_values = dataset.dropna().unique()

        # Check if the column has fewer than 5 unique values
        if len(unique_values) < 5:
            results.append({
                "dataset_name": dataset_name,
                "unique_values": unique_values.tolist()
            })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv("datasets_with_less_than_5_unique_values.csv", index=False)

# Display results
results_df
