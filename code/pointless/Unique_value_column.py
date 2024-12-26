import os
import pandas as pd


import pandas as pd



# Path to the directory containing datasets
datasets_dir = "datasets"

# File to keep track of processed datasets
tracking_file = "datasets_tracking.csv"

# Initialize or load the tracking DataFrame
if os.path.exists(tracking_file):
    tracking_df = pd.read_csv(tracking_file)
else:
    tracking_df = pd.DataFrame(columns=["Dataset Name", "Status"])

# Initialize list to store results
results = []

# Iterate through all CSV files in the directory
for file_name in os.listdir(datasets_dir):
    if file_name.endswith(".csv"):
        dataset_path = os.path.join(datasets_dir, file_name)
        dataset_name = os.path.splitext(file_name)[0]  # Get dataset name without extension

        # Check if the dataset is already processed
        if dataset_name in tracking_df["Dataset Name"].values:
            status = tracking_df.loc[tracking_df["Dataset Name"] == dataset_name, "Status"].iloc[0]
            if status == "Complete":
                print(f"Skipping {dataset_name} (already processed).")
                continue

        # Update tracking file with "Processing" status
        if dataset_name not in tracking_df["Dataset Name"].values:
            tracking_df = pd.concat([tracking_df, pd.DataFrame({"Dataset Name": [dataset_name], "Status": ["Processing"]})])
        else:
            tracking_df.loc[tracking_df["Dataset Name"] == dataset_name, "Status"] = "Processing"
        tracking_df.to_csv(tracking_file, index=False)

        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)

            # Check each column for less than 10 unique values
            for column in df.columns:
                unique_values = df[column].dropna().unique()
                # import pdb;pdb.set_trace()  # Remove NaNs and get unique values
                if len(unique_values) < 10:
                    results.append({
                        "Dataset Name": dataset_name,
                        "Column Name": column,
                        "Unique Values": unique_values.tolist()
                    })

            # Mark dataset as complete
            tracking_df.loc[tracking_df["Dataset Name"] == dataset_name, "Status"] = "Complete"
            tracking_df.to_csv(tracking_file, index=False)

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
            # Mark dataset as failed
            tracking_df.loc[tracking_df["Dataset Name"] == dataset_name, "Status"] = f"Failed: {e}"
            tracking_df.to_csv(tracking_file, index=False)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_csv_path = "columns_with_less_than_10_unique_values.csv"
results_df.to_csv(results_csv_path, index=False)

# Display results
print(f"Analysis complete. Results saved to {results_csv_path}.")
print(results_df)
