import pandas as pd
import os

# Define the input directory containing the CSV files and the output file
input_directory = "data/test_set"  # Replace with your directory path
output_file = "data/output_summary.csv"

# Initialize a list to store the results
summary = []

# Iterate through all CSV files in the directory
for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_directory, file_name)
        
        # Read the CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue
        
        # Format the column names and data types
        columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
        
        # Append the results
        summary.append({"dataset": file_name.strip(".csv"), "columns": columns_info})

# Create a DataFrame for the summary
summary_df = pd.DataFrame(summary)

# Save the summary to a new CSV file
summary_df.to_csv(output_file, index=False)

print(f"Summary saved to {output_file}")
