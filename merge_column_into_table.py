import pandas as pd

# Input and output file paths
input_file = "unique_columns_info.csv"  # Replace with your input file path
output_file = "only_names.csv"  # Replace with your desired output file path

# Read the input CSV
df = pd.read_csv(input_file)

# Combine Column_Name and Data_Type into the desired format
df["Combined"] = df["Column_Name"] # + "; " + df["Data_Type"]

# Group by Dataset_ID and join the combined entries with ', '
result = df.groupby("Dataset_ID")["Combined"].apply(lambda x: ", ".join(x)).reset_index()

# Rename columns for clarity
result.columns = ["Dataset_ID", "Columns"]

# Save the result to a new CSV
result.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
