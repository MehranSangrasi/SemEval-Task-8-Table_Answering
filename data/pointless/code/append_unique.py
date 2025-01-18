import pandas as pd
import ast  # For safely evaluating unique values as lists

# Load the first CSV
datasets_csv_path = "data/output_summary.csv"  # Path to the first CSV
datasets_df = pd.read_csv(datasets_csv_path)

# Load the second CSV
unique_values_csv_path = "columns_with_less_than_10_unique_values_test.csv"  # Path to the second CSV
unique_values_df = pd.read_csv(unique_values_csv_path)

# Initialize a dictionary to store unique values for quick lookup
unique_values_dict = {}
for _, row in unique_values_df.iterrows():
    dataset = row["Dataset Name"]
    column = row["Column Name"]
    unique_values = ast.literal_eval(row["Unique Values"])  # Convert string to list
    
    # Store in a nested dictionary: {dataset: {column: unique_values}}
    if dataset not in unique_values_dict:
        unique_values_dict[dataset] = {}
    unique_values_dict[dataset][column] = unique_values

# Function to append unique values to column dtype
def append_unique_values(dataset, columns):
    updated_columns = []
    for column in columns.split(", "):  # Split columns
        if "(" in column:  # Ensure it has a dtype
            col_name, dtype = column.rsplit(" ", 1)  # Split column name and dtype
            dtype = dtype.strip("()")  # Clean up dtype
            # Append unique values if available
            unique_values = unique_values_dict.get(dataset, {}).get(col_name, None)
            if unique_values:
                dtype += f", unique_values: {unique_values}"
            updated_columns.append(f"{col_name} ({dtype})")
        else:
            updated_columns.append(column)  # Keep unchanged if no dtype found
    return ", ".join(updated_columns)

# Update the columns for each dataset
datasets_df["columns"] = datasets_df.apply(
    lambda row: append_unique_values(row["dataset"], row["columns"]), axis=1
)

# Save or display the updated DataFrame
updated_csv_path = "updated_datasets.csv"  # Path to save the updated CSV
datasets_df.to_csv(updated_csv_path, index=False)

# Display the updated DataFrame for verification
print(datasets_df.head())
