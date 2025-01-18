import pandas as pd
import ast  # Safely evaluate unique values as lists

# Load the first CSV with dataset schemas
datasets_csv_path = "data/output_summary.csv"  # Path to the first CSV
datasets_df = pd.read_csv(datasets_csv_path)

# Load the second CSV with unique values
unique_values_csv_path = "columns_with_less_than_10_unique_values_test.csv"  # Path to the second CSV
unique_values_df = pd.read_csv(unique_values_csv_path)

# Create a dictionary for fast lookup of unique values
unique_values_dict = {}
for _, row in unique_values_df.iterrows():
    dataset = row["Dataset Name"].strip()
    column = row["Column Name"].strip()
    unique_values = ast.literal_eval(row["Unique Values"])  # Convert string to list
    
    # Store unique values in a nested dictionary {dataset: {column: unique_values}}
    if dataset not in unique_values_dict:
        unique_values_dict[dataset] = {}
    unique_values_dict[dataset][column] = unique_values

# Function to append unique values to column data types
def append_unique_values(dataset, columns):
    """
    Append unique values to the data type of each column for a given dataset.

    Args:
        dataset (str): Name of the dataset.
        columns (str): Comma-separated column definitions with their data types.

    Returns:
        str: Updated column definitions with unique values appended to their data types.
    """
    updated_columns = []
    for column in columns.split(", "):  # Split columns by comma
        if "(" in column:  # Ensure the column has a dtype
            col_name, dtype = column.rsplit(" ", 1)  # Split column name and dtype
            dtype = dtype.strip("()")  # Remove parentheses around dtype
            # Append unique values if they exist for this column
            unique_values = unique_values_dict.get(dataset, {}).get(col_name.strip(), None)
            if unique_values:
                dtype += f", unique_values: {unique_values}"
            updated_columns.append(f"{col_name} ({dtype})")
        else:
            updated_columns.append(column)  # Keep unchanged if no dtype found
    return ", ".join(updated_columns)

# Update columns for all datasets
datasets_df["columns"] = datasets_df.apply(
    lambda row: append_unique_values(row["dataset"].strip(), row["columns"]), axis=1
)

# Check for unmatched columns and datasets
unmatched = []
for dataset, columns_dict in unique_values_dict.items():
    if dataset not in datasets_df["dataset"].values:
        unmatched.append((dataset, list(columns_dict.keys())))

# Save the updated DataFrame to a new CSV
updated_csv_path = "updated_datasets_with_unique_values.csv"  # Path to save the updated CSV
datasets_df.to_csv(updated_csv_path, index=False)

# Display unmatched datasets and columns for debugging
if unmatched:
    print("Unmatched datasets or columns:")
    for dataset, columns in unmatched:
        print(f"Dataset: {dataset}, Columns: {columns}")

# Display the last few rows of the updated DataFrame for verification
print(datasets_df.tail())  # Display the last few rows to verify changes
