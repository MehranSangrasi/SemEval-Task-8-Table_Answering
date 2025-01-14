import pandas as pd

# Paths to the input files
dataset_columns_file = "extracted_dataset_columns.csv"  # File with dataset ID and columns
unique_values_file = "unique_values.csv"      # File with dataset names, column names, and unique values

# Read both files into DataFrames
df_columns = pd.read_csv(dataset_columns_file)
df_unique_values = pd.read_csv(unique_values_file)

# Function to add unique values to column names
def integrate_unique_values(columns, dataset_id, unique_values_df):
    column_list = [col.strip() for col in columns.split(',')]
    updated_columns = []
    for col in column_list:
        col_name = col.split(';')[0].strip()  # Extract the column name
        # Check for a match in the unique values file
        match = unique_values_df[
            (unique_values_df['Dataset Name'] == dataset_id) &
            (unique_values_df['Column Name'] == col_name)
        ]
        if not match.empty:
            # Append unique values to the column name
            unique_values = match['Unique Values'].iloc[0]
            updated_col = f"[{col_name}, unique_values: {unique_values}] ; {col.split(';', 1)[1]}"
        else:
            # Keep the original column if no unique values are found
            updated_col = col
        updated_columns.append(updated_col)
    return ', '.join(updated_columns)

# Process each dataset
for index, row in df_columns.iterrows():
    dataset_id = row['dataset_id']
    columns = row['columns']
    # Update the columns with integrated unique values
    df_columns.at[index, 'columns'] = integrate_unique_values(columns, dataset_id, df_unique_values)

# Save the updated DataFrame to a new CSV file
output_file = "updated_columns_with_unique_values.csv"
df_columns.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
