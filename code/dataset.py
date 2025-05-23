import pandas as pd

# Paths to the input files
dataset_columns_file = "table_info.csv"  # File with dataset ID and columns
unique_values_file = "unique_values.csv"      # File with dataset names, column names, and unique values

# Read both files into DataFrames
df_columns = pd.read_csv(dataset_columns_file)
df_unique_values = pd.read_csv(unique_values_file)

# Function to add unique values to column names
def integrate_unique_values(columns, dataset_id, unique_values_df):
    column_list = [col.strip() for col in columns.split(',')]
    updated_columns = []
    
    for col in column_list:
        col_parts = col.split(';')
        
        # Ensure there are at least two parts: the column name and datatype
        if len(col_parts) == 2:
            col_name = col_parts[0].strip()  # Extract the column name
            col_datatype = col_parts[1].strip()  # Extract the column datatype
        else:
            # If no semicolon, assume only column name is provided
            col_name = col_parts[0].strip()
            col_datatype = "unknown"  # Or any other default datatype if applicable
        
        # Check for a match in the unique values file
        match = unique_values_df[
            (unique_values_df['Dataset Name'] == dataset_id) &
            (unique_values_df['Column Name'] == col_name)
        ]
        
        if not match.empty:
            # Append unique values in the requested format
            unique_values = match['Unique Values'].iloc[0]
            updated_col = f"{col_name} ({col_datatype}, unique_values: {unique_values})"
        else:
            # Keep the original column if no unique values are found
            updated_col = f"{col_name} ({col_datatype})"
        
        updated_columns.append(updated_col)
    
    # Join columns with newline separator instead of comma
    return '\n'.join(updated_columns)

# Process each dataset
for index, row in df_columns.iterrows():
    dataset_id = row['Dataset_ID']
    columns = row['Columns']
    # Update the columns with integrated unique values
    df_columns.at[index, 'Columns'] = integrate_unique_values(columns, dataset_id, df_unique_values)

# Save the updated DataFrame to a new CSV file
output_file = "updated_columns_with_unique_values.csv"
df_columns.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
