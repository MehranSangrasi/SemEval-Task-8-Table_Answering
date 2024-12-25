import pandas as pd
import ast

# Read CSV file containing conversations and dataset names
data1_csv_path = 'updated_conversation_final2.csv'  # Replace with your file path
data1_df = pd.read_csv(data1_csv_path)

# Read CSV file containing column names, unique values, and dataset names
data2_csv_path = 'columns_with_less_than_10_unique_values.csv'  # Replace with your file path
data2_df = pd.read_csv(data2_csv_path)
data2_df["Unique Values"] = data2_df["Unique Values"].apply(ast.literal_eval)  # Convert string to list

# Function to add unique values to matching columns
def add_unique_values_to_columns(row):
    try:
        dataset_name = row['dataset_name']
        if pd.isna(dataset_name):
            return row['conversations']  # Skip if no dataset name is present

        # Parse the conversations to extract Dataset Columns
        conversations_list = ast.literal_eval(row['conversations'])
        for entry in conversations_list:
            if entry.get('role') == 'user':
                content = entry.get('content', '')
                if 'Dataset Columns:' in content:
                    preamble, dataset_columns_line = content.split('Dataset Columns:', 1)
                    dataset_columns_line = dataset_columns_line.strip()
                    updated_columns = []

                    # Process each column; datatype pair
                    for part in dataset_columns_line.split(', '):
                        if ';' not in part:
                            updated_columns.append(part)
                            continue

                        column_name = part.split(';')[0].strip()
                        column_type = part.split(';')[1].strip()

                        # Match dataset name and column name in data2
                        matched_row = data2_df[(data2_df['Dataset Name'] == dataset_name) &
                                               (data2_df['Column Name'] == column_name)]
                        if not matched_row.empty:
                            unique_values = matched_row.iloc[0]['Unique Values']
                            # Format unique values as a clean list
                            unique_values_str = ', '.join(map(str, unique_values))
                            updated_columns.append(f"{column_name} ({column_type}, unique_values: [{unique_values_str}])")
                        else:
                            updated_columns.append(f"{column_name} ({column_type})")

                    # Update the content with modified Dataset Columns
                    updated_content = preamble + 'Dataset Columns: ' + ', '.join(updated_columns)
                    entry['content'] = updated_content

        return str(conversations_list)
    except Exception as e:
        print(f"Error processing row: {e}")
        return row['conversations']

# Apply the function to update conversations
data1_df['conversations'] = data1_df.apply(add_unique_values_to_columns, axis=1)

# Save the updated data back to a new CSV file
updated_csv_path = 'final_updated_conversation.csv'  # Save to a new file
data1_df.to_csv(updated_csv_path, index=False)
print(f"Updated data saved to {updated_csv_path}")
