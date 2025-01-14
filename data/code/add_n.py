import re
import pandas as pd

# Load the CSV
csv_path = "updated_datasets_with_unique_values.csv"  # Replace with your file path
df = pd.read_csv(csv_path)

# Function to replace commas separating columns with \n
def format_columns(column_str):
    # Match commas that separate columns but ignore those within unique_values
    formatted_str = re.sub(r'\), ', r')\n', column_str)  # Replace ", " after ")" with "\n"
    return formatted_str

# Apply the transformation to the 'columns' column
df["columns"] = df["columns"].apply(format_columns)

# Save the updated CSV
output_path = "formatted_data.csv"  # Replace with your desired output file path
df.to_csv(output_path, index=False)

# Display a sample row to verify
print(df.head())
