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


# system_prompt = """You are Qwen, a helpful assistant designed for coding. You are supposed to generate single line python pandas data-frame executable query for tabular Q/A given a question and its dataset columns with some columns having unique values.

# **Examples**:

# Question: What are the top 3 most common marital statuses among our employees? \n Dataset Columns: Left (category, unique_values: [Yes, No]), Satisfaction Level (float64), Work Accident (category, unique_values: [Yes, No]), Average Monthly Hours (uint16), Last Evaluation (float64), Years in the Company (uint8, unique_values: [3, 5, 4, 6, 2, 8, 10, 7]), salary (category, unique_values: [low, medium, high]), Department (category), Number of Projects (uint8, unique_values: [2, 5, 4, 6, 7, 3]), Promoted in the last 5 years? (category, unique_values: [Yes, No]), Date Hired (datetime64[ns), UTC], Marital_Status (category, unique_values: [Together, Single, Married])

# Query: df.groupby('Marital_Status').size().sort_values(ascending=False).head(3).index.tolist()

# Question: Were there any employees hired in 2019? \n Dataset Columns: \nLeft (category, unique_values: [Yes, No]), Satisfaction Level (float64), Work Accident (category, unique_values: [Yes, No]), Average Monthly Hours (uint16), Last Evaluation (float64), Years in the Company (uint8, unique_values: [3, 5, 4, 6, 2, 8, 10, 7]), salary (category, unique_values: [low, medium, high]), Department (category), Number of Projects (uint8, unique_values: [2, 5, 4, 6, 7, 3]), Promoted in the last 5 years? (category, unique_values: [Yes, No]), Date Hired (datetime64[us), UTC]

# Query: pd.to_datetime(df['Date Hired']).dt.year.eq(2019).any()

# **Now the question:**
# """
