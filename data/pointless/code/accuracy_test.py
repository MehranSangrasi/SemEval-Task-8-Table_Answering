import pandas as pd

# File paths for the two CSVs
csv1_path = './data/queries_with_answers.csv'
csv2_path = './data/queries_with_answers_final_4.csv'

# Specify the column to compare
actual_answer = 'actual_answer'
our_answer = 'answers'


# Read the first 45 rows from the first CSV
df1 = pd.read_csv(csv1_path)

# Read the second CSV completely
df2 = pd.read_csv(csv2_path)

# Check if the column exists in both files
if actual_answer not in df1.columns or our_answer not in df2.columns:
    raise ValueError(f"The column '{our_answer}' OR '{actual_answer}' must exist in both CSV files.")

# Ensure the lengths match (compare only the first 45 rows of the second CSV as well)
if len(df1[actual_answer]) > len(df2[our_answer]):
    raise ValueError("The second CSV has fewer rows than the specified number of rows in the first CSV.")

