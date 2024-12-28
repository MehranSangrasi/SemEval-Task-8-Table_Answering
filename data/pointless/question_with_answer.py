import pandas as pd

def match_questions(file1, file2, output_file):
    # Load the files into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure there is a 'question' column in both files
    if 'question' not in df1.columns or 'question' not in df2.columns:
        raise ValueError("Both files must have a 'question' column.")

    # Merge the DataFrames on the 'question' column
    merged_df = pd.merge(df1, df2, on='question', how='inner')

    # Define the columns to extract from the second file
    required_columns = ['question', 'answer', 'type', 'columns_used', 'column_types', 'sample_answer', 'dataset']
    
    # Check if all required columns are in the second file
    missing_columns = [col for col in required_columns if col not in df2.columns]
    if missing_columns:
        raise ValueError(f"The second file is missing these required columns: {missing_columns}")

    # Extract the matched questions and their associated information
    output_df = merged_df[required_columns]

    # Save the results to a new file
    output_df.to_csv(output_file, index=False)
    print(f"Matched questions and their associated details saved to {output_file}.")

# Example usage:
file1 = 'final_wrong.csv'  # Replace with the path to your first file
file2 = 'semeval_train.csv'  # Replace with the path to your second file
output_file = 'matched_questions_answers.csv'

match_questions(file1, file2, output_file)
