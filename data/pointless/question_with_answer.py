import pandas as pd

def match_questions(file1, file2, output_file):
    # Load the files into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure there is a 'Question' column in both files
    if 'question' not in df1.columns or 'question' not in df2.columns:
        raise ValueError("Both files must have a 'Question' column.")

    # Merge the DataFrames on the 'Question' column
    merged_df = pd.merge(df1, df2, on='question', how='inner')

    # Ensure the second file has an 'Answer' column
    if 'answer' not in df2.columns:
        raise ValueError("The second file must have an 'Answer' column.")

    # Extract the matched questions and their answers
    output_df = merged_df[['question', 'answer']]

    # Save the results to a new file
    output_df.to_csv(output_file, index=False)
    print(f"Matched questions and their answers saved to {output_file}.")

# Example usage:
file1 = 'final_wrong.csv'  # Replace with the path to your first file
file2 = 'semeval_train.csv'  # Replace with the path to your second file
output_file = 'matched_questions_answers.csv'

match_questions(file1, file2, output_file)
