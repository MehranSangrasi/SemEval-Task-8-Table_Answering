import pandas as pd
import ast

# Read CSV file containing conversations
data1_csv_path = 'conversation_final2.csv'  # Replace with your file path
data1_df = pd.read_csv(data1_csv_path)

# Read CSV file containing questions and dataset names
data2_csv_path = 'semeval_train.csv'  # Replace with your file path
data2_df = pd.read_csv(data2_csv_path)

# Extract question from the 'conversations' column
def extract_question(conversations):
    try:
        conversations_list = ast.literal_eval(conversations)  # Parse the string to a list of dictionaries
        for entry in conversations_list:
            if entry.get('role') == 'user':  # Look for the 'user' role
                content = entry.get('content', '')
                if content.startswith("Question:"):
                    return content.split("\n")[0].replace("Question:", "").strip()
        return None
    except Exception as e:
        print(f"Error extracting question: {e}")
        return None

# Extract questions into a temporary column
data1_df['extracted_question'] = data1_df['conversations'].apply(extract_question)

# Match questions and add dataset name
def match_questions_and_add_dataset(row):
    question = row['extracted_question']
    matched_row = data2_df[data2_df['question'] == question]  # Match the question

    if not matched_row.empty:
        dataset_name = matched_row.iloc[0]['dataset']  # Assuming the column in data2_csv is 'dataset'
        return dataset_name
    return None  # Return None if no match is found

# Add the dataset name to the first CSV
data1_df['dataset_name'] = data1_df.apply(match_questions_and_add_dataset, axis=1)

# Remove the temporary extracted question column
data1_df = data1_df.drop(columns=['extracted_question'])

# Save the updated data back to a new CSV file
updated_csv_path = 'updated_conversation_final2.csv'  # Save to a new file to avoid overwriting
data1_df.to_csv(updated_csv_path, index=False)
print(f"Updated data saved to {updated_csv_path}")