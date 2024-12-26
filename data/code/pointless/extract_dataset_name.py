import pandas as pd
import ast

# Read CSV file containing conversations
data1_csv_path = 'eval_conversations.csv'  # Replace with your file path
data1_df = pd.read_csv(data1_csv_path)

# Read CSV file containing questions and dataset names
data2_csv_path = 'dev_set.csv'  # Replace with your file path
data2_df = pd.read_csv(data2_csv_path)

# Extract question from the 'conversations' column
def extract_question(conversations):
    try:
        # Parse the string to a dictionary
        conversations_dict = ast.literal_eval(conversations)
        if conversations_dict.get('role') == 'user':  # Look for the 'user' role
            content = conversations_dict.get('content', '')
            if content.startswith("Question:"):
                return content.split("\n")[0].replace("Question:", "").strip()
        return None
    except Exception as e:
        print(f"Error extracting question: {e}")
        return None

# Match questions and add dataset name
def match_questions_and_add_dataset(conversations):
    try:
        # Extract the question from conversations
        question = extract_question(conversations)
        matched_row = data2_df[data2_df['question'] == question]  # Match the question

        if not matched_row.empty:
            return matched_row.iloc[0]['dataset']  # Assuming the column in data2_csv is 'dataset'
        return None
    except Exception as e:
        print(f"Error matching questions: {e}")
        return None

# Add the dataset name directly to the conversations column
data1_df['dataset_name'] = data1_df['conversations'].apply(match_questions_and_add_dataset)

# Save the updated data back to a new CSV file
updated_csv_path = 'dev_conversation_final2.csv'  # Save to a new file to avoid overwriting
data1_df.to_csv(updated_csv_path, index=False)
print(f"Updated data saved to {updated_csv_path}")
