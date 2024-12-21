import pandas as pd
import ast

# Load the dataset
input_file = "./data/conversations.csv"
data = pd.read_csv(input_file)

# Function to process each conversation

def process_conversation(conversation):
    conversation = ast.literal_eval(conversation)  # Convert string representation to a Python object
    user_content = "".join([c['value'] for c in conversation if c['from'] == 'human'])
    assistant_content = "".join([c['value'] for c in conversation if c['from'] == 'gpt'])
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

# Process each row and create a new DataFrame
new_data = []
for _, row in data.iterrows():
    conversations = process_conversation(row['conversations'])
    new_data.append({"conversations": conversations})

# Convert the processed data to a DataFrame
new_conversations_df = pd.DataFrame(new_data)

# Save the new DataFrame to a CSV file
output_file = "data/conversations_final.csv"
new_conversations_df.to_csv(output_file, index=False)
