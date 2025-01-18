import pandas as pd
import ast
from datasets import load_dataset

# Load the dataset
dev = pd.read_csv("data/dev_set.csv")
table_info = pd.read_csv("data/table_info.csv")

# Function to process each conversation
conversations = []

for index, row in dev.iterrows():
    content = f'Question: {row["question"]} \n Dataset Columns: {table_info.loc[table_info["Dataset_ID"] == row["dataset"]]["Columns"].values[0]}'
    print(content)
    conversation = {'role': 'user', 'content': content}
    conversations.append(conversation)
    
# Convert each dictionary in the list to a string
conversations_as_strings = [str(conversation) for conversation in conversations]

# Create a DataFrame with a single column
conversations_df = pd.DataFrame({'conversations': conversations_as_strings})

# Save the DataFrame to a CSV file
output_file_path = "data/eval_conversations.csv"
conversations_df.to_csv(output_file_path, index=False)

print(f"Conversations saved to {output_file_path}.")
    
