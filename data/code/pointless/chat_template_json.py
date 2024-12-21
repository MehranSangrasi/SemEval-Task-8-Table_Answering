import pandas as pd
import json

# Load the CSV file
data = pd.read_csv("data/matched_data_2.csv")

# Initialize an empty list to store the conversations
conversations = []

# Function to map prompt and answer into the desired format
def map_template(prompt, answer):
    return [{"from": "human", "value": f"{prompt}"}, 
            {"from": "gpt", "value": f"{answer}"}]

# Create the conversations
for index, row in data.iterrows():
    prompt = f"Question: {row['question']} \n Dataset Columns: {row['columns']}"
    answer = row['query']
    conversation = map_template(prompt, answer)
    # Convert the list to a JSON string
    conversations.append(json.dumps(conversation))

# Save as a JSON object with a single key 'conversations'
output = {"conversations": conversations}

with open("data/conversations.json", "w", encoding="utf-8") as json_file:
    json.dump(output, json_file, indent=2, ensure_ascii=False)

print("Conversations have been saved to 'data/conversations.json'.")
