import pandas as pd

# Load the data to process
data = pd.read_csv("data/correct_from_wrong.csv")

new_data = []

def map_template(prompt, answer):
    return [
        {"role": "user", "content": f"{prompt}"},
        {"role": "assistant", "content": f"{answer}"},
    ]

# Process each row and map it into the desired structure
for index, row in data.iterrows():
    prompt = f"Question: {row['question']} \n Dataset Columns: {row['columns']}"
    answer = row['query']
    conversation = map_template(prompt, answer)
    new_data.append({"conversations": conversation})

# Convert new data into a DataFrame
new_conversations_df = pd.DataFrame(new_data)

# Append to the existing CSV file
# new_conversations_df.to_csv(
#     "data/conversations.csv",
#     mode='a',  # Append mode
#     header=False,  # Don't write the header again
#     index=False  # Avoid writing the index column
# )

new_conversations_df.to_csv("data/conversations_final.csv", index=False)
